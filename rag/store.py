"""SQLite side of the index: file manifest, FTS5 keyword index, embedding cache.

A file's manifest row and its keyword rows change in one transaction, so the
keyword index can never disagree with the manifest. The `pending` flag covers
the vector store, which is a separate system: it is set before Chroma is
updated and cleared after, so a crash in between is repaired on the next sync.
"""
import hashlib
import re
import sqlite3
import threading
import time
from array import array
from pathlib import Path

STOPWORDS = set("a an and are as at be but by do does for from how i in is it of on or that the this to was "
                "what when where which who why will with you your".split())


def fts_query(text: str, max_terms: int = 32) -> str:
    """Raw user text -> a safe FTS5 query (quoted terms joined with OR)."""
    terms = [t for t in re.findall(r"\w+", text.lower()) if t not in STOPWORDS and len(t) > 1]
    return " OR ".join(f'"{t}"' for t in dict.fromkeys(terms[:max_terms]))


class Store:
    def __init__(self, db_path: Path):
        self.db = sqlite3.connect(str(db_path), check_same_thread=False)
        self.lock = threading.RLock()
        with self.lock, self.db:
            self.db.execute("CREATE TABLE IF NOT EXISTS files (path TEXT PRIMARY KEY, mtime REAL, size INTEGER,"
                            " sha TEXT, chunks INTEGER, indexed_at REAL, pending INTEGER DEFAULT 0)")
            self.db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
            self.db.execute("CREATE TABLE IF NOT EXISTS embed_cache (key TEXT PRIMARY KEY, vec BLOB, used REAL)")
        try:
            with self.lock, self.db:
                self.db.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, heading, name,"
                    " chunk_id UNINDEXED, path UNINDEXED, root UNINDEXED, ext UNINDEXED, ord UNINDEXED,"
                    " tokenize='porter unicode61 remove_diacritics 2')")
            self.fts = True
        except sqlite3.OperationalError:  # this Python's SQLite was built without FTS5
            self.fts = False

    # ---------- manifest ----------

    def get(self, path: str):
        """(mtime, size, sha) for a fully indexed file, else None."""
        with self.lock:
            return self.db.execute("SELECT mtime, size, sha FROM files WHERE path=? AND pending=0", (path,)).fetchone()

    def touch(self, path: str, mtime: float, size: int):
        with self.lock, self.db:
            self.db.execute("UPDATE files SET mtime=?, size=? WHERE path=?", (mtime, size, path))

    def replace_file(self, path: str, mtime: float, size: int, sha: str, rows: list[dict]):
        """Atomically swap a file's keyword rows and manifest row; leaves it marked pending."""
        with self.lock, self.db:
            if self.fts:
                self.db.execute("DELETE FROM chunks_fts WHERE path=?", (path,))
                self.db.executemany(
                    "INSERT INTO chunks_fts (text, heading, name, chunk_id, path, root, ext, ord)"
                    " VALUES (:text, :heading, :name, :id, :path, :root, :ext, :ord)", rows)
            self.db.execute("INSERT OR REPLACE INTO files VALUES (?,?,?,?,?,?,1)",
                            (path, mtime, size, sha, len(rows), time.time()))

    def clear_pending(self, path: str):
        with self.lock, self.db:
            self.db.execute("UPDATE files SET pending=0 WHERE path=?", (path,))

    def delete_file(self, path: str):
        with self.lock, self.db:
            if self.fts:
                self.db.execute("DELETE FROM chunks_fts WHERE path=?", (path,))
            self.db.execute("DELETE FROM files WHERE path=?", (path,))

    def clear(self):
        with self.lock, self.db:
            self.db.execute("DELETE FROM files")
            if self.fts:
                self.db.execute("DELETE FROM chunks_fts")

    def paths(self) -> list[str]:
        with self.lock:
            return [r[0] for r in self.db.execute("SELECT path FROM files")]

    def totals(self) -> tuple[int, int]:
        with self.lock:
            return self.db.execute("SELECT COUNT(*), COALESCE(SUM(chunks),0) FROM files").fetchone()

    def meta(self, key: str, value: str | None = None) -> str | None:
        with self.lock, self.db:
            if value is not None:
                self.db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, value))
                return value
            row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
            return row[0] if row else None

    # ---------- keyword search ----------

    def keyword_search(self, query: str, limit: int, roots: list[str] | None = None,
                       exts: list[str] | None = None) -> list[dict]:
        match = fts_query(query)
        if not self.fts or not match:
            return []
        sql = ("SELECT chunk_id, path, name, heading, ord, text, bm25(chunks_fts, 1.0, 2.0, 2.0) AS rank"
               " FROM chunks_fts WHERE chunks_fts MATCH ?")
        args: list = [match]
        for column, values in (("root", roots), ("ext", exts)):
            if values:
                sql += f" AND {column} IN ({','.join('?' * len(values))})"
                args += values
        sql += " ORDER BY rank LIMIT ?"  # FTS5 bm25(): lower is better
        with self.lock:
            rows = self.db.execute(sql, args + [limit]).fetchall()
        return [{"id": r[0], "source": r[1], "name": r[2], "heading": r[3], "chunk": r[4], "text": r[5],
                 "bm25": round(-r[6], 3)} for r in rows]

    # ---------- embedding cache (unchanged text is never re-embedded) ----------

    @staticmethod
    def cache_key(signature: str, text: str) -> str:
        return hashlib.sha256((signature + "\x00" + text).encode()).hexdigest()

    def cache_get(self, keys: list[str]) -> dict[str, list[float]]:
        found = {}
        with self.lock, self.db:
            for i in range(0, len(keys), 500):
                part = keys[i : i + 500]
                marks = ",".join("?" * len(part))
                for key, blob in self.db.execute(f"SELECT key, vec FROM embed_cache WHERE key IN ({marks})", part):
                    vec = array("f")
                    vec.frombytes(blob)
                    found[key] = vec.tolist()
                self.db.execute(f"UPDATE embed_cache SET used=? WHERE key IN ({marks})", [time.time()] + part)
        return found

    def cache_put(self, items: dict[str, list[float]]):
        now = time.time()
        with self.lock, self.db:
            self.db.executemany("INSERT OR REPLACE INTO embed_cache VALUES (?,?,?)",
                                [(k, array("f", v).tobytes(), now) for k, v in items.items()])

    def cache_prune(self, max_age_days: float = 30):
        with self.lock, self.db:
            self.db.execute("DELETE FROM embed_cache WHERE used < ?", (time.time() - max_age_days * 86400,))
