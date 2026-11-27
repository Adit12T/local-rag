"""Incremental index and hybrid retrieval.

Indexing: only files that are new or whose content changed are re-chunked and
re-embedded; files that disappear are removed.

Retrieval: dense vectors (Chroma) + BM25 keywords (SQLite FTS5), merged with
reciprocal rank fusion, optionally reranked by a cross-encoder, then capped
per file. `abstain` is set when nothing retrieved looks relevant.
"""
import hashlib
import logging
import os
import threading
import time
from collections import deque
from pathlib import Path

import chromadb

from .config import Config
from .loaders import SEP, build_chunks
from .store import Store

log = logging.getLogger("rag.index")

INDEX_VERSION = "3"  # bump when chunking or stored fields change, to force a rebuild
COLLECTION = "documents_v2"


def _is_under(path: str, parent: str) -> bool:
    return path == parent or path.startswith(parent.rstrip(os.sep) + os.sep)


class Indexer:
    def __init__(self, cfg: Config, embedder, reranker=None, ocr=None):
        self.cfg = cfg
        self.ocr = ocr
        self.embedder = embedder  # .embed_documents(list[str]), .embed_query(str), .signature
        self.reranker = reranker
        self.upload_dir: Path | None = None  # set by the server when uploads are on
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        self.store = Store(cfg.data_dir / "index.sqlite")
        self.client = chromadb.PersistentClient(path=str(cfg.data_dir / "chroma"))
        self.lock = threading.RLock()
        self.events: deque = deque(maxlen=50)  # recent index changes, shown in the UI
        self.busy = False
        self.last_sync: float | None = None
        if not self.store.fts and cfg.retrieval_mode != "dense":
            log.warning("SQLite FTS5 is not available in this Python build; using dense retrieval only")

        # Vectors and chunks from a different model / prefix / chunker are not comparable: rebuild.
        signature = "|".join([INDEX_VERSION, embedder.signature, str(cfg.chunk_chars), str(cfg.chunk_overlap),
                              str(cfg.code_chunk_chars), f"ocr={bool(ocr and ocr.enabled)}"])
        previous = self.store.meta("signature")
        if previous != signature:
            if previous:
                log.warning("Index settings changed; rebuilding the index")
            for name in (COLLECTION, "documents"):  # "documents" = pre-v2 collection
                try:
                    self.client.delete_collection(name)
                except Exception:
                    pass
            self.store.clear()
        self.store.meta("signature", signature)
        self.col = self.client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    # ---------- what are we allowed to read? ----------

    def _root_of(self, real: str) -> Path | None:
        return next((r for r in self.cfg.paths if _is_under(real, str(r))), None)

    def is_allowed(self, path: Path) -> bool:
        """True only for supported files that really live inside a configured path."""
        try:
            real = path.resolve()
            if not real.is_file() or real.suffix.lower() not in self.cfg.extensions:
                return False
            root = self._root_of(str(real))
            if root is None:  # outside the allowed paths (e.g. a symlink pointing elsewhere)
                return False
            if root.is_dir() and any(part in self.cfg.ignore for part in real.relative_to(root).parts):
                return False
            return real.stat().st_size <= self.cfg.max_file_mb * 1024 * 1024
        except OSError:
            return False

    def _walk(self, scope: Path):
        if scope.is_file():
            yield scope
            return
        for dirpath, dirnames, filenames in os.walk(scope, followlinks=False):
            dirnames[:] = [d for d in dirnames if d not in self.cfg.ignore]
            for name in filenames:
                yield Path(dirpath) / name

    # ---------- indexing ----------

    def _embed_documents(self, texts: list[str]) -> list[list[float]]:
        keys = [Store.cache_key(self.embedder.signature, t) for t in texts]
        cached = self.store.cache_get(keys)
        missing = {k: t for k, t in zip(keys, texts) if k not in cached}
        if missing:
            fresh = dict(zip(missing, self.embedder.embed_documents(list(missing.values()))))
            self.store.cache_put(fresh)
            cached.update(fresh)
        return [cached[k] for k in keys]

    def index_file(self, path: Path) -> str:
        """Index one file if new/changed. Returns added | updated | unchanged | skipped | removed."""
        if not self.is_allowed(path):
            return self.remove_path(path) and "removed" or "skipped"
        real = path.resolve()
        key = str(real)
        with self.lock:
            try:
                st = real.stat()
                known = self.store.get(key)
                if known and known[0] == st.st_mtime and known[1] == st.st_size:
                    return "unchanged"
                sha = hashlib.sha256(real.read_bytes()).hexdigest()
                if known and known[2] == sha:  # touched but content identical
                    self.store.touch(key, st.st_mtime, st.st_size)
                    return "unchanged"
                root = self._root_of(key)
                label = root.name if root.is_file() else f"{root.name}/{real.relative_to(root).as_posix()}"
                chunks = build_chunks(real, self.cfg.chunk_chars, self.cfg.chunk_overlap, self.cfg.code_chunk_chars,
                                      self.ocr)
                # A one-line "where am I" header is embedded with each chunk so it keeps its context.
                vectors = self._embed_documents(
                    [f"File: {label}{SEP + h if h else ''}\n\n{t}" for h, t in chunks]) if chunks else []
            except Exception as e:  # unreadable / half-written file, or Ollama down: retried on next event/rescan
                log.warning("Could not index %s: %s", key, e)
                self._event("error", key, str(e))
                return "skipped"

            prefix = hashlib.sha1(key.encode()).hexdigest()[:20]
            rows = [{"id": f"{prefix}:{i}", "text": t, "heading": h, "name": real.name, "path": key,
                     "root": str(root), "ext": real.suffix.lower(), "ord": i} for i, (h, t) in enumerate(chunks)]
            self.store.replace_file(key, st.st_mtime, st.st_size, sha, rows)  # marked pending
            self.col.delete(where={"source": key})
            for i in range(0, len(rows), 2000):
                part = rows[i : i + 2000]
                self.col.add(
                    ids=[r["id"] for r in part],
                    documents=[r["text"] for r in part],
                    embeddings=vectors[i : i + 2000],
                    metadatas=[{"source": key, "name": r["name"], "heading": r["heading"], "chunk": r["ord"],
                                "root": r["root"], "ext": r["ext"], "mtime": st.st_mtime} for r in part],
                )
            self.store.clear_pending(key)

            action = "updated" if known else "added"
            if not chunks:
                self._event("warning", key, "no extractable text" + (" (scanned PDF?)" if key.endswith(".pdf") else ""))
            else:
                self._event(action, key)
            log.info("%s %s (%d chunks)", action, key, len(chunks))
            return action

    def remove_path(self, path: Path) -> int:
        """Drop a deleted file, or everything under a deleted folder. Returns files removed."""
        key = str(Path(os.path.abspath(path)))
        removed = 0
        with self.lock:
            for known in self.store.paths():
                if _is_under(known, key):
                    self.col.delete(where={"source": known})
                    self.store.delete_file(known)
                    self._event("removed", known)
                    log.info("removed %s", known)
                    removed += 1
        return removed

    def sync(self, scope: Path | None = None) -> dict:
        """Bring the index in line with disk, for everything or just one subtree."""
        scopes = [scope] if scope else self.cfg.paths
        stats = {"added": 0, "updated": 0, "removed": 0, "unchanged": 0, "skipped": 0}
        self.busy = True
        try:
            for root in scopes:
                seen = set()
                if root.exists():
                    for f in self._walk(root):
                        if self.is_allowed(f):
                            seen.add(str(f.resolve()))
                            stats[self.index_file(f)] += 1
                for known in self.store.paths():
                    if _is_under(known, str(root)) and known not in seen:
                        stats["removed"] += self.remove_path(Path(known))
            if scope is None:
                # paths taken out of config.yaml are no longer allowed: forget them
                for known in self.store.paths():
                    if self._root_of(known) is None:
                        stats["removed"] += self.remove_path(Path(known))
                self.store.cache_prune()
                self.last_sync = time.time()
        finally:
            self.busy = False
        return stats

    # ---------- retrieval ----------

    def _dense(self, query: str, n: int, roots, exts, files=None) -> list[dict]:
        total = self.col.count()
        if total == 0:
            return []
        conds = [{f: {"$in": v}} for f, v in (("root", roots), ("ext", exts), ("source", files)) if v]
        where = None if not conds else conds[0] if len(conds) == 1 else {"$and": conds}
        res = self.col.query(query_embeddings=[self.embedder.embed_query(query)], n_results=min(n, total), where=where)
        return [
            {"id": i, "text": doc, "source": m["source"], "name": m["name"], "heading": m.get("heading", ""),
             "chunk": m["chunk"], "dense": round(1 - dist, 3)}
            for i, doc, m, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]

    def search(self, query: str, k: int | None = None, roots: list[str] | None = None,
               exts: list[str] | None = None, mode: str | None = None, rerank: bool = True,
               extra_queries: list[str] | None = None, dense_texts: list[str] | None = None,
               files: list[str] | None = None) -> dict:
        """Hybrid search for `query` (plus any alternative phrasings), graded CRAG-style.

        Every query contributes a dense and a keyword ranking; `dense_texts` (e.g. a HyDE passage)
        contribute dense rankings only. All rankings are fused with RRF, then reranked against
        `query`. Returns {"hits", "grade", "best", "abstain", "trace"}, where grade is
        correct | ambiguous | incorrect.
        """
        cfg, t0 = self.cfg, time.time()
        k = k or cfg.top_k
        mode = mode or cfg.retrieval_mode
        if not self.store.fts:
            mode = "dense"
        queries = list(dict.fromkeys([query, *(q for q in extra_queries or [] if q and q.strip())]))
        rankings, dense_n, keyword_n, keyword_hit = [], 0, 0, False
        for q in queries:
            if mode != "keyword":
                rankings.append(self._dense(q, cfg.candidates, roots, exts, files))
                dense_n += len(rankings[-1])
            if mode != "dense":
                rankings.append(self.store.keyword_search(q, cfg.candidates, roots, exts, files))
                keyword_n += len(rankings[-1])
                keyword_hit = keyword_hit or bool(rankings[-1])
        if mode != "keyword":
            for text in dense_texts or []:
                rankings.append(self._dense(text, cfg.candidates, roots, exts, files))

        # Reciprocal rank fusion: score = sum over rankings of 1 / (rrf_k + rank)
        pool: dict[str, dict] = {}
        for results in rankings:
            for rank, hit in enumerate(results, 1):
                entry = pool.setdefault(hit["id"], {**hit, "rrf": 0.0})
                for f in ("dense", "bm25"):
                    if f in hit:
                        entry[f] = max(entry.get(f, hit[f]), hit[f])
                entry["rrf"] += 1 / (cfg.rrf_k + rank)
        ranked = sorted(pool.values(), key=lambda h: h["rrf"], reverse=True)[: cfg.candidates]
        t_retrieve = time.time()

        scores = self.reranker.rerank(query, ranked) if (rerank and self.reranker) else None
        if scores is not None:
            for hit, s in zip(ranked, scores):
                hit["rerank"] = round(s, 4)
            ranked.sort(key=lambda h: h["rerank"], reverse=True)
            best = ranked[0]["rerank"] if ranked else 0.0
            if not ranked or best < cfg.rerank_min_score:
                grade = "incorrect"
            else:
                grade = "correct" if best >= cfg.crag_correct_score else "ambiguous"
                # per-chunk evaluation: chunks the reranker scores as irrelevant are not passed on
                ranked = [h for h in ranked if h["rerank"] >= cfg.rerank_min_score]
        else:
            best = max((h.get("dense", 0) for h in ranked), default=0.0)
            if not ranked or (not keyword_hit and best < cfg.min_dense_score):
                grade = "incorrect"
            else:
                grade = "correct" if best >= cfg.crag_correct_dense else "ambiguous"

        hits, per_file = [], {}
        # diversity: at most max_per_file chunks from one file, unless the user picked a few documents
        cap = k if files and len(files) <= 3 else cfg.max_per_file
        for hit in ranked:
            if per_file.get(hit["source"], 0) < cap:
                per_file[hit["source"]] = per_file.get(hit["source"], 0) + 1
                hit["rrf"] = round(hit["rrf"], 4)
                hit["score"] = hit.get("rerank", hit["rrf"])
                hit["query"] = query
                hits.append(hit)
            if len(hits) == k:
                break
        return {
            "hits": hits,
            "grade": grade,
            "best": round(float(best), 4),
            "abstain": grade == "incorrect",
            "trace": {"mode": mode, "reranked": scores is not None, "queries": len(queries),
                      "dense_n": dense_n, "keyword_n": keyword_n,
                      "retrieve_ms": round((t_retrieve - t0) * 1000), "rerank_ms": round((time.time() - t_retrieve) * 1000)},
        }

    def neighbors(self, hit: dict, span: int = 1) -> tuple[list[dict], list[dict]]:
        """Adjacent chunks (before, after) from the same file and section, nearest first."""
        prefix, _, ordinal = hit["id"].rpartition(":")
        try:
            ordinal = int(ordinal)
        except ValueError:
            return [], []
        wanted = [o for d in range(1, span + 1) for o in (ordinal - d, ordinal + d) if o >= 0]
        if not wanted:
            return [], []
        got = self.col.get(ids=[f"{prefix}:{o}" for o in wanted], include=["documents", "metadatas"])
        found = {}
        for cid, doc, meta in zip(got["ids"], got["documents"], got["metadatas"]):
            if meta.get("source") == hit["source"] and meta.get("heading", "") == hit.get("heading", ""):
                found[int(cid.rpartition(":")[2])] = {"id": cid, "text": doc}
        before = [found[o] for o in range(ordinal - 1, ordinal - span - 1, -1) if o in found]
        after = [found[o] for o in range(ordinal + 1, ordinal + span + 1) if o in found]
        return before, after

    # ---------- status ----------

    def documents(self) -> list[dict]:
        """Every indexed file, for the document picker."""
        out = []
        for path, size, mtime, chunks, indexed_at in self.store.documents():
            root = self._root_of(path)
            if root is None:
                continue
            out.append({"path": path, "name": Path(path).name, "size": size, "mtime": mtime, "chunks": chunks,
                        "indexed_at": indexed_at, "folder": root.name,
                        "uploaded": bool(self.upload_dir and _is_under(path, str(self.upload_dir)))})
        return out

    def overview(self, path: str, n: int) -> list[dict]:
        """Up to n chunks spread evenly over one file (always its start and end), in order.
        Used for "summarize this document", where a search for the question words finds nothing useful."""
        total = next((row[3] for row in self.store.documents() if row[0] == path), 0)
        if not total:
            return []
        n = max(1, min(n, total))
        picks = sorted({round(i * (total - 1) / (n - 1)) for i in range(n)}) if n > 1 else [0]
        prefix = hashlib.sha1(path.encode()).hexdigest()[:20]
        res = self.col.get(ids=[f"{prefix}:{i}" for i in picks], include=["documents", "metadatas"])
        rows = sorted(zip(res["ids"], res["documents"], res["metadatas"]), key=lambda r: r[2]["chunk"])
        return [{"id": i, "text": doc, "source": m["source"], "name": m["name"], "heading": m.get("heading", ""),
                 "chunk": m["chunk"], "score": 0.0, "query": "", "overview": True} for i, doc, m in rows]

    def _event(self, action: str, path: str, detail: str = ""):
        self.events.appendleft({"time": time.time(), "action": action, "path": path, "detail": detail})

    def status(self) -> dict:
        files, chunks = self.store.totals()
        exts = sorted({Path(p).suffix.lower() for p in self.store.paths()})
        return {
            "files": files,
            "chunks": chunks,
            "busy": self.busy,
            "last_sync": self.last_sync,
            "paths": [str(p) for p in self.cfg.paths],
            "extensions": exts,
            "mode": self.cfg.retrieval_mode if self.store.fts else "dense",
            "reranker": bool(self.reranker and self.reranker.enabled),
            "ocr": bool(self.ocr and self.ocr.enabled),
            "events": list(self.events)[:15],
        }
