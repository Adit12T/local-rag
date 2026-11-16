"""Saved conversations (SQLite). Kept apart from the index so a rebuild never touches them."""
import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path


class ChatStore:
    def __init__(self, db_path: Path):
        self.db = sqlite3.connect(str(db_path), check_same_thread=False)
        self.lock = threading.Lock()
        with self.lock, self.db:
            self.db.execute("CREATE TABLE IF NOT EXISTS conversations (id TEXT PRIMARY KEY, title TEXT,"
                            " created REAL, updated REAL)")
            self.db.execute("CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT,"
                            " conversation TEXT, role TEXT, content TEXT, extra TEXT, time REAL)")
            self.db.execute("CREATE INDEX IF NOT EXISTS messages_conv ON messages (conversation, id)")

    def create(self, title: str) -> str:
        cid, now = uuid.uuid4().hex, time.time()
        with self.lock, self.db:
            self.db.execute("INSERT INTO conversations VALUES (?,?,?,?)", (cid, title.strip()[:80] or "Chat", now, now))
        return cid

    def exists(self, cid: str) -> bool:
        with self.lock:
            return self.db.execute("SELECT 1 FROM conversations WHERE id=?", (cid,)).fetchone() is not None

    def add(self, cid: str, role: str, content: str, extra: dict | None = None):
        now = time.time()
        with self.lock, self.db:
            self.db.execute("INSERT INTO messages (conversation, role, content, extra, time) VALUES (?,?,?,?,?)",
                            (cid, role, content, json.dumps(extra or {}), now))
            self.db.execute("UPDATE conversations SET updated=? WHERE id=?", (now, cid))

    def recent(self, limit: int = 50) -> list[dict]:
        with self.lock:
            rows = self.db.execute("SELECT id, title, updated FROM conversations ORDER BY updated DESC LIMIT ?",
                                   (limit,)).fetchall()
        return [{"id": r[0], "title": r[1], "updated": r[2]} for r in rows]

    def messages(self, cid: str, last: int | None = None) -> list[dict]:
        sql = "SELECT role, content, extra FROM messages WHERE conversation=? ORDER BY id"
        with self.lock:
            rows = self.db.execute(sql, (cid,)).fetchall()
        rows = rows[-last:] if last else rows
        return [{"role": r[0], "content": r[1], **json.loads(r[2] or "{}")} for r in rows]

    def delete(self, cid: str):
        with self.lock, self.db:
            self.db.execute("DELETE FROM messages WHERE conversation=?", (cid,))
            self.db.execute("DELETE FROM conversations WHERE id=?", (cid,))
