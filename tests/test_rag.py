"""End-to-end test with a fake embedder/LLM, so it runs without Ollama.

    python tests/test_rag.py
"""
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from rag.config import Config  # noqa: E402
from rag.index import Indexer  # noqa: E402
from rag.loaders import markdown_sections, python_sections  # noqa: E402
from rag.server import create_app  # noqa: E402
from rag.store import fts_query  # noqa: E402
from rag.watcher import Watcher  # noqa: E402


class FakeLLM:
    """Bag-of-words 'embeddings': related texts share words, so they land close together."""
    signature = "fake"

    def __init__(self):
        self.embedded = 0
        self.rewrites = []
        self.last_messages = None

    def _vec(self, t):
        v = [0.0] * 256
        for w in re.findall(r"[a-z0-9]+", t.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 256] += 1
        norm = math.sqrt(sum(x * x for x in v)) or 1
        return [x / norm for x in v]

    def embed_documents(self, texts):
        self.embedded += len(texts)
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)

    def chat_stream(self, messages):
        self.last_messages = messages
        yield "stub answer [1]"

    def chat_json(self, messages, schema):
        self.rewrites.append(messages[-1]["content"])
        return {"query": "kafka retention orders topic"}


def wait_for(cond, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.2)
    return False


def names(result):
    return [h["name"] for h in result["hits"]]


def test_chunkers():
    md = "intro\n\n# Setup\ntext\n\n## Windows\nwin steps\n\n```\n# not a heading\n```\n\n## Linux\nlinux steps\n"
    sections = markdown_sections(md)
    assert [h for h, _ in sections] == ["", "Setup", "Setup › Windows", "Setup › Linux"], sections
    assert "# not a heading" in sections[2][1]

    py = "import os\n\nX = 1\n\n@decorator\ndef first():\n    return 1\n\nclass Thing:\n    def m(self):\n        pass\n"
    sections = python_sections(py)
    assert [h for h, _ in sections] == ["", "def first", "class Thing"], sections
    assert sections[1][1].startswith("@decorator")
    assert python_sections("def broken(:")[0][0] == ""  # falls back on syntax errors

    assert fts_query('what is "ERR_4012" -NEAR(') == '"err_4012" OR "near"'


def main():
    test_chunkers()

    tmp = Path(tempfile.mkdtemp()).resolve()
    docs, code, outside = tmp / "docs", tmp / "code", tmp / "outside"
    (docs / "sub").mkdir(parents=True)
    (docs / "node_modules").mkdir()
    code.mkdir()
    outside.mkdir()
    (docs / "kafka.md").write_text("# Kafka\n\n## Retention\nKafka retention is seven days for the orders topic.\n\n"
                                   "## Errors\nIf the consumer fails with ERR_4012, restart the broker.\n")
    (docs / "sub" / "db.txt").write_text("The billing database runs on Postgres 15.")
    (docs / "node_modules" / "junk.md").write_text("should be ignored")
    (docs / "image.png").write_bytes(b"\x89PNG")
    (code / "billing.py").write_text("import os\n\ndef compute_invoice_total(items):\n    return sum(items)\n\n"
                                     "def send_reminder(user):\n    pass\n")
    (outside / "secret.md").write_text("The secret launch code is tangerine.")
    os.symlink(outside / "secret.md", docs / "link.md")  # symlink escaping the allowed folder

    cfg = Config(paths=[docs, code], extensions={".md", ".txt", ".py"}, ignore={"node_modules"},
                 data_dir=tmp / "data", debounce_seconds=0.3, rescan_minutes=0, rerank="off")
    llm = FakeLLM()
    ix = Indexer(cfg, llm)
    assert ix.store.fts, "SQLite FTS5 not available in this Python"

    # 1. initial sync respects extensions, ignore list and the allowed-path boundary
    stats = ix.sync()
    assert stats["added"] == 3, stats
    assert sorted(Path(p).name for p in ix.store.paths()) == ["billing.py", "db.txt", "kafka.md"]
    assert not any("tangerine" in h["text"] for h in ix.search("secret launch code tangerine")["hits"])

    # 2. structure-aware chunks carry their heading
    top = ix.search("kafka retention orders topic")["hits"][0]
    assert (top["name"], top["heading"]) == ("kafka.md", "Kafka › Retention"), top
    top = ix.search("compute invoice total")["hits"][0]
    assert (top["name"], top["heading"]) == ("billing.py", "def compute_invoice_total"), top

    # 3. each retrieval mode works; exact identifiers are found by keywords
    for mode in ("keyword", "dense", "hybrid"):
        r = ix.search("ERR_4012", mode=mode)
        assert r["trace"]["mode"] == mode and r["hits"][0]["heading"] == "Kafka › Errors", (mode, r)
    hybrid = ix.search("ERR_4012")["hits"][0]
    assert "dense" in hybrid and "bm25" in hybrid and hybrid["rrf"] > 1 / 61, hybrid  # found by both, fused

    # 4. metadata filters
    assert set(names(ix.search("billing", roots=[str(code)]))) == {"billing.py"}
    assert set(names(ix.search("billing", exts=[".txt"]))) == {"db.txt"}

    # 5. abstain when nothing relevant is found, not when something is
    assert ix.search("zebra quantum marmalade")["abstain"] is True
    assert ix.search("kafka retention")["abstain"] is False

    # 6. a second sync re-embeds nothing; a renamed file is re-indexed from the embedding cache
    before = llm.embedded
    assert ix.sync()["unchanged"] == 3 and llm.embedded == before
    (docs / "sub" / "db.txt").rename(docs / "sub" / "database.txt")
    stats = ix.sync()
    assert stats["added"] == 1 and stats["removed"] == 1, stats
    assert llm.embedded == before + 1, "only the changed 'File:' header chunk should be embedded"

    # 7. crash between the keyword index and the vector store is repaired by the next sync
    ix.store.db.execute("UPDATE files SET pending=1 WHERE path=?", (str(docs / "kafka.md"),))
    ix.store.db.commit()
    assert ix.sync()["added"] == 1

    # 8. live updates through the watcher: modify, create, delete, new folder, folder delete
    w = Watcher(ix)
    w.start()
    time.sleep(0.5)

    def texts(q):
        return " ".join(h["text"] for h in ix.search(q)["hits"])

    (docs / "kafka.md").write_text("# Kafka\n\n## Retention\nKafka retention is now thirty days for the orders topic.\n")
    assert wait_for(lambda: "thirty" in texts("kafka retention"))
    assert "seven" not in texts("kafka retention"), "stale chunk left behind"
    assert not ix.search("ERR_4012", mode="keyword")["hits"], "stale keyword row left behind"

    (docs / "new.md").write_text("The on-call rotation changes every Tuesday.")
    assert wait_for(lambda: "new.md" in names(ix.search("on-call rotation")))

    (docs / "sub" / "database.txt").unlink()
    assert wait_for(lambda: "database.txt" not in names(ix.search("billing database postgres")))

    (docs / "team").mkdir()
    (docs / "team" / "a.md").write_text("Priya owns the claims service.")
    assert wait_for(lambda: "a.md" in names(ix.search("claims service owner")))

    shutil.rmtree(docs / "team")
    assert wait_for(lambda: "a.md" not in names(ix.search("claims service owner")))
    w.stop()

    # 9. changes made while the app was not running are caught by the next sync
    (docs / "offline.md").write_text("Written while the watcher was stopped.")
    (docs / "new.md").unlink()
    stats = ix.sync()
    assert stats["added"] == 1 and stats["removed"] == 1, stats
    files_now = ix.store.totals()[0]
    del ix, w

    # 10. HTTP API: chat, follow-up rewriting, abstention, prompt hardening, query log
    client = TestClient(create_app(cfg, llm, watch=False), base_url="http://localhost")
    assert client.get("/").status_code == 200
    assert client.get("/api/status").json()["files"] == files_now

    def ask(question, history=()):
        r = client.post("/api/chat", json={"question": question, "history": list(history)})
        return [json.loads(line) for line in r.text.splitlines()]

    lines = ask("kafka retention")
    assert lines[0]["type"] == "sources" and lines[0]["sources"][0]["name"] == "kafka.md"
    assert lines[1] == {"type": "token", "text": "stub answer [1]"} and lines[-1]["type"] == "done"
    assert not llm.rewrites, "no rewrite without history"
    prompt = llm.last_messages
    assert "untrusted data" in prompt[0]["content"] and '<source id="1"' in prompt[-1]["content"]

    lines = ask("and how long is it?", [{"role": "user", "content": "tell me about kafka"},
                                        {"role": "assistant", "content": "Kafka is used for orders."}])
    assert len(llm.rewrites) == 1 and lines[0]["query"] == "kafka retention orders topic"
    assert lines[0]["sources"][0]["name"] == "kafka.md"

    lines = ask("zebra quantum marmalade")
    assert lines[0]["abstain"] is True and "couldn't find" in lines[1]["text"]

    logged = [json.loads(l) for l in (cfg.data_dir / "queries.jsonl").read_text().splitlines()]
    assert len(logged) == 3 and logged[0]["hits"][0]["source"].endswith("kafka.md")

    # 11. security: foreign Host headers are rejected (DNS rebinding); token is enforced when set
    assert client.get("/api/status", headers={"host": "evil.example.com"}).status_code == 400
    cfg.auth_token = "s3cret"
    secured = TestClient(create_app(cfg, llm, watch=False), base_url="http://localhost")
    assert secured.get("/api/status").status_code == 401
    assert secured.get("/api/status", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert secured.get("/api/status", headers={"Authorization": "Bearer s3cret"}).status_code == 200
    assert secured.get("/").status_code == 200

    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
