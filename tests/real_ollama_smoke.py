"""Smoke test against a REAL Ollama server (not collected by pytest; run it directly).

    ollama pull nomic-embed-text && ollama pull qwen2.5:1.5b && ollama pull granite4.1-guardian:8b
    python tests/real_ollama_smoke.py

Environment: OLLAMA_URL (default http://127.0.0.1:11434), CHAT_MODEL (qwen2.5:1.5b),
EMBED_MODEL (nomic-embed-text), GUARDIAN_MODEL (granite4.1-guardian:8b; empty to skip).

Checks, in order:
1. Granite Guardian answers in a format we parse, and flags unsupported claims (verify: guardian).
2. The whole app on real models: indexing with real embeddings, retrieval of the right file, a
   streamed answer that contains the fact, answer verification, and the conversation being saved.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from rag.config import Config  # noqa: E402
from rag.ollama_client import Ollama  # noqa: E402
from rag.server import create_app  # noqa: E402

URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
CHAT = os.environ.get("CHAT_MODEL", "qwen2.5:1.5b")
EMBED = os.environ.get("EMBED_MODEL", "nomic-embed-text")
GUARDIAN = os.environ.get("GUARDIAN_MODEL", "granite4.1-guardian:8b")

DOC = """# Orion Billing service

Orion Billing is owned by team Kestrel. The service listens on port 8431 and stores invoices in
PostgreSQL 16. Failed payment-gateway calls are retried three times, starting with a 250 ms backoff.
The nightly reconciliation job runs at 02:30 UTC.
"""
CLAIMS = [  # (claim, supported by DOC?)
    ("The Orion Billing service listens on port 8431.", True),
    ("Orion Billing is owned by team Kestrel.", True),
    ("The nightly reconciliation job runs at 02:30 UTC.", True),
    ("Orion Billing stores invoices in MongoDB.", False),
    ("The service listens on port 9090.", False),
    ("Orion Billing is owned by the payments platform team in Berlin.", False),
]

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'} {name}" + (f" — {detail}" if detail else ""), flush=True)
    if not ok:
        failures.append(name)


def guardian_checks(llm: Ollama) -> None:
    correct = 0
    for claim, truth in CLAIMS:
        t0 = time.time()
        try:
            got = llm.guardian_check(GUARDIAN, DOC, claim)
        except Exception as e:  # noqa: BLE001
            check(f"guardian parses a verdict for: {claim}", False, str(e))
            continue
        correct += got == truth
        print(f"   {'supported' if got else 'UNSUPPORTED':<11} (expected {'supported' if truth else 'unsupported'}) "
              f"{time.time() - t0:5.1f}s  {claim}", flush=True)
    check("guardian verdicts parse and match", correct >= len(CLAIMS) - 1, f"{correct}/{len(CLAIMS)} correct")


def app_checks(llm: Ollama, verify: str) -> None:
    tmp = Path(tempfile.mkdtemp())
    docs = tmp / "docs"
    docs.mkdir()
    (docs / "orion-billing.md").write_text(DOC)
    (docs / "team-handbook.md").write_text("# Team handbook\n\nThe daily standup is at 09:45. "
                                           "The on-call rotation changes every Monday.\n")
    cfg = Config(paths=[docs], extensions={".md"}, ignore=set(), data_dir=tmp / "data", ollama_url=URL,
                 embed_model=EMBED, chat_model=CHAT, rerank="off", ocr="off", rescan_minutes=0, verify=verify)
    app = create_app(cfg, llm)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        for _ in range(120):
            status = client.get("/api/status").json()
            if status["files"] == 2 and not status["busy"]:
                break
            time.sleep(1)
        check("real embeddings index both files", status["files"] == 2, f"{status['files']} files, {status['chunks']} chunks")
        errors = [e for e in status.get("events", []) if e.get("action") == "error"]
        check("indexing reported no errors", not errors, json.dumps(errors)[:300])

        t0 = time.time()
        r = client.post("/api/chat", json={"question": "Which port does the Orion Billing service listen on?"})
        events = [json.loads(line) for line in r.text.splitlines() if line.strip()]
        text = "".join(e.get("text", "") for e in events if e["type"] == "token")
        sources = next((e for e in events if e["type"] == "sources"), {}).get("sources", [])
        check("chat request succeeds", r.status_code == 200 and not any(e["type"] == "error" for e in events),
              f"HTTP {r.status_code}, {time.time() - t0:.1f}s")
        check("right file retrieved first", bool(sources) and "orion-billing" in sources[0].get("path", ""),
              sources[0].get("path", "") if sources else "no sources")
        check("answer contains the fact", "8431" in text, text[:160].replace("\n", " "))
        print(f"   answer: {text[:300]!r}", flush=True)
        if verify != "off":
            v = [e for e in events if e["type"] == "verification"]
            check(f"answer verified with '{verify}'", bool(v) and v[-1]["checked"] > 0,
                  json.dumps({k: v[-1][k] for k in ("checked", "unsupported", "citation_errors")}) if v else "no verification event")
        cid = next((e["id"] for e in events if e["type"] == "conversation"), None)
        saved = client.get(f"/api/conversations/{cid}").json() if cid else []
        check("conversation saved", len(saved) == 2 and saved[1]["role"] == "assistant", f"{len(saved)} messages")


def main() -> int:
    llm = Ollama(URL, EMBED, CHAT, None, None, 8192, 0.1)
    print(f"Ollama {URL}  chat={CHAT}  embed={EMBED}  guardian={GUARDIAN or '(skipped)'}\n", flush=True)
    if GUARDIAN:
        guardian_checks(llm)
    app_checks(llm, verify="guardian" if GUARDIAN else "llm")
    print(f"\n{'ALL PASSED' if not failures else 'FAILED: ' + ', '.join(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
