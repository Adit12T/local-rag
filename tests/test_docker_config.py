"""Settings used by the Docker image: RAG_* environment overrides and the polling watcher."""
import tempfile
import time
from pathlib import Path

import pytest

from rag.config import Config, env_overrides, load_config


def write_config(tmp: Path, docs: Path) -> Path:
    path = tmp / "config.yaml"
    path.write_text(f"paths: [{docs}]\nextensions: [.md]\nhost: 127.0.0.1\nport: 8000\n")
    return path


def test_env_overrides_types():
    out = env_overrides({
        "RAG_OLLAMA_URL": "http://ollama:11434",
        "RAG_PORT": "9000",
        "RAG_TEMPERATURE": "0.3",
        "RAG_WATCH_POLLING": "true",
        "RAG_ALLOW_UNAUTHENTICATED_BIND": "1",
        "RAG_PATHS": "/docs, /more ,",
        "RAG_EMBED_QUERY_PREFIX": "",
        "UNRELATED": "x",
    })
    assert out == {
        "ollama_url": "http://ollama:11434",
        "port": 9000,
        "temperature": 0.3,
        "watch_polling": True,
        "allow_unauthenticated_bind": True,
        "paths": ["/docs", "/more"],
        "embed_query_prefix": None,
    }


def test_env_rejects_bad_values():
    with pytest.raises(SystemExit):
        env_overrides({"RAG_PORT": "eighty"})
    with pytest.raises(SystemExit):
        env_overrides({"RAG_WATCH_POLLING": "maybe"})


def test_env_wins_over_yaml(monkeypatch, tmp_path):
    docs, other = tmp_path / "docs", tmp_path / "other"
    docs.mkdir()
    other.mkdir()
    cfg_file = write_config(tmp_path, docs)
    monkeypatch.setenv("RAG_PATHS", str(other))
    monkeypatch.setenv("RAG_HOST", "0.0.0.0")
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path / "data"))
    cfg = load_config(str(cfg_file))
    assert cfg.paths == [other.resolve()]
    assert cfg.host == "0.0.0.0"
    assert cfg.data_dir == tmp_path / "data"


def test_polling_watcher_picks_up_new_file():
    from rag.index import Indexer
    from rag.watcher import Watcher
    from tests.test_rag import FakeLLM, wait_for

    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        docs = tmp / "docs"
        docs.mkdir()
        (docs / "a.md").write_text("# A\nalpha text")
        cfg = Config(paths=[docs], extensions={".md"}, ignore=set(), data_dir=tmp / "data", rerank="off",
                     debounce_seconds=0.2, rescan_minutes=0, watch_polling=True)
        indexer = Indexer(cfg, FakeLLM())
        indexer.sync()
        watcher = Watcher(indexer)
        assert type(watcher.observer).__name__ == "PollingObserver"
        watcher.start()
        try:
            (docs / "b.md").write_text("# B\nzebra crossing")
            assert wait_for(lambda: any(h["name"] == "b.md" for h in indexer.search("zebra crossing", mode="keyword")["hits"]),
                            timeout=15)
        finally:
            watcher.stop()
