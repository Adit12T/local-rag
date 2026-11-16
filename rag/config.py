from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class Config:
    paths: list[Path]
    extensions: set[str]
    ignore: set[str]
    max_file_mb: float = 20

    # Ollama
    ollama_url: str = "http://localhost:11434"
    embed_model: str = "nomic-embed-text"
    chat_model: str = "llama3.2"
    embed_query_prefix: str | None = None   # None = pick automatically for the model
    embed_doc_prefix: str | None = None
    num_ctx: int = 8192
    temperature: float = 0.1

    # Chunking
    chunk_chars: int = 1200
    chunk_overlap: int = 150
    code_chunk_chars: int = 2400

    # Retrieval
    retrieval_mode: str = "hybrid"          # hybrid | dense | keyword
    candidates: int = 30                    # per retriever, before fusion / reranking
    rrf_k: int = 60
    top_k: int = 6
    max_per_file: int = 3
    rerank: str = "auto"                    # auto | on | off
    rerank_model: str = "ms-marco-MiniLM-L-12-v2"
    rerank_min_score: float = 0.01          # abstain below this (reranker installed)
    min_dense_score: float = 0.25           # abstain below this when there is no keyword hit (no reranker)

    # Chat
    rewrite_queries: bool = True
    history_turns: int = 6
    log_queries: bool = True

    # Auto-update
    debounce_seconds: float = 1.5
    rescan_minutes: float = 10

    # Server
    data_dir: Path = Path("./rag_data")
    host: str = "127.0.0.1"
    port: int = 8000
    auth_token: str = ""
    allowed_hosts: list[str] = field(default_factory=list)


def load_config(path: str = "config.yaml") -> Config:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    paths = [Path(p).expanduser().resolve() for p in raw.pop("paths", None) or []]
    if not paths:
        raise SystemExit(f"{path}: add at least one entry under 'paths:'")
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise SystemExit("These paths in config.yaml do not exist:\n  " + "\n  ".join(missing))
    exts = {e.lower() if e.startswith(".") else "." + e.lower() for e in raw.pop("extensions", [".txt", ".md"])}
    ignore = set(raw.pop("ignore", []))
    if "data_dir" in raw:
        raw["data_dir"] = Path(raw["data_dir"]).expanduser()
    unknown = set(raw) - {f.name for f in fields(Config)}
    if unknown:
        raise SystemExit(f"{path}: unknown setting(s): {', '.join(sorted(unknown))}")
    cfg = Config(paths=paths, extensions=exts, ignore=ignore, **raw)
    if cfg.retrieval_mode not in ("hybrid", "dense", "keyword"):
        raise SystemExit("retrieval_mode must be hybrid, dense or keyword")
    if cfg.rerank not in ("auto", "on", "off"):
        raise SystemExit("rerank must be auto, on or off")
    return cfg
