import os
import typing
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
    ocr: str = "auto"                       # auto | on | off (scanned PDF pages)

    # Corrective RAG
    corrective: bool = True                 # grade retrieval; retry weak (ambiguous) results
    crag_correct_score: float = 0.5         # reranker score at/above which retrieval counts as correct
    crag_correct_dense: float = 0.6         # same, by vector similarity, when no reranker is installed
    max_corrective_retries: int = 1
    hyde: bool = False                      # also search with a hypothetical answer passage when ambiguous
    expand_neighbors: int = 1               # add adjacent chunks from the same section
    context_chars: int = 12000              # budget for all source text sent to the model
    refine_strips: bool = True              # drop irrelevant sentences inside chunks (needs the reranker)
    strip_min_score: float = 0.01

    # Agentic RAG
    router: str = "auto"                    # auto | llm | heuristic | off
    max_sub_queries: int = 3
    iterate: str = "auto"                   # auto | on | off: extra retrieval rounds for multi-part questions
    max_rounds: int = 2
    max_context_chunks: int = 8
    agent_mode: str = "auto"                # auto (on unless the chat model is 3B or smaller) | on | off
    agent_max_steps: int = 6
    self_correct: bool = True               # verification found unsupported claims: search for them, then rewrite

    # Answer verification
    verify: str = "auto"                    # auto | llm | guardian | minicheck | off
    verify_model: str = ""                  # empty: granite4.1-guardian:8b for guardian, bespoke-minicheck for minicheck
    verify_regenerate: bool = True
    verify_threshold: float = 0.34          # regenerate once if more than this share of sentences is unsupported

    # Web search fallback (off: queries about private files would leave the machine)
    web_search: bool = False
    searxng_url: str = ""
    web_max_results: int = 3

    # Chat
    rewrite_queries: bool = True
    history_turns: int = 6
    log_queries: bool = True
    save_chats: bool = True

    # Auto-update
    debounce_seconds: float = 1.5
    rescan_minutes: float = 10
    watch_polling: bool = False             # poll for changes instead of OS events (network drives, some Docker mounts)

    # Server
    data_dir: Path = Path("./rag_data")
    host: str = "127.0.0.1"
    port: int = 8000
    auth_token: str = ""
    allowed_hosts: list[str] = field(default_factory=list)
    # Allows a non-loopback host without auth_token. Only for a container whose published
    # port is bound to 127.0.0.1 on the machine running it (see docker-compose.yml).
    allow_unauthenticated_bind: bool = False


ENV_PREFIX = "RAG_"
_TRUE, _FALSE = {"1", "true", "yes", "on"}, {"0", "false", "no", "off", ""}


def _env_value(name: str, raw: str, kind):
    if kind is bool:
        if raw.strip().lower() in _TRUE:
            return True
        if raw.strip().lower() in _FALSE:
            return False
        raise SystemExit(f"{ENV_PREFIX}{name.upper()} must be true or false, not {raw!r}")
    if kind in (int, float):
        try:
            return kind(raw)
        except ValueError:
            raise SystemExit(f"{ENV_PREFIX}{name.upper()} must be a number, not {raw!r}") from None
    return raw


def env_overrides(environ=None) -> dict:
    """Settings from RAG_<SETTING> environment variables, e.g. RAG_OLLAMA_URL or RAG_PORT.

    List settings (RAG_PATHS, RAG_EXTENSIONS, RAG_IGNORE, RAG_ALLOWED_HOSTS) are comma-separated.
    Used by the Docker image; they win over config.yaml.
    """
    environ = os.environ if environ is None else environ
    hints = typing.get_type_hints(Config)
    out = {}
    for name in [f.name for f in fields(Config)]:
        raw = environ.get(ENV_PREFIX + name.upper())
        if raw is None:
            continue
        kind = hints[name]
        if name in ("paths", "extensions", "ignore", "allowed_hosts"):
            out[name] = [v.strip() for v in raw.split(",") if v.strip()]
        elif name == "data_dir":
            out[name] = raw
        elif kind in (bool, int, float, str):
            out[name] = _env_value(name, raw, kind)
        else:  # str | None
            out[name] = raw or None
    return out


def load_config(path: str = "config.yaml") -> Config:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    raw.update(env_overrides())
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
    # YAML 1.1 reads an unquoted `on`/`off` as a boolean; map it back to the mode string.
    for name in ("rerank", "ocr", "iterate", "router", "verify", "agent_mode"):
        if isinstance(raw.get(name), bool):
            raw[name] = "on" if raw[name] else "off"
    cfg = Config(paths=paths, extensions=exts, ignore=ignore, **raw)
    if cfg.retrieval_mode not in ("hybrid", "dense", "keyword"):
        raise SystemExit("retrieval_mode must be hybrid, dense or keyword")
    for name in ("rerank", "ocr", "iterate", "agent_mode"):
        if getattr(cfg, name) not in ("auto", "on", "off"):
            raise SystemExit(f"{name} must be auto, on or off")
    if cfg.router not in ("auto", "llm", "heuristic", "off"):
        raise SystemExit("router must be auto, llm, heuristic or off")
    if cfg.verify not in ("auto", "llm", "guardian", "minicheck", "off"):
        raise SystemExit("verify must be auto, llm, guardian, minicheck or off")
    if cfg.web_search and not cfg.searxng_url.startswith(("http://", "https://")):
        raise SystemExit("web_search needs searxng_url (the address of your SearXNG instance)")
    cfg.max_sub_queries = max(1, min(cfg.max_sub_queries, 5))
    cfg.max_rounds = max(1, min(cfg.max_rounds, 3))
    cfg.max_corrective_retries = max(0, min(cfg.max_corrective_retries, 2))
    cfg.agent_max_steps = max(1, min(cfg.agent_max_steps, 12))
    return cfg
