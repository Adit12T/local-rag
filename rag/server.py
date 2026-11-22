"""Web chat UI + API. Starts the initial sync and the file watcher."""
import hmac
import json
import logging
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .chats import ChatStore
from .config import Config
from .index import Indexer
from .ocr import Ocr
from .pipeline import Pipeline
from .reranker import Reranker
from .watcher import Watcher

log = logging.getLogger("rag.server")
STATIC = Path(__file__).parent / "static"

class Filters(BaseModel):
    roots: list[str] = Field(default_factory=list, max_length=50)
    exts: list[str] = Field(default_factory=list, max_length=50)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: list[dict] = Field(default_factory=list, max_length=200)
    conversation_id: str | None = Field(default=None, max_length=64)
    filters: Filters = Filters()
    web_query: str | None = Field(default=None, max_length=200)  # set only after the user approves a web search


def create_app(cfg: Config, llm, watch: bool = True) -> FastAPI:
    reranker = Reranker(cfg.rerank_model, str(cfg.data_dir / "models"), cfg.rerank)
    indexer = Indexer(cfg, llm, reranker, Ocr(cfg.ocr))
    chats = ChatStore(cfg.data_dir / "chats.sqlite") if cfg.save_chats else None
    pipeline = Pipeline(cfg, llm, indexer, reranker)
    app_settings = pipeline.settings()
    watcher = Watcher(indexer)
    log_lock = threading.Lock()

    def initial_sync():
        try:
            log.info("Initial sync finished: %s", indexer.sync())
        except Exception:
            log.exception("Initial sync failed")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if watch:
            watcher.start()  # start first so nothing changed during the sync is missed
            threading.Thread(target=initial_sync, daemon=True, name="rag-sync").start()
        yield
        if watch:
            watcher.stop()

    app = FastAPI(title="Local RAG", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.indexer = indexer
    app.state.pipeline = pipeline

    # Reject requests whose Host header is not ours: blocks DNS-rebinding attacks, where a
    # web page you visit points its own domain at 127.0.0.1 to read your files through this API.
    hosts = {"localhost", "127.0.0.1", cfg.host, *cfg.allowed_hosts} - {"0.0.0.0", "::"}
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=sorted(hosts))

    @app.middleware("http")
    async def require_token(request: Request, call_next):
        if cfg.auth_token and request.url.path.startswith("/api/"):
            given = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
            if not hmac.compare_digest(given.encode(), cfg.auth_token.encode()):
                return JSONResponse({"error": "missing or wrong token"}, status_code=401)
        return await call_next(request)

    def log_query(entry: dict):
        if not cfg.log_queries:
            return
        path = cfg.data_dir / "queries.jsonl"
        with log_lock:
            if path.exists() and path.stat().st_size > 5_000_000:
                path.replace(path.with_suffix(".jsonl.1"))
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")

    @app.get("/")
    def home():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/status")
    def status():
        return {**indexer.status(), "pipeline": app_settings}

    @app.post("/api/rescan")
    def rescan():
        threading.Thread(target=indexer.sync, daemon=True).start()
        return {"started": True}

    @app.get("/api/search")
    def search(q: str, mode: str | None = None, k: int | None = None):
        """Retrieval only, with per-retriever scores: for debugging and tuning."""
        return indexer.search(q, k=k, mode=mode)

    @app.get("/api/conversations")
    def conversations():
        return chats.recent() if chats else []

    @app.get("/api/conversations/{cid}")
    def conversation(cid: str):
        if not chats or not chats.exists(cid):
            return JSONResponse({"error": "not found"}, status_code=404)
        return chats.messages(cid)

    @app.delete("/api/conversations/{cid}")
    def delete_conversation(cid: str):
        if chats:
            chats.delete(cid)
        return {"deleted": True}

    @app.post("/api/chat")
    def chat(req: ChatRequest):
        if req.web_query and not cfg.web_search:
            return JSONResponse({"error": "web search is turned off (web_search in config.yaml)"}, status_code=400)
        allowed_roots = {str(p) for p in cfg.paths}
        filters = {"roots": [r for r in req.filters.roots if r in allowed_roots] or None,
                   "exts": [e.lower() for e in req.filters.exts if e.lower() in cfg.extensions] or None}

        def stream():
            t0 = time.time()
            cid, past = req.conversation_id, req.history
            if chats and cid and chats.exists(cid):
                past = chats.messages(cid)  # the saved conversation is the source of truth
            history = [{"role": m["role"], "content": str(m.get("content", ""))[:2000]}
                       for m in past[-cfg.history_turns * 2:]
                       if isinstance(m, dict) and m.get("role") in ("user", "assistant")]
            steps, verification = [], None
            try:
                if chats:
                    if not (cid and chats.exists(cid)):
                        cid = chats.create(req.question)
                    yield json.dumps({"type": "conversation", "id": cid}) + "\n"
                result = None
                for event in pipeline.run(req.question, history, filters, req.web_query):
                    if event["type"] == "_result":
                        result = event
                        continue
                    if event["type"] == "step":
                        steps.append(event)
                    elif event["type"] == "verification":
                        verification = event
                    yield json.dumps(event) + "\n"
                if chats:
                    chats.add(cid, "user", req.question)
                    chats.add(cid, "assistant", result["answer"], {
                        "sources": result["sources"], "abstain": result["abstain"], "query": result["plan"]["query"],
                        "steps": steps, "verification": verification, "revised": result["revised"]})
                yield json.dumps({"type": "done", "llm_calls": result["llm_calls"],
                                  "ms": round((time.time() - t0) * 1000)}) + "\n"
                v = result["verification"] or {}
                log_query({"time": t0, "question": req.question, "query": result["plan"]["query"],
                           "route": result["plan"]["route"], "sub_queries": result["plan"]["sub_queries"],
                           "grade": result["grade"], "best": result["best"], "abstain": result["abstain"],
                           "steps": steps, "llm_calls": result["llm_calls"], "total_ms": result["total_ms"],
                           "verified": bool(v), "unsupported": v.get("unsupported"), "checked": v.get("checked"),
                           "citation_errors": v.get("citation_errors"), "revised": result["revised"],
                           "web": bool(req.web_query), "hits": result["hits"], "answer": result["answer"]})
            except Exception as e:
                log.exception("Chat failed")
                yield json.dumps({"type": "error", "text": str(e)}) + "\n"

        return StreamingResponse(stream(), media_type="application/x-ndjson")

    return app
