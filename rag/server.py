"""Web chat UI + API. Starts the initial sync and the file watcher."""
import hmac
import json
import logging
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, File, Request, UploadFile
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
MAX_UPLOAD_FILES = 10


def safe_name(raw: str | None) -> str:
    """A plain file name: no folders, no hidden files, only ordinary characters."""
    name = str(raw or "").replace("\\", "/").split("/")[-1]
    name = re.sub(r"[^\w.\- ()]+", "_", name).strip(" ._")[:120]
    return name

class Filters(BaseModel):
    roots: list[str] = Field(default_factory=list, max_length=50)
    exts: list[str] = Field(default_factory=list, max_length=50)
    files: list[str] = Field(default_factory=list, max_length=50)  # chat only with these documents


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: list[dict] = Field(default_factory=list, max_length=200)
    conversation_id: str | None = Field(default=None, max_length=64)
    filters: Filters = Filters()
    web_query: str | None = Field(default=None, max_length=200)  # set only after the user approves a web search


def create_app(cfg: Config, llm, watch: bool = True) -> FastAPI:
    upload_dir = None
    if cfg.uploads:  # uploaded documents live in their own allowed folder inside data_dir
        upload_dir = (cfg.data_dir / "uploads").resolve()
        upload_dir.mkdir(parents=True, exist_ok=True)
        if upload_dir not in cfg.paths:
            cfg.paths.append(upload_dir)
    reranker = Reranker(cfg.rerank_model, str(cfg.data_dir / "models"), cfg.rerank)
    indexer = Indexer(cfg, llm, reranker, Ocr(cfg.ocr))
    indexer.upload_dir = upload_dir
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

    @app.middleware("http")
    async def same_origin(request: Request, call_next):
        # A web page you visit can send a plain form POST (for example a file upload) to this server.
        # Browsers always send Origin on such requests: refuse any that do not come from this page.
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            origin = request.headers.get("origin")
            if origin is not None and urlsplit(origin).netloc != request.headers.get("host", ""):
                return JSONResponse({"error": "cross-origin request refused"}, status_code=403)
            if request.url.path == "/api/documents" and request.method == "POST":
                limit = MAX_UPLOAD_FILES * cfg.max_file_mb * 1024 * 1024 + 1024 * 1024
                size = request.headers.get("content-length")
                if not size or not size.isdigit() or int(size) > limit:
                    return JSONResponse({"error": f"upload too large (at most {cfg.max_file_mb:g} MB per file)"},
                                        status_code=413)
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

    @app.get("/api/documents")
    def documents():
        return {"documents": indexer.documents(), "uploads": bool(upload_dir), "max_mb": cfg.max_file_mb,
                "extensions": sorted(cfg.extensions)}

    @app.post("/api/documents")
    def upload(files: list[UploadFile] = File(...)):
        if not upload_dir:
            return JSONResponse({"error": "uploads are turned off (uploads in config.yaml)"}, status_code=403)
        if len(files) > MAX_UPLOAD_FILES:
            return JSONResponse({"error": f"at most {MAX_UPLOAD_FILES} files at a time"}, status_code=400)
        limit, results = int(cfg.max_file_mb * 1024 * 1024), []
        for f in files:
            name = safe_name(f.filename)
            ext = Path(name).suffix.lower()
            if not name or not Path(name).stem or ext not in cfg.extensions:
                results.append({"name": f.filename, "status": "rejected",
                                "detail": f"file type not supported ({ext or 'none'})"})
                continue
            tmp = upload_dir / f".part-{uuid.uuid4().hex}"
            size, too_big = 0, False
            try:
                with tmp.open("wb") as out:
                    while block := f.file.read(1024 * 1024):
                        size += len(block)
                        if size > limit:
                            too_big = True
                            break
                        out.write(block)
                if too_big or size == 0:
                    results.append({"name": name, "status": "rejected",
                                    "detail": "empty file" if size == 0 else f"larger than {cfg.max_file_mb:g} MB"})
                    continue
                target, n = upload_dir / name, 1
                while target.exists():
                    target, n = upload_dir / f"{Path(name).stem} ({n}){ext}", n + 1
                tmp.replace(target)
            finally:
                tmp.unlink(missing_ok=True)
            status = indexer.index_file(target)
            chunks = next((d["chunks"] for d in indexer.documents() if d["path"] == str(target)), 0)
            results.append({"name": target.name, "path": str(target), "status": status, "chunks": chunks})
            log.info("uploaded %s (%d bytes, %s)", target, size, status)
        return {"results": results}

    @app.delete("/api/documents")
    def delete_document(path: str):
        real = Path(path).resolve()
        if not upload_dir or real.parent != upload_dir or not real.is_file():
            return JSONResponse({"error": "only uploaded documents can be deleted here"}, status_code=404)
        real.unlink()
        indexer.remove_path(real)
        return {"deleted": str(real)}

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
        if req.filters.files:
            known = set(indexer.store.paths())
            filters["files"] = [f for f in req.filters.files if f in known] or None
            if not filters["files"]:
                return JSONResponse({"error": "the selected documents are no longer indexed"}, status_code=400)

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
