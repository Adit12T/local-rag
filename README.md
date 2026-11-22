# Local RAG

Chat with the files in folders you choose. Everything runs on your machine
(Ollama + Chroma + SQLite), and the index updates itself when files change.

## Features

**Retrieval quality**
- **Hybrid search**: BM25 keyword search (SQLite FTS5) and vector search (Chroma), merged with
  reciprocal rank fusion. Exact terms such as error codes, class names and IDs are found as
  reliably as paraphrased questions.
- **Optional cross-encoder reranking** (FlashRank, ONNX, no PyTorch) over the fused candidates.
- **Follow-up rewriting**: "and on Windows?" is rewritten into a standalone search query using
  the conversation, and the rewritten query is shown in the UI.
- **"Not found" instead of guessing**: when nothing retrieved looks relevant, the model is not
  called and the closest matches are listed.
- **Structure-aware chunking**: Markdown by heading, Python by function/class, PDF by page, Word
  by heading style. Each chunk is embedded with a `File: path › Section` header so it keeps its context.
- **Correct embedding prefixes** for nomic-embed-text, mxbai-embed-large and qwen3-embedding.
- **Filters** by folder and file type, and at most `max_per_file` chunks from one file.

**Indexing**
- Reads only the paths you allow; symlinks that lead outside them are not followed.
- Automatic updates on create, edit, delete, rename and move, plus a sync at every start and a periodic rescan.
- Content-hash change detection and an embedding cache: unchanged text is never re-embedded,
  so renames and restarts are cheap.
- The keyword index and file manifest change in one transaction; an interrupted update is repaired on the next sync.

**Safety and operations**
- Sources are passed to the model as delimited, untrusted data, with instructions not to follow anything written inside them.
- Host-header check (blocks DNS-rebinding attacks on the localhost API), optional access token,
  and a refusal to listen on a non-loopback address without a token.
- Query log (`rag_data/queries.jsonl`) with the retrieved chunks and per-stage scores and timings.
- `python -m rag.eval` measures hit@k and MRR per retrieval mode on your own questions.

## Setup

1. Install [Ollama](https://ollama.com) and pull the two models:

   ```
   ollama pull nomic-embed-text
   ollama pull llama3.2
   ```

2. Install the Python dependencies (Python 3.10+):

   ```
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   pip install -r requirements-rerank.txt   # optional: reranker (downloads a ~34 MB model once)
   ```

3. Edit `config.yaml` and list the folders/files it may read under `paths:`.

4. Run it and open http://127.0.0.1:8000

   ```
   python -m rag
   ```

## Run it with Docker

Needs only Docker (Docker Desktop on macOS/Windows). Ollama runs in its own container and the
two models are pulled on the first start (a few GB, kept in a volume).

```
RAG_DOCS=~/Documents/notes docker compose up -d
```

Open http://127.0.0.1:8000. Follow the first start with `docker compose logs -f ollama-pull rag`.

| Variable | Default | Meaning |
|---|---|---|
| `RAG_DOCS` | `./sample-docs` | Folder to chat with, mounted read-only at `/docs` |
| `RAG_CHAT_MODEL` / `RAG_EMBED_MODEL` | `llama3.2` / `nomic-embed-text` | Models to pull and use |
| `RAG_WATCH_POLLING` | `false` | Set `true` if edits on the host are not picked up (some Docker Desktop setups) |
| `RAG_AUTH_TOKEN` | empty | Set it before publishing the port beyond 127.0.0.1 |
| `RAG_PORT_PUBLISHED` | `8000` | Port on this machine |

More than one folder: add more mounts under `/docs` in `docker-compose.yml`
(for example `- ~/projects:/docs/projects:ro`); everything under `/docs` is indexed.

Variants:

```
# Use the Ollama already installed on this machine (faster on a Mac: Docker cannot use its GPU)
docker compose -f docker-compose.yml -f docker-compose.host-ollama.yml up -d rag

# NVIDIA GPU on Linux (NVIDIA Container Toolkit installed)
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d

# Image without the reranker (smaller)
RAG_RERANK_INSTALL=false docker compose build rag
```

How the container stays safe:

- Your folder is mounted read-only; the index lives in the `rag-data` volume.
- The app runs as a non-root user.
- Inside the container it listens on 0.0.0.0, but the port is published on `127.0.0.1` only, so
  nothing on your network can reach it. That is the only case `RAG_ALLOW_UNAUTHENTICATED_BIND` is
  meant for; without it the app still refuses a non-loopback address unless `auth_token` is set.
- The Host-header check still blocks DNS-rebinding attacks.

Every `config.yaml` setting can be overridden with an environment variable named `RAG_<SETTING>`
(for example `RAG_TOP_K=8`). To tune many settings, mount your own copy of `config.yaml` at
`/app/config.yaml` (commented line in `docker-compose.yml`).

`docker compose down` stops everything; `docker compose down -v` also deletes the index and the models.

## How a question is answered

```
question ──(history?)──> rewrite to standalone query
                              │
              ┌───────────────┴───────────────┐
        vector search (Chroma)        keyword search (FTS5 BM25)
              └───────────────┬───────────────┘
                 reciprocal rank fusion (top 30)
                              │
                 rerank (if installed) ──> relevant enough? ── no ──> "not found" + closest matches
                              │ yes
                 cap per file, keep top 6
                              │
                 answer with [n] citations (streamed)
```

## How access works

- Only the entries under `paths:` are read. Folders are read recursively.
- Names in `ignore:` (`.git`, `node_modules`, `target`, ...) are skipped, as are
  file types not in `extensions:` and files over `max_file_mb`.
- Removing a path from `config.yaml` removes its content from the index on the next start.
- The server listens on 127.0.0.1 only by default. To reach it from another device, set `host`
  and an `auth_token` (and add the name you browse to under `allowed_hosts`).

## How auto-update works

| What happens                         | What the RAG does                                   |
|--------------------------------------|-----------------------------------------------------|
| File created or edited               | Re-indexed ~1.5 s after it stops changing           |
| File deleted, renamed or moved       | Old chunks removed; new location indexed            |
| Folder added or deleted              | Whole subtree indexed / removed                     |
| Changes made while the app was off   | Caught by the sync that runs at every start         |
| A missed event (e.g. network drive)  | Caught by the full rescan every `rescan_minutes`    |

The **Activity** button in the UI shows recent index changes, including files with no
extractable text (for example scanned PDFs). **Rescan** forces a full check.

## Tuning

All settings are in `config.yaml`, with comments. The ones that matter most:

| Setting | Default | Notes |
|---|---|---|
| `retrieval_mode` | `hybrid` | `dense` or `keyword` to compare |
| `candidates` / `top_k` | 30 / 6 | Pool per retriever / chunks sent to the model |
| `rerank` | `auto` | Uses the reranker when `flashrank` is installed |
| `rerank_min_score`, `min_dense_score` | 0.01, 0.25 | "Not found" thresholds. Starting values: calibrate them with the eval script |
| `chunk_chars` / `code_chunk_chars` | 1200 / 2400 | Changing these rebuilds the index |
| `embed_model`, `chat_model` | `nomic-embed-text`, `llama3.2` | Any models you have pulled; a new embedding model rebuilds the index |
| `num_ctx` | 8192 | Lower it on low-memory machines |

### Measuring instead of guessing

Put 20-50 real questions in `eval/golden.yaml`, each with the file that holds the answer, then:

```
python -m rag.eval eval/golden.yaml
```

It prints hit@1 / hit@5 / hit@10 and MRR for dense, keyword, hybrid and (if installed)
hybrid + rerank, plus the lowest top score among your answerable questions, which is the
number to set the "not found" threshold below.

`GET /api/search?q=...&mode=hybrid` returns raw retrieval results with every score, for debugging.

## Limits

- Scanned PDFs are not OCR'd; they show up in Activity as "no extractable text".
- Code other than Python is split by size at blank lines, not by function.
- Chats are kept in the browser tab only and are gone on reload.
- The reranker model is English-centric.

## Test

`pip install httpx pytest && python -m pytest tests` runs chunking, indexing, hybrid search, the
watcher, the API and the security checks end to end with a fake model (no Ollama needed), plus the Docker settings: `RAG_*` overrides
and the polling watcher.

## Layout

```
config.yaml            allowed paths and settings
rag/index.py           incremental indexer, hybrid retrieval, fusion, abstention
rag/store.py           SQLite: file manifest, FTS5 keyword index, embedding cache
rag/loaders.py         text extraction and structure-aware chunking
rag/reranker.py        optional FlashRank reranker
rag/ollama_client.py   embeddings (with task prefixes), streaming chat, structured output
rag/watcher.py         file watcher, debounce, periodic rescan
rag/server.py          FastAPI app: chat, query rewriting, security, query log
rag/eval.py            golden-set retrieval evaluation
rag/static/index.html  chat UI
Dockerfile             app image (Python 3.12, non-root, healthcheck)
docker-compose*.yml    app + Ollama; host-Ollama and GPU variants
sample-docs/           what the container indexes until you set RAG_DOCS
```
