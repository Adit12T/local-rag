# Local RAG

Chat with your documents: upload them in the browser, or point the app at folders you choose.
Everything runs on your machine (Ollama + Chroma + SQLite), and the index updates itself when
files change.

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
- **Structure-aware chunking**: Markdown by heading, Python by function/class, Java, Kotlin,
  JS/TS, C#, Go, C/C++ and Rust by type and method (`class OrderService › cancelExpiredOrders()`),
  PDF by page, Word by heading style. Each chunk is embedded with a `File: path › Section`
  header so it keeps its context.
- **Optional OCR** for scanned PDF pages (RapidOCR, fully local; only pages without a text layer are OCR'd).
- **Correct embedding prefixes** for nomic-embed-text, mxbai-embed-large and qwen3-embedding.
- **Filters** by folder and file type, and at most `max_per_file` chunks from one file.
- **Small-to-big context**: a matched chunk pulls in its neighbours from the same section
  (`expand_neighbors`), and long prose chunks are trimmed to the sentences that match the question
  when the reranker is installed (`refine_strips`). A character budget (`context_chars`) caps the prompt.

**Corrective and agentic retrieval**
- **Corrective RAG**: every search is graded `correct`, `ambiguous` or `incorrect` from the reranker
  score (or vector similarity when no reranker is installed). Ambiguous results trigger one retry with
  an alternative query (and HyDE if `hyde: true`). Incorrect results are not sent to the model at all:
  the answer is "not found" and the closest matches are only shown.
- **Query planning**: simple questions use a heuristic route; comparisons and multi-part questions are
  split into up to `max_sub_queries` sub-queries (structured output, with a heuristic fallback). Small
  chat models (3B and below, e.g. `llama3.2`) keep the LLM planner and verifier off under `auto`.
- **Agentic RAG** (`agent_mode: auto`, the default): after the first graded search, the chat model
  decides what else it needs. It can call read-only tools (`search`, `keyword_search`, `read_file`,
  `list_folder`) and calls `finish` when it has enough, for up to `agent_max_steps` steps. It sees the
  first search's results and grade, and the sub-questions from the planner.
  - Tools are limited to your configured folders and to the folder and file-type filters set in the UI.
  - The "not found" decision comes from the grades of the searches only. A file the agent read does not
    make an unrelated question answerable.
  - The agent only collects sources; the answer is still written by the normal grounded prompt.
  - `auto` turns the agent on unless the chat model is 3B or smaller. If the model cannot call tools,
    the app falls back to the standard pipeline and stops trying.
- **Iterative retrieval** (when the agent is off): the model judges whether the evidence is sufficient;
  if not, it searches again for what is missing, up to `max_rounds`, and stops when a round adds nothing.
- **Self-correction** (`self_correct: true`): when verification finds unsupported statements, the app
  searches for each of them (up to 3), adds what it finds to the sources, and rewrites the answer once.

**Answer verification**
- Each answer sentence is checked against the sources it cites: unsupported sentences and wrong
  citation numbers are flagged in the UI. Judged by the chat model (`verify: llm`), by IBM Granite
  Guardian through Ollama (`verify: guardian`, Apache 2.0, `ollama pull granite4.1-guardian:8b`), or by
  `bespoke-minicheck` (`verify: minicheck`, non-commercial licence). At most one rewrite is attempted.

**Opt-in web search**
- Off by default. With `web_search: true` and your own SearXNG instance (`searxng_url`), the UI offers a
  button that searches the exact query you approve. Web results are untrusted and marked as such.

**Indexing**
- Reads only the paths you allow; symlinks that lead outside them are not followed.
- Automatic updates on create, edit, delete, rename and move, plus a sync at every start and a periodic rescan.
- Content-hash change detection and an embedding cache: unchanged text is never re-embedded,
  so renames and restarts are cheap.
- The keyword index and file manifest change in one transaction; an interrupted update is repaired on the next sync.

**Document chat**
- Upload documents in the chat (the **Documents** panel, or drop files anywhere on the page). They are
  indexed at once and selected, so the next question is answered from them only.
- Select one or more documents (uploaded or from your folders) to limit the chat to them. Clear the
  selection, or click **All files**, to search everything again.
- **Summarize** and **Key points** read passages spread over the whole selected document (start to end),
  not only the passages that match the words of the question.
- Uploaded files are kept in `rag_data/uploads` and can be deleted from the panel. Files in your folders
  are never deleted by the app. Set `uploads: false` to turn uploads off.

**Chat**
- Conversations are saved (`rag_data/chats.sqlite`) and can be reopened or deleted from the UI.
  Set `save_chats: false` to keep nothing.

**Safety and operations**
- Sources are passed to the model as delimited, untrusted data, with instructions not to follow anything written inside them.
- Host-header check (blocks DNS-rebinding attacks on the localhost API), optional access token,
  and a refusal to listen on a non-loopback address without a token.
- Same-origin check on every write request: a web page you visit cannot upload documents to this
  server or send it questions.
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
   pip install -r requirements-ocr.txt      # optional: OCR for scanned PDFs (models are bundled)
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
question ──(history?)──> rewrite to standalone query ──> plan (simple / multi-part, sub-queries)
                                                              │
              ┌───────────────────────────────┴───────────────────────────────┐
        vector search (Chroma)                                 keyword search (FTS5 BM25)
              └───────────────────────────────┬───────────────────────────────┘
                              RRF over all queries (top 30 each)
                                              │
                 grade: correct / ambiguous / incorrect (reranker or similarity)
                    │ incorrect → "not found" + closest matches (model not called)
                    │ ambiguous → one retry (alternative query, HyDE if on)
                                              │
                 agent (default): search / read_file / list_folder ... finish
                 (agent off: sufficient? ── no ──> search for what is missing)
                                              │
                 neighbours + refined strips, cap per file, context budget
                                              │
                 answer with [n] citations (streamed)
                                              │
                 verify each sentence against its sources
                    │ problems → search for the unsupported statements → one rewrite
```

Each step is shown in the UI under "Steps", and the verification result under the answer.
Set `router`, `verify`, `iterate`, `corrective` or `agent_mode` in `config.yaml` to turn stages off.

## How access works

- Only the entries under `paths:` and the upload folder (`rag_data/uploads`) are read. Folders are
  read recursively. With uploads on, `paths:` can be empty.
- Uploads must have a type in `extensions:` and be at most `max_file_mb` (10 files per upload).
  File names are cleaned (no folders, no hidden files), and a file never overwrites another.
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
| `ocr` | `auto` | OCRs scanned PDF pages when the OCR packages are installed; installing them rebuilds the index |
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

- The corrective, planning, verification and agent features were tested with a deterministic stand-in
  for Ollama (`tests/fake_ollama.py`). Answer quality with real models, the reranker's score scale and the
  default thresholds still need checking on your own files and questions (`python -m rag.eval --answers`).
- `bespoke-minicheck` is licensed CC BY-NC (non-commercial only), so it is never chosen automatically.
  For a dedicated checker you can use commercially, choose `verify: guardian` (Granite Guardian, Apache 2.0).
- The agent and the LLM planner need a chat model that can call tools well (qwen3:4b or larger is a
  reasonable start). With the default `llama3.2` (3B), `agent_mode: auto` keeps the agent off. Set
  `agent_mode: on` to force it.
- The agent adds model calls: about one per tool step (up to `agent_max_steps`) before the answer.
- Unquoted `on`/`off` in YAML are read as booleans; the loader maps them back to the mode names.
- Without the OCR packages, scanned PDFs show up in Activity as "no extractable text". OCR covers
  PDFs only (not standalone image files) and is slow on large scans (roughly a second or more per page on CPU).
- The brace-language splitter is a lightweight scanner, not a parser. If a file's braces do not
  balance by its rules, that file falls back to size-based splitting.
- The reranker and the default OCR models are English-centric.

## Test

```
pip install -r requirements.txt pytest httpx
python -m pytest -q tests
```

`tests/test_rag.py` covers chunking, indexing, hybrid search, the watcher, the API and the security
checks. `tests/test_pipeline.py` covers grading, retries, planning, iteration, self-correction, the agent's path
limits, verification, web search and the API limits. `tests/test_documents.py` covers uploads, document
selection, summaries, deletion and the upload safety checks. `tests/test_docker_config.py` covers the Docker
settings: `RAG_*` overrides and the polling watcher. All of them run against fakes (no Ollama needed).
To exercise the whole app against the stand-in model:
`python tests/fake_ollama.py 11500` in one terminal, then point `ollama_url` at it and run `python -m rag`.

Two more checks run in CI:

- **Browser test** of saved chats (create, follow up, new chat, reload, reopen, continue, delete) in
  headless Chrome against the stand-in model: start `python tests/fake_ollama.py 11500` and
  `python -m rag -c tests/ui/config.yaml`, then `cd tests/ui && npm install && npm test`.
- **Real Ollama smoke test** (`.github/workflows/real-ollama.yml`): installs Ollama, pulls
  `nomic-embed-text`, `qwen2.5:1.5b` and `granite4.1-guardian:8b`, and checks Granite Guardian's
  verdicts, indexing with real embeddings, retrieval, a streamed answer, verification and saved chats.
  Run it on your own machine with `python tests/real_ollama_smoke.py` once those models are pulled.

## Layout

```
config.yaml            allowed paths and settings
rag/index.py           incremental indexer, hybrid retrieval, fusion, abstention
rag/store.py           SQLite: file manifest, FTS5 keyword index, embedding cache
rag/loaders.py         text extraction and structure-aware chunking (Markdown, Python, brace languages)
rag/reranker.py        optional FlashRank reranker
rag/ocr.py             optional OCR for scanned PDF pages
rag/chats.py           saved conversations
rag/ollama_client.py   embeddings (with task prefixes), streaming chat, structured output
rag/watcher.py         file watcher, debounce, periodic rescan
rag/server.py          FastAPI app: chat, query rewriting, security, query log
rag/pipeline.py        traditional, corrective and agentic flow (grading, planning, iteration, context)
rag/verify.py          sentence-level answer verification and citation checks
rag/agent.py           agentic retrieval: read-only, path- and filter-bounded tools
rag/websearch.py       opt-in SearXNG client
rag/eval.py            golden-set retrieval and answer evaluation
rag/static/index.html  chat UI
Dockerfile             app image (Python 3.12, non-root, healthcheck)
docker-compose*.yml    app + Ollama; host-Ollama and GPU variants
sample-docs/           what the container indexes until you set RAG_DOCS
```
