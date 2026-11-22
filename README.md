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
- **Iterative retrieval**: after retrieval the model judges whether the evidence is sufficient; if not,
  it searches again for what is missing, up to `max_rounds`, and stops as soon as a round adds nothing.
- **Optional tool agent** (`agent_mode: true`, experimental): the model may call read-only tools
  (`search`, `keyword_search`, `read_file`, `list_folder`) for up to `agent_max_steps` steps. Tools are
  limited to your configured folders, and the agent only collects sources; the answer is still written
  by the normal grounded prompt.

**Answer verification**
- Each answer sentence is checked against the sources it cites: unsupported sentences and wrong
  citation numbers are flagged in the UI. Judged by the chat model (`verify: llm`) or by
  `bespoke-minicheck` through Ollama (`verify: minicheck`). At most one rewrite is attempted.

**Opt-in web search**
- Off by default. With `web_search: true` and your own SearXNG instance (`searxng_url`), the UI offers a
  button that searches the exact query you approve. Web results are untrusted and marked as such.

**Indexing**
- Reads only the paths you allow; symlinks that lead outside them are not followed.
- Automatic updates on create, edit, delete, rename and move, plus a sync at every start and a periodic rescan.
- Content-hash change detection and an embedding cache: unchanged text is never re-embedded,
  so renames and restarts are cheap.
- The keyword index and file manifest change in one transaction; an interrupted update is repaired on the next sync.

**Chat**
- Conversations are saved (`rag_data/chats.sqlite`) and can be reopened or deleted from the UI.
  Set `save_chats: false` to keep nothing.

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
   pip install -r requirements-ocr.txt      # optional: OCR for scanned PDFs (models are bundled)
   ```

3. Edit `config.yaml` and list the folders/files it may read under `paths:`.

4. Run it and open http://127.0.0.1:8000

   ```
   python -m rag
   ```

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
                 sufficient? ── no ──> search for what is missing (max_rounds)
                                              │ yes
                 neighbours + refined strips, cap per file, context budget
                                              │
                 answer with [n] citations (streamed)
                                              │
                 verify each sentence against its sources ── problems ──> one rewrite
```

Each step is shown in the UI under "Steps", and the verification result under the answer.
Set `router`, `verify`, `iterate`, `corrective` or `agent_mode` in `config.yaml` to turn stages off.

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
- `bespoke-minicheck` is licensed CC BY-NC; check that license before commercial use.
- The agent mode and LLM planner need a capable chat model (qwen3:4b or larger is a reasonable start).
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
checks. `tests/test_pipeline.py` covers grading, retries, planning, iteration, the agent's path
limits, verification, web search and the API limits. Both run against fakes (no Ollama needed).
To exercise the whole app against the stand-in model:
`python tests/fake_ollama.py 11500` in one terminal, then point `ollama_url` at it and run `python -m rag`.

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
rag/agent.py           experimental read-only tool agent (path-bounded)
rag/websearch.py       opt-in SearXNG client
rag/eval.py            golden-set retrieval and answer evaluation
rag/static/index.html  chat UI
```
