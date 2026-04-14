# EgoVault — How It Works

EgoVault is a pipeline: raw data in, searchable answers out — entirely on your machine.

---

## Overview

```
inbox/ + Gmail + exports
        │
        ▼
  Adapters → Normalize → SQLite vault
                               │
                    LLM enrichment (llama-server)
                               │
                    Chunking + Embeddings
                               │
          ┌────────── RAG retrieval ──────────┐
          │  BM25 · vector · RRF · HyDE       │
          │  Contextual Retrieval             │
          │  Sentence Window · CRAG-lite      │
          └────────────────┬──────────────────┘
                           │
                   LLM answer generation
              (terminal · browser · MCP server)
```

---

## Stages

### 1 — Ingest

Adapters read raw source files (email exports, platform dumps, local documents) and normalise every record into the same shape: sender, timestamp, body, platform, source. Records are written to `vault.db` via SHA-256 deduplication — re-ingesting the same export is always a safe no-op.

```bash
egovault scan              # ingest inbox/ folder
egovault gmail-sync        # pull Gmail incrementally
```

### 2 — Enrich

A local LLM (llama-server / llama.cpp) reads each record and produces:
- a concise **summary**
- a list of **gems** (links, decisions, action items, key facts)
- a **contextual prefix** — a one-paragraph description of where the record sits in your history, prepended to the body before embedding

Enrichment runs in the background automatically while you chat.

### 3 — Embed

Every record (and its sentence-window chunks) is converted to a dense vector by `nomic-embed-text` running inside llama-server. Vectors are stored in `vault.db` alongside the records — no external vector database needed.

Additionally, at index time, the LLM generates hypothetical questions each record could answer (**HyPE**). Those questions are also embedded, giving a second vector lane tuned for question-to-answer matching.

### 4 — Retrieve (hybrid RAG)

Every query runs four lanes in parallel, then merges results via **Reciprocal Rank Fusion (RRF)**:

| Lane | Technique | Good at |
|------|-----------|---------|
| 1 | FTS5 BM25 full-text search | exact keywords, names, dates |
| 2 | Dense cosine similarity | semantic meaning, paraphrases |
| 3 | HyPE question embeddings | "answer this question" matching |
| 4 | Sentence-window chunks | pinpointing exact passages |

A **CRAG-lite** confidence gate checks whether the top results are actually relevant. If not, it falls back to a broader search strategy before generating an answer.

Retrieved snippets are optionally reranked by a BM25 reranker (default) or a cross-encoder model (`sentence-transformers`).

### 5 — Answer

The top-ranked context is assembled and passed to the LLM with a system prompt. The model generates an answer with inline source attribution. The agent can call tools (search, filter by date/platform, get gems) in a loop before producing the final response.

---

## Storage

Everything lives in a single file: `data/vault.db` (SQLite, WAL mode).

| Table | Contents |
|-------|----------|
| `normalized_records` | One row per record — body, metadata, enrichment status |
| `records_fts` | FTS5 virtual table for BM25 full-text search |
| `record_embeddings` | Dense float32 vectors per record |
| `record_question_embeddings` | HyPE hypothetical question vectors |
| `record_chunks` | Sentence-window sub-record chunks + embeddings |
| `enrichment_results` | LLM enrichment output (summary, gems, context prefix) |
| `extracted_gems` | Structured items extracted by enrichment |

To move your entire vault: copy `data/vault.db`.

---

## Architecture layers

```
FRONTEND        terminal REPL · Streamlit web · Telegram bot · MCP server
                    ↓ AgentSession.process(user_input)
AGENT CORE      intent detection · slash commands · tool executor · agent loop
                    ↓ retrieve() / enrich()
PROCESSING      rag.py (4-lane RRF) · enrichment.py (LLM pipeline)
                    ↓ VaultStore queries
DATA            store.py · schema.py · adapter.py · registry.py
                    ↑ Iterator[NormalizedRecord]
INPUT           adapters: gmail_imap · local_inbox · telegram_export · …
```

---

## LLM backend

EgoVault uses [llama-server](https://github.com/ggml-org/llama.cpp) (llama.cpp) as its default backend — a lightweight local inference server with an OpenAI-compatible API. With `manage = true` in `egovault.toml` (the default), EgoVault starts and stops llama-server automatically and downloads the GGUF model on first run.

Any OpenAI-compatible endpoint works as a drop-in replacement: set `[llm] provider = "openai"` with a `base_url` and `api_key` to point at OpenAI, LM Studio, vLLM, Groq, or any compatible service.

See [docs/installation.md](installation.md) for llama-server setup and VRAM sizing.
