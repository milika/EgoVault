# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

---

## [0.1.9] -- 2026-04-14

### Added
- **Telegram forward-to-vault** -- forward any Telegram message to the EgoVault bot to store it directly in your vault; no API credentials or MTProto setup required
- **Web search tool** -- the agent can search the public web via any SearXNG instance during a chat session; configure `[web_search] searxng_url` in `egovault.toml` to enable; disabled by default
- **Unified frontend dispatch** -- all interfaces (terminal, browser, Telegram) share the same agent pipeline; consistent behaviour across every way you access your vault

### Changed
- **Model default** -- switched from Gemma 4 E4B (~7.5 GB) to Gemma 4 E2B (~3 GB); fits comfortably on 12 GB GPUs with far more room for context
- **VRAM context sizing** -- context window size is now auto-computed from *free* VRAM at launch time (via `nvidia-smi`) rather than from total VRAM minus manual estimates; the model loads unrestricted and the KV cache claims what is left
- **Removed Ollama dependency** -- the LLM and embedding layers now talk directly to llama-server via the OpenAI-compatible API; Ollama is no longer required or supported

---

## [0.1.0] � 2026-04-11

Initial public release. Fully functional local-first personal data vault with hybrid RAG chat and Gmail integration.

### Added

#### Data & storage
- Single-file SQLite vault (`vault.db`) � portable, no server required
- SHA-256 deduplication � re-ingesting the same export is always a safe no-op
- Change detection for local files � unchanged files are skipped on re-scan
- Full-text search via FTS5 (BM25 ranking)
- Semantic vector search with `nomic-embed-text` embeddings

#### Retrieval pipeline
- 4-lane hybrid retrieval � BM25, dense cosine similarity, HyPE hypothetical question embeddings, and sentence-window chunks � merged via Reciprocal Rank Fusion (RRF)
- Contextual Retrieval � LLM-generated context prefix prepended to each record before indexing
- HyPE (Hypothetical Passage Embeddings) � hypothetical questions generated and embedded at index time for better question-to-answer matching
- Sentence Window Retrieval � fine-grained chunk retrieval with surrounding-context expansion
- CRAG-lite � confidence gate after reranking with automatic fallback strategies (HyDE, broaden, empty)
- BM25 reranker (built-in, zero extra dependencies) and optional cross-encoder reranker

#### Data sources
- **Local files** � PDF, DOCX, HTML, Markdown, EPUB, spreadsheets, presentations, plain text
- **Gmail Takeout** � import from `.mbox` export, no authentication required
- **Gmail live sync** � incremental OAuth2 sync; deduplication cross-compatible with Takeout imports
- **Gmail IMAP** � username + app password, no OAuth needed

#### Interfaces
- Terminal REPL (`egovault chat`) � Rich rendering, source attribution, slash commands
- Streamlit browser UI (`egovault web`) � agentic RAG loop, source panel, top-N slider
- MCP server (`egovault mcp`) � stdio transport for AnythingLLM and compatible clients

#### LLM backend
- llama-server (llama.cpp) managed startup � auto-starts, downloads GGUF model on first run, stops on exit
- OpenAI-compatible interface � swap in OpenAI, LM Studio, vLLM, Groq, or any compatible endpoint via `egovault.toml`
- VRAM-aware context sizing � automatically picks the largest context window that fits your GPU

#### Configuration & packaging
- `egovault.toml` for all settings
- `egovault` and `ego` CLI aliases
- Optional dependency groups: `[local]`, `[web]`, `[gmail]`, `[reranker]`
- MIT licence

---

[Unreleased]: https://github.com/milika/EgoVault/compare/v0.1.9...HEAD
[0.1.9]: https://github.com/milika/EgoVault/compare/v0.1.0...v0.1.9
[0.1.0]: https://github.com/milika/EgoVault/releases/tag/v0.1.0