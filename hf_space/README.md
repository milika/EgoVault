---
title: EgoVault Demo
emoji: 🔒
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: true
tags:
  - rag
  - local-llm
  - privacy
  - llama-cpp
  - personal-knowledge
  - sqlite
  - streamlit
license: mit
---

# EgoVault — Demo

> ⚠️ **This is a UI demo with synthetic data and the HF Inference API standing in for local llama-server.**
> EgoVault is designed to run **100% locally** — no cloud, no data leaving your machine.
> [Get the real thing on GitHub →](https://github.com/milika/egovault)

## What this demo shows

- Hybrid RAG retrieval: FTS5 keyword search over a synthetic personal data vault
- Chat interface powered by HF Inference API (Mistral-7B)
- Sources panel showing exactly which records were retrieved

## What the real EgoVault does (locally)

- Ingests Gmail, Telegram exports, and local files into a SQLite vault
- 4-lane retrieval: FTS5 + semantic (vector) + HyPE + sentence-window chunks → RRF fusion
- All LLM inference via llama-server (llama.cpp) — auto-selects GGUF model and context size from free VRAM
- No cloud dependencies whatsoever
