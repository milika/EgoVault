"""Processing layer RAG shim.

During the Cycle-7.5 migration the canonical code lives in
``egovault.chat.rag``.  This module re-exports everything from there so that
``from egovault.processing.rag import retrieve`` works immediately.

When the migration completes, ``chat/rag.py`` will become the shim and this
will contain the actual code.
"""
# ruff: noqa: F401, F403
from egovault.chat.rag import *  # noqa: F401, F403
from egovault.chat.rag import (
    RetrievedChunk,
    _kwic_snippet,
    _sanitize_query,
    _path_score,
    _bm25_rerank,
    _crossencoder_rerank,
    rerank_chunks,
    embed_text,
    _rrf_fuse,
    _hyde_query,
    retrieve_semantic,
    retrieve_hype,
    plan_search_queries,
    retrieve,
    retrieve_sentence_window,
    assemble_context,
    build_prompt,
    extract_owner_profile,
    vault_summary_context,
    source_attribution,
)
