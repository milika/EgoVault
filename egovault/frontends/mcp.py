"""EgoVault MCP server — exposes vault RAG tools for AnythingLLM and other MCP clients.

Launch with:  egovault mcp

Connect from AnythingLLM:
  Settings → AI Providers → MCP → Custom Agent:
    Command: egovault mcp
    (or full path: .venv/Scripts/egovault mcp on Windows)

Included in: pip install egovault
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from egovault.config import Settings
    from egovault.core.store import VaultStore


# ---------------------------------------------------------------------------
# Module-level runtime state — set once in launch()
# ---------------------------------------------------------------------------

_store: "VaultStore | None" = None
_settings: "Settings | None" = None


def _get_store() -> "VaultStore":
    if _store is None:
        raise RuntimeError("MCP server not initialised — call launch() first")
    return _store


def _get_settings() -> "Settings":
    if _settings is None:
        raise RuntimeError("MCP server not initialised — call launch() first")
    return _settings


# ---------------------------------------------------------------------------
# MCP server definition
# ---------------------------------------------------------------------------

def _build_server():
    """Build and return the FastMCP server with all EgoVault tools registered."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise ImportError(
            "MCP SDK is not installed. Run: pip install egovault "
            "(or pip install mcp if it was removed manually)."
        ) from exc

    app = FastMCP(
        "EgoVault",
        instructions=(
            "Local-first personal data vault. "
            "Search and chat over your ingested emails, documents, and web history "
            "using hybrid RAG (FTS5 + semantic + HyPE + sentence-window retrieval)."
        ),
    )

    # -----------------------------------------------------------------------
    # Tool: search_vault
    # -----------------------------------------------------------------------

    @app.tool()
    def search_vault(query: str, top_n: int = 10, platform: str = "") -> str:
        """Hybrid RAG search over the EgoVault personal knowledge base.

        Runs FTS5 keyword search + semantic cosine similarity + HyPE +
        sentence-window retrieval fused via Reciprocal Rank Fusion, then
        reranks the merged candidate pool.

        Returns a JSON array of matching chunks. Each item has:
          - source: file path or thread name
          - platform: adapter id (e.g. 'gmail', 'local')
          - score: relevance score (higher is better)
          - snippet: first 400 chars of the chunk body
          - timestamp: record timestamp

        Args:
            query: Natural-language search query.
            top_n: Number of chunks to return (default 10).
            platform: Optional filter by platform id. Empty = all platforms.
        """
        from egovault.chat.rag import retrieve

        store = _get_store()
        settings = _get_settings()

        try:
            chunks = retrieve(
                store, query,
                top_n=top_n,
                reranker_cfg=settings.reranker,
                crag_cfg=settings.crag,
            )
        except Exception as exc:
            return json.dumps({"error": str(exc)})

        if platform:
            chunks = [c for c in chunks if getattr(c, "platform", "") == platform]

        results = [
            {
                "id": getattr(c, "id", ""),
                "source": getattr(c, "file_path", "") or getattr(c, "thread_name", ""),
                "platform": getattr(c, "platform", ""),
                "score": round(getattr(c, "score", 0.0), 4),
                "snippet": (getattr(c, "body", "") or "")[:400],
                "timestamp": str(getattr(c, "timestamp", "")),
            }
            for c in chunks
        ]
        return json.dumps(results, ensure_ascii=False, indent=2)

    # -----------------------------------------------------------------------
    # Tool: chat
    # -----------------------------------------------------------------------

    @app.tool()
    def chat(message: str, history: str = "[]", top_n: int = 10) -> str:
        """Conversational RAG chat over the personal vault.

        Retrieves relevant context using hybrid RAG, assembles a prompt,
        and calls the LLM configured in egovault.toml.

        Returns JSON with:
          - answer: the LLM response text
          - sources: list of source file/thread names attributed to the answer
          - chunks_used: number of context chunks retrieved

        Args:
            message: The user's question or message.
            history: JSON list of {role, content} dicts for conversation context.
            top_n: Number of RAG chunks to retrieve (default 10).
        """
        from egovault.chat.rag import (
            assemble_context,
            build_prompt,
            plan_search_queries,
            retrieve,
            source_attribution,
            vault_summary_context,
        )
        from egovault.chat.session import _call_llm

        store = _get_store()
        settings = _get_settings()
        llm_cfg = settings.llm
        llm_kwargs = dict(
            base_url=llm_cfg.base_url,
            model=llm_cfg.model,
            timeout=llm_cfg.timeout_seconds,
            provider=llm_cfg.provider,
            api_key=llm_cfg.api_key,
            num_gpu=llm_cfg.num_gpu,
            num_thread=llm_cfg.num_thread,
        )

        try:
            conv_history: list[dict] = json.loads(history) if history.strip() else []
        except json.JSONDecodeError:
            conv_history = []

        planned = None
        try:
            _planner_kwargs = {**llm_kwargs, "timeout": min(15, llm_cfg.timeout_seconds)}
            planned = plan_search_queries(message, conv_history, _call_llm, _planner_kwargs)
        except Exception:
            pass

        chunks: list = []
        try:
            chunks = retrieve(
                store, message,
                top_n=top_n,
                planned_query=planned,
                reranker_cfg=settings.reranker,
                crag_cfg=settings.crag,
            )
        except Exception:
            pass

        context = assemble_context(chunks)
        if not context:
            context = vault_summary_context(store)

        from datetime import date as _date
        messages = build_prompt(message, context, history=conv_history, today=_date.today().isoformat())

        try:
            answer, _ = _call_llm(messages=messages, **llm_kwargs)
        except Exception as exc:
            return json.dumps({"error": str(exc), "answer": "", "sources": [], "chunks_used": 0})

        sources = source_attribution(chunks) if chunks else []
        return json.dumps(
            {"answer": answer, "sources": sources[:10], "chunks_used": len(chunks)},
            ensure_ascii=False,
        )

    # -----------------------------------------------------------------------
    # Tool: vault_stats
    # -----------------------------------------------------------------------

    @app.tool()
    def vault_stats() -> str:
        """Return statistics about the EgoVault knowledge base.

        Includes total record count, enrichment status, embedding coverage,
        and a per-platform breakdown. Returns JSON.
        """
        store = _get_store()
        try:
            stats = store.get_vault_stats()
            return json.dumps(stats, ensure_ascii=False, indent=2)
        except AttributeError:
            # Fallback for VaultStore implementations without get_vault_stats()
            try:
                platforms = store.get_platforms() if hasattr(store, "get_platforms") else []
                return json.dumps({"platforms": platforms}, ensure_ascii=False)
            except Exception as exc2:
                return json.dumps({"error": str(exc2)})
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    # -----------------------------------------------------------------------
    # Tool: list_platforms
    # -----------------------------------------------------------------------

    @app.tool()
    def list_platforms() -> str:
        """List all data source platforms currently stored in the vault.

        Returns a JSON array of platform ids, e.g. ["gmail", "local", "facebook"].
        """
        store = _get_store()
        try:
            platforms = store.get_platforms()
            return json.dumps(platforms, ensure_ascii=False)
        except AttributeError:
            return json.dumps({"error": "get_platforms() not available in this VaultStore version"})
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    # -----------------------------------------------------------------------
    # Tool: get_gems
    # -----------------------------------------------------------------------

    @app.tool()
    def get_gems(gem_type: str = "", limit: int = 50) -> str:
        """Retrieve extracted gems (links, decisions, recommendations, actions).

        Gems are structured insights distilled by the LLM enrichment pipeline.

        Args:
            gem_type: Filter by type: 'Link', 'Decision', 'Recommendation', 'Action'.
                      Empty = return all types.
            limit: Maximum items to return (default 50, capped at 200).
        """
        store = _get_store()
        limit = min(limit, 200)
        try:
            gems = store.get_gems(gem_type=gem_type or None, limit=limit)
            return json.dumps(gems, ensure_ascii=False, indent=2)
        except AttributeError:
            return json.dumps({"error": "get_gems() not available in this VaultStore version"})
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    # -----------------------------------------------------------------------
    # Tool: record_feedback
    # -----------------------------------------------------------------------

    @app.tool()
    def record_feedback(record_id: str, rating: int, comment: str = "") -> str:
        """Submit retrieval feedback on a record (thumbs up / thumbs down).

        Feedback is stored locally and can be used to audit and improve
        future retrieval quality.

        Args:
            record_id: The ID of the record being rated (from search_vault results).
            rating: 1 for positive, -1 for negative.
            comment: Optional note, e.g. 'wrong source' or 'hallucination'.
        """
        if rating not in (1, -1):
            return json.dumps({"error": "rating must be 1 (positive) or -1 (negative)"})
        store = _get_store()
        try:
            store.upsert_feedback(record_id=record_id, rating=rating, comment=comment)
            return json.dumps({"status": "ok", "record_id": record_id, "rating": rating})
        except AttributeError:
            return json.dumps({"error": "upsert_feedback() not available in this VaultStore version"})
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    return app


# ---------------------------------------------------------------------------
# Public entry point — called by the CLI
# ---------------------------------------------------------------------------

def launch(store: "VaultStore", settings: "Settings") -> None:
    """Start the EgoVault MCP server on stdio transport.

    Blocks until the client disconnects or the process is interrupted.
    stdout is owned by the MCP protocol — do not write to it before calling
    this function.
    """
    global _store, _settings
    _store = store
    _settings = settings

    app = _build_server()
    app.run(transport="stdio")


# ---------------------------------------------------------------------------
# DEAD CODE TOMBSTONE — kept as reference only, remove after 2026-07-01
# ---------------------------------------------------------------------------
# The Gradio web UI (_chat_generator, build_app, launch with host/port/share)
# was replaced by this MCP server in April 2026.
# See docs/rag-ui-research.md for the original Gradio feature research and
# the migration rationale.


def _gradio_removed_notice(*_args, **_kwargs):  # type: ignore[no-untyped-def]
    """Placeholder so any stale import of the old Gradio symbols fails loudly."""
    raise NotImplementedError(
        "The Gradio web UI has been removed. "
        "Use 'egovault mcp' and connect AnythingLLM."
    )


# Stubs for anything that might still reference the old names
build_app = _gradio_removed_notice
_chat_generator_REMOVED = None  # mypy-safe sentinel for old Symbol: was `_chat_gen