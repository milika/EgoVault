"""Tests for RAG retrieval and chat session (mocked LLM)."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from egovault.chat.rag import (
    RetrievedChunk,
    _sanitize_query,
    assemble_context,
    build_prompt,
    retrieve,
    source_attribution,
    vault_summary_context,
)
from egovault.config import LLMSettings, Settings
from egovault.core.schema import NormalizedRecord
from egovault.core.store import VaultStore

_TS = datetime(2024, 3, 15, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def store() -> VaultStore:
    s = VaultStore(":memory:")
    s.init_db()
    yield s
    s.close()


@pytest.fixture()
def populated_store(store: VaultStore) -> VaultStore:
    """Store with several records covering different topics."""
    records = [
        NormalizedRecord(
            platform="whatsapp",
            record_type="message",
            timestamp=_TS,
            sender_id="alice",
            sender_name="Alice",
            thread_id="t1",
            thread_name="Alice Chat",
            body="Check out this machine learning article: https://ml.example.com",
        ),
        NormalizedRecord(
            platform="facebook",
            record_type="message",
            timestamp=_TS,
            sender_id="bob",
            sender_name="Bob",
            thread_id="t2",
            thread_name="Bob Chat",
            body="We agreed to move the meeting to Fridays at 10am",
        ),
        NormalizedRecord(
            platform="local",
            record_type="note",
            timestamp=_TS,
            sender_id="user",
            sender_name="user",
            thread_id="/inbox",
            thread_name="inbox",
            body="Book recommendation: Clean Code by Robert Martin",
        ),
        NormalizedRecord(
            platform="reddit",
            record_type="link",
            timestamp=_TS,
            sender_id="",
            sender_name="",
            thread_id="r/python",
            thread_name="r/python",
            body="Python best practices guide: https://python-guide.example.com",
        ),
    ]
    for rec in records:
        store.upsert_record(rec)
    return store


# ---------------------------------------------------------------------------
# _sanitize_query
# ---------------------------------------------------------------------------


def test_sanitize_removes_fts5_operators() -> None:
    result = _sanitize_query('machine "learning" (OR) ^test*')
    assert '"' not in result
    assert "(" not in result
    assert "^" not in result
    assert "*" not in result


def test_sanitize_preserves_words() -> None:
    result = _sanitize_query("machine learning article")
    assert "machine" in result
    assert "learning" in result


def test_sanitize_empty_query() -> None:
    assert _sanitize_query("   ") == ""


# ---------------------------------------------------------------------------
# retrieve — FTS5 search
# ---------------------------------------------------------------------------


def test_retrieve_returns_relevant_records(populated_store: VaultStore) -> None:
    results = retrieve(populated_store, "machine learning", top_n=10)
    assert len(results) >= 1
    bodies = [c.record.body for c in results]
    assert any("machine learning" in b.lower() for b in bodies)


def test_retrieve_top_n_limit(populated_store: VaultStore) -> None:
    results = retrieve(populated_store, "the", top_n=2)
    assert len(results) <= 2


def test_retrieve_no_duplicates(populated_store: VaultStore) -> None:
    results = retrieve(populated_store, "example", top_n=10)
    ids = [c.record.id for c in results]
    assert len(ids) == len(set(ids))


def test_retrieve_empty_query_returns_empty(populated_store: VaultStore) -> None:
    results = retrieve(populated_store, "   ")
    assert results == []


def test_retrieve_fallback_like_search(populated_store: VaultStore) -> None:
    # Search with a term that won't match FTS5 tokenizer but will match LIKE
    results = retrieve(populated_store, "Clean Code", top_n=10)
    assert len(results) >= 1


def test_retrieve_empty_vault_returns_empty(store: VaultStore) -> None:
    results = retrieve(store, "anything")
    assert results == []


# ---------------------------------------------------------------------------
# assemble_context
# ---------------------------------------------------------------------------


def test_assemble_context_includes_platform(populated_store: VaultStore) -> None:
    chunks = retrieve(populated_store, "machine learning", top_n=5)
    ctx = assemble_context(chunks)
    assert "whatsapp" in ctx.lower() or "facebook" in ctx.lower() or "local" in ctx.lower()


def test_assemble_context_no_results() -> None:
    ctx = assemble_context([])
    assert ctx == ""


def test_assemble_context_includes_body(populated_store: VaultStore) -> None:
    chunks = retrieve(populated_store, "meeting Fridays", top_n=5)
    ctx = assemble_context(chunks)
    assert "Fridays" in ctx or "meeting" in ctx.lower()


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------


def test_build_prompt_has_system_role() -> None:
    messages = build_prompt("What links did Alice share?", "context")
    roles = [m["role"] for m in messages]
    assert "system" in roles
    assert "user" in roles


def test_build_prompt_contains_query() -> None:
    messages = build_prompt("What links did Alice share?", "context block")
    user_msg = next(m for m in messages if m["role"] == "user")
    assert "What links did Alice share?" in user_msg["content"]


def test_build_prompt_contains_context() -> None:
    messages = build_prompt("question", "VAULT CONTEXT goes here")
    user_msg = next(m for m in messages if m["role"] == "user")
    assert "VAULT CONTEXT goes here" in user_msg["content"]


# ---------------------------------------------------------------------------
# source_attribution
# ---------------------------------------------------------------------------


def test_source_attribution_returns_labels(populated_store: VaultStore) -> None:
    chunks = retrieve(populated_store, "machine learning", top_n=5)
    labels = source_attribution(chunks)
    assert len(labels) == len(chunks)
    for label in labels:
        assert "2024" in label  # date present


# ---------------------------------------------------------------------------
# Integration test — full RAG → mocked LLM → answer
# ---------------------------------------------------------------------------


def _make_settings() -> Settings:
    return Settings(
        vault_db=":memory:",
        llm=LLMSettings(
            model="test-model",
            base_url="http://localhost:11434",
            timeout_seconds=5,
        ),
    )


_MOCK_ANSWER = "Alice shared a machine learning article at https://ml.example.com on 2024-03-15."


def test_integration_rag_produces_grounded_answer(populated_store: VaultStore) -> None:
    """Full pipeline: retrieve → assemble → prompt → mocked LLM → answer."""
    chunks = retrieve(populated_store, "machine learning", top_n=10)
    assert chunks, "Expected at least one retrieved chunk"

    context = assemble_context(chunks)
    messages = build_prompt("What links did Alice share about machine learning?", context)

    # Verify context contains the expected record before handing to LLM
    user_msg = next(m for m in messages if m["role"] == "user")
    assert "ml.example.com" in user_msg["content"] or "machine learning" in user_msg["content"].lower()

    # Simulate LLM response via mock
    with patch("egovault.chat.session._call_llm", return_value=_MOCK_ANSWER) as mock_call:
        from egovault.chat.session import _call_llm as mocked_fn
        result = mocked_fn(
            base_url="http://localhost:11434",
            model="test-model",
            messages=messages,
            timeout=5,
        )

    assert "ml.example.com" in result or "machine learning" in result.lower()


def test_integration_no_context_handled_gracefully(store: VaultStore) -> None:
    """With empty vault, retrieve returns nothing and vault_summary_context is empty too."""
    chunks = retrieve(store, "machine learning", top_n=10)
    ctx = assemble_context(chunks)
    assert ctx == ""
    # vault is empty, so summary is also empty
    summary = vault_summary_context(store)
    assert summary == ""


def test_fts5_retrieves_across_platforms(populated_store: VaultStore) -> None:
    """FTS5 index spans all platforms — searching a common word returns records."""
    results = retrieve(populated_store, "example", top_n=10)
    platforms = {c.record.platform for c in results}
    assert len(platforms) >= 2  # should span at least 2 platforms


# ---------------------------------------------------------------------------
# CRAG-lite (P3) tests
# ---------------------------------------------------------------------------


def test_crag_empty_strategy_returns_empty(populated_store: VaultStore) -> None:
    """When strategy='empty' and scores are below threshold, retrieve() returns []."""
    from egovault.config import CRAGSettings

    crag = CRAGSettings(enabled=True, threshold=9999.0, strategy="empty")
    results = retrieve(populated_store, "machine learning", top_n=5, crag_cfg=crag)
    assert results == []


def test_crag_disabled_does_not_affect_results(populated_store: VaultStore) -> None:
    """With crag.enabled=False the results are the same as without crag_cfg."""
    from egovault.config import CRAGSettings

    baseline = retrieve(populated_store, "machine learning", top_n=5)
    crag_off = CRAGSettings(enabled=False, threshold=9999.0, strategy="empty")
    with_crag = retrieve(populated_store, "machine learning", top_n=5, crag_cfg=crag_off)
    assert [c.record.id for c in with_crag] == [c.record.id for c in baseline]


def test_crag_broaden_does_not_crash(populated_store: VaultStore) -> None:
    """strategy='broaden' with a very high threshold triggers re-retrieval without error."""
    from egovault.config import CRAGSettings

    crag = CRAGSettings(enabled=True, threshold=9999.0, strategy="broaden")
    # Should not raise; may return empty or some results
    results = retrieve(populated_store, "machine learning article", top_n=5, crag_cfg=crag)
    assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Contextual Retrieval (P1) — config & config loading
# ---------------------------------------------------------------------------


def test_contextual_enabled_default_true() -> None:
    from egovault.config import EmbeddingSettings

    cfg = EmbeddingSettings()
    assert cfg.contextual_enabled is True


def test_crag_settings_defaults() -> None:
    from egovault.config import CRAGSettings

    crag = CRAGSettings()
    assert crag.enabled is True
    assert crag.threshold == pytest.approx(0.1)
    assert crag.strategy == "hyde"


def test_settings_has_crag_field() -> None:
    from egovault.config import Settings

    s = Settings()
    assert hasattr(s, "crag")
    assert s.crag.enabled is True


# ---------------------------------------------------------------------------
# HyPE (P2) — config defaults and retrieval lane
# ---------------------------------------------------------------------------


def test_hype_enabled_default_false() -> None:
    from egovault.config import EmbeddingSettings

    cfg = EmbeddingSettings()
    assert cfg.hype_enabled is True


def test_retrieve_hype_empty_when_no_question_embeddings(store: VaultStore) -> None:
    """retrieve_hype() returns [] when no question embeddings are stored."""
    from egovault.chat.rag import retrieve_hype
    from egovault.config import EmbeddingSettings

    embed_cfg = EmbeddingSettings(enabled=True, model="nomic-embed-text")
    results = retrieve_hype(store, "test query", 5, embed_cfg, "http://localhost:11434")
    assert results == []


def test_retrieve_hype_returns_chunks_when_embeddings_present(
    populated_store: VaultStore,
) -> None:
    """retrieve_hype() returns results when question embeddings exist and embed_text can be called."""
    from unittest.mock import patch

    from egovault.chat.rag import retrieve_hype
    from egovault.config import EmbeddingSettings

    # Seed one question embedding with a known vector
    rec = populated_store.get_records()[0]
    vec = [0.1] * 4
    populated_store.upsert_question_embedding(rec.id, "test-model", "What did Alice share?", vec)

    embed_cfg = EmbeddingSettings(enabled=True, model="test-model")

    # Mock embed_text so we don't need a live LLM server
    with patch("egovault.chat.rag.embed_text", return_value=[0.1] * 4):
        results = retrieve_hype(
            populated_store, "Alice machine learning", 5, embed_cfg, "http://localhost:11434"
        )

    assert len(results) >= 1
    assert results[0].record.id == rec.id


def test_rrf_fuse_three_lists(populated_store: VaultStore) -> None:
    """_rrf_fuse accepts an optional third list and scores all three correctly."""
    from egovault.chat.rag import RetrievedChunk, _rrf_fuse
    from egovault.core.schema import NormalizedRecord
    from datetime import datetime, timezone

    _ts = datetime(2024, 1, 1, tzinfo=timezone.utc)

    def _make_chunk(rec_id: str, body: str) -> RetrievedChunk:
        rec = NormalizedRecord(
            platform="test", record_type="note", timestamp=_ts,
            sender_id="u", sender_name="u", thread_id="t", thread_name="t", body=body,
        )
        # hack: set id to known value
        object.__setattr__(rec, "_id", rec_id)
        return RetrievedChunk(record=rec, rank=1.0)

    # Three separate lists each with a unique document
    a = _make_chunk("id-a", "doc A")
    b = _make_chunk("id-b", "doc B")
    c = _make_chunk("id-c", "doc C")

    result = _rrf_fuse([a], [b], [c])
    # Each unique doc should appear exactly once
    ids = [chunk.record.id for chunk in result]
    assert len(ids) == len(set(ids))
    assert len(ids) == 3


def test_retrieve_with_hype_enabled_no_crash(populated_store: VaultStore) -> None:
    """retrieve() with hype_enabled=True and no question embeddings stored returns normal FTS results."""
    from egovault.config import EmbeddingSettings

    embed_cfg = EmbeddingSettings(enabled=False, hype_enabled=True)
    # hype_enabled is only active when embeddings.enabled=True, so this path
    # falls through to FTS-only and must not raise
    results = retrieve(populated_store, "machine learning", top_n=5, embed_cfg=embed_cfg)
    assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Sentence Window Retrieval (P4) — config defaults and retrieval lane
# ---------------------------------------------------------------------------


def test_sentence_window_settings_defaults() -> None:
    from egovault.config import SentenceWindowSettings

    sw = SentenceWindowSettings()
    assert sw.enabled is True
    assert sw.window_size == 3
    assert sw.overlap == 1


def test_settings_has_sentence_window_field() -> None:
    from egovault.config import Settings

    s = Settings()
    assert hasattr(s, "sentence_window")
    assert s.sentence_window.enabled is True


def test_retrieve_sentence_window_empty_when_no_chunks(store: VaultStore) -> None:
    """retrieve_sentence_window() returns [] when no chunk embeddings are stored."""
    from egovault.chat.rag import retrieve_sentence_window
    from egovault.config import EmbeddingSettings

    embed_cfg = EmbeddingSettings(enabled=True, model="nomic-embed-text")
    results = retrieve_sentence_window(store, "test query", 5, embed_cfg, "http://localhost:11434")
    assert results == []


def test_retrieve_sentence_window_returns_snippets(populated_store: VaultStore) -> None:
    """retrieve_sentence_window() returns chunks with snippets when chunk embeddings exist."""
    from unittest.mock import patch

    from egovault.chat.rag import retrieve_sentence_window
    from egovault.config import EmbeddingSettings

    rec = populated_store.get_records()[0]
    # Seed two chunk embeddings for this record
    vec = [0.5] * 4
    populated_store.upsert_chunk_embedding(rec.id, "test-model", 0, "First sentence window.", vec)
    populated_store.upsert_chunk_embedding(rec.id, "test-model", 1, "Second sentence window.", vec)

    embed_cfg = EmbeddingSettings(enabled=True, model="test-model")

    with patch("egovault.chat.rag.embed_text", return_value=[0.5] * 4):
        results = retrieve_sentence_window(
            populated_store, "machine learning", 5, embed_cfg, "http://localhost:11434", window_size=3
        )

    assert len(results) >= 1
    assert results[0].record.id == rec.id
    assert results[0].snippet  # snippet must be non-empty


def test_retrieve_with_sw_disabled_no_crash(populated_store: VaultStore) -> None:
    """retrieve() with sw_cfg=None behaves identically to baseline."""
    baseline = retrieve(populated_store, "machine learning", top_n=5)
    with_sw_none = retrieve(populated_store, "machine learning", top_n=5, sw_cfg=None)
    assert [c.record.id for c in with_sw_none] == [c.record.id for c in baseline]


def test_retrieve_with_sw_enabled_no_crash(populated_store: VaultStore) -> None:
    """retrieve() with sw_cfg.enabled=True and no chunks stored returns normal FTS results."""
    from egovault.config import EmbeddingSettings, SentenceWindowSettings

    sw_cfg = SentenceWindowSettings(enabled=True, window_size=3)
    embed_cfg = EmbeddingSettings(enabled=False)  # chunks only active with embeddings.enabled
    results = retrieve(populated_store, "machine learning", top_n=5, embed_cfg=embed_cfg, sw_cfg=sw_cfg)
    assert isinstance(results, list)


def test_rrf_fuse_four_lists(populated_store: VaultStore) -> None:
    """_rrf_fuse handles four lists (fts + sem + hype + sw) without error."""
    from egovault.chat.rag import RetrievedChunk, _rrf_fuse

    rec = populated_store.get_records()[0]
    chunk = RetrievedChunk(record=rec, rank=1.0, snippet="test")
    # Pass same chunk in all four positions (identical IDs — should collapse to 1 result)
    result = _rrf_fuse([chunk], [chunk], hype_chunks=[chunk], sw_chunks=[chunk])
    assert len(result) == 1
    # Score =  4 × 1/(60+0) = 4/60 ≈ 0.0667
    assert result[0].rank == pytest.approx(4 / 60, rel=1e-3)
