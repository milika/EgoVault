"""Unit tests for VaultStore — insert, dedup, upsert, query."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from egovault.core.schema import NormalizedRecord
from egovault.core.store import VaultStore

_TS = datetime(2024, 3, 15, 10, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rec(**kwargs) -> NormalizedRecord:
    defaults = dict(
        platform="test",
        record_type="message",
        timestamp=_TS,
        sender_id="u1",
        sender_name="Alice",
        thread_id="thread-1",
        thread_name="Test Thread",
        body="Hello world",
    )
    defaults.update(kwargs)
    return NormalizedRecord(**defaults)


@pytest.fixture()
def store() -> VaultStore:
    s = VaultStore(":memory:")
    s.init_db()
    yield s
    s.close()


# ---------------------------------------------------------------------------
# insert / dedup
# ---------------------------------------------------------------------------


def test_insert_returns_true(store: VaultStore) -> None:
    r = _rec()
    assert store.upsert_record(r) is True


def test_dedup_returns_false(store: VaultStore) -> None:
    r = _rec()
    store.upsert_record(r)
    assert store.upsert_record(r) is False


def test_different_bodies_produce_different_records(store: VaultStore) -> None:
    r1 = _rec(body="Hello")
    r2 = _rec(body="World")
    assert store.upsert_record(r1) is True
    assert store.upsert_record(r2) is True
    assert store.get_records() == [r1, r2] or len(store.get_records()) == 2


def test_re_ingest_same_export_no_new_rows(store: VaultStore) -> None:
    r = _rec()
    store.upsert_record(r)
    store.upsert_record(r)
    assert len(store.get_records()) == 1


# ---------------------------------------------------------------------------
# get_unenriched_records
# ---------------------------------------------------------------------------


def test_fresh_record_is_unenriched(store: VaultStore) -> None:
    r = _rec()
    store.upsert_record(r)
    pending = store.get_unenriched_records(limit=10)
    assert len(pending) == 1
    assert pending[0].id == r.id


def test_limit_respected(store: VaultStore) -> None:
    for i in range(5):
        store.upsert_record(_rec(body=f"msg {i}"))
    assert len(store.get_unenriched_records(limit=3)) == 3


# ---------------------------------------------------------------------------
# mark_enriched
# ---------------------------------------------------------------------------


def test_mark_enriched_done(store: VaultStore) -> None:
    r = _rec()
    store.upsert_record(r)
    store.mark_enriched(r.id, 1)
    assert store.get_unenriched_records(limit=10) == []


def test_mark_enriched_failed(store: VaultStore) -> None:
    r = _rec()
    store.upsert_record(r)
    store.mark_enriched(r.id, 2)
    # enriched = 2 (failed), not 0, so not returned
    assert store.get_unenriched_records(limit=10) == []


# ---------------------------------------------------------------------------
# get_records — filters
# ---------------------------------------------------------------------------


def test_get_records_no_filter_returns_all(store: VaultStore) -> None:
    store.upsert_record(_rec(body="A"))
    store.upsert_record(_rec(body="B"))
    assert len(store.get_records()) == 2


def test_get_records_platform_filter(store: VaultStore) -> None:
    store.upsert_record(_rec(platform="facebook", body="fb"))
    store.upsert_record(_rec(platform="whatsapp", body="wa"))
    results = store.get_records({"platform": "facebook"})
    assert len(results) == 1
    assert results[0].platform == "facebook"


def test_get_records_record_type_filter(store: VaultStore) -> None:
    store.upsert_record(_rec(record_type="message", body="msg"))
    store.upsert_record(_rec(record_type="link", body="https://example.com"))
    results = store.get_records({"record_type": "link"})
    assert len(results) == 1
    assert results[0].record_type == "link"


def test_get_records_unknown_column_raises(store: VaultStore) -> None:
    with pytest.raises(ValueError, match="Unknown filter column"):
        store.get_records({"DROP TABLE normalized_records--": "x"})


# ---------------------------------------------------------------------------
# Round-trip fidelity
# ---------------------------------------------------------------------------


def test_roundtrip_preserves_fields(store: VaultStore) -> None:
    r = _rec(
        platform="facebook",
        record_type="link",
        body="https://example.com",
        sender_id="uid42",
        sender_name="Bob",
        thread_id="t99",
        thread_name="Bob's Chat",
        attachments=["photo.jpg", "video.mp4"],
        raw={"original_key": "original_value"},
    )
    store.upsert_record(r)
    results = store.get_records()
    assert len(results) == 1
    fetched = results[0]
    assert fetched.id == r.id
    assert fetched.platform == r.platform
    assert fetched.record_type == r.record_type
    assert fetched.timestamp == r.timestamp
    assert fetched.body == r.body
    assert fetched.attachments == r.attachments
    assert fetched.raw == r.raw
    assert fetched.sender_name == r.sender_name


# ---------------------------------------------------------------------------
# Contextual Retrieval (P1) — schema v4
# ---------------------------------------------------------------------------


def test_contextual_body_column_exists(store: VaultStore) -> None:
    """After init_db(), normalized_records must have a contextual_body column."""
    cur = store._con.execute("PRAGMA table_info(normalized_records)")
    cols = {row[1] for row in cur.fetchall()}
    assert "contextual_body" in cols


def test_upsert_contextual_body(store: VaultStore) -> None:
    r = _rec()
    store.upsert_record(r)
    store.upsert_contextual_body(r.id, "Context blurb. Original body follows.\n\n" + r.body)
    row = store._con.execute(
        "SELECT contextual_body FROM normalized_records WHERE id = ?", (r.id,)
    ).fetchone()
    assert row is not None
    assert "Context blurb" in row["contextual_body"]


def test_get_record_text_prefers_contextual_body(store: VaultStore) -> None:
    r = _rec(body="raw body text")
    store.upsert_record(r)
    # Before contextual_body: falls back to body
    assert "raw body text" in store.get_record_text_by_id(r.id)
    # After contextual_body: uses it instead
    store.upsert_contextual_body(r.id, "context prefix. raw body text")
    assert "context prefix" in store.get_record_text_by_id(r.id)


def test_get_uncontextualized_record_ids(store: VaultStore) -> None:
    r1 = _rec(body="first")
    r2 = _rec(body="second")
    store.upsert_record(r1)
    store.upsert_record(r2)
    # Both start uncontextualized
    pending = store.get_uncontextualized_record_ids()
    assert r1.id in pending
    assert r2.id in pending
    # Contextualize one
    store.upsert_contextual_body(r1.id, "ctx. first")
    pending2 = store.get_uncontextualized_record_ids()
    assert r1.id not in pending2
    assert r2.id in pending2


def test_upsert_contextual_body_empty_string_clears(store: VaultStore) -> None:
    r = _rec()
    store.upsert_record(r)
    store.upsert_contextual_body(r.id, "some prefix. " + r.body)
    store.upsert_contextual_body(r.id, "")  # clear
    row = store._con.execute(
        "SELECT contextual_body FROM normalized_records WHERE id = ?", (r.id,)
    ).fetchone()
    assert row["contextual_body"] is None


def test_schema_version_is_current(store: VaultStore) -> None:
    from egovault.core.store import _SCHEMA_VERSION

    version = store.get_setting("schema_version")
    assert version == str(_SCHEMA_VERSION)


# ---------------------------------------------------------------------------
# HyPE (P2) — record_question_embeddings table and methods
# ---------------------------------------------------------------------------


def test_question_embeddings_table_exists(store: VaultStore) -> None:
    """After init_db(), record_question_embeddings table must exist."""
    cur = store._con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='record_question_embeddings'"
    )
    assert cur.fetchone() is not None


def test_upsert_and_retrieve_question_embedding(store: VaultStore) -> None:
    r = _rec()
    store.upsert_record(r)
    vec = [0.1, 0.2, 0.3]
    store.upsert_question_embedding(r.id, "test-model", "What did I write?", vec)
    rows = store.get_all_question_embeddings("test-model")
    assert len(rows) == 1
    rec_id, question_text, blob = rows[0]
    assert rec_id == r.id
    assert question_text == "What did I write?"
    assert len(blob) == len(vec) * 4  # float32 bytes


def test_question_embedding_multiple_questions(store: VaultStore) -> None:
    r = _rec()
    store.upsert_record(r)
    questions = ["Question one?", "Question two?", "Question three?"]
    for q in questions:
        store.upsert_question_embedding(r.id, "test-model", q, [0.5, 0.5])
    rows = store.get_all_question_embeddings("test-model")
    assert len(rows) == 3
    stored_texts = {row[1] for row in rows}
    assert stored_texts == set(questions)


def test_get_records_without_hype_questions(store: VaultStore) -> None:
    r1 = _rec(body="first")
    r2 = _rec(body="second")
    store.upsert_record(r1)
    store.upsert_record(r2)
    # Both start without questions
    pending = store.get_records_without_hype_questions("test-model")
    assert r1.id in pending
    assert r2.id in pending
    # Add questions for r1 only
    store.upsert_question_embedding(r1.id, "test-model", "What is first?", [0.1])
    pending2 = store.get_records_without_hype_questions("test-model")
    assert r1.id not in pending2
    assert r2.id in pending2


def test_question_embeddings_model_isolation(store: VaultStore) -> None:
    r = _rec()
    store.upsert_record(r)
    store.upsert_question_embedding(r.id, "model-a", "Question for A?", [0.1])
    # model-b sees nothing
    assert store.get_all_question_embeddings("model-b") == []
    # model-b records-without-questions includes r
    pending_b = store.get_records_without_hype_questions("model-b")
    assert r.id in pending_b


# ---------------------------------------------------------------------------
# Sentence Window Retrieval (P4) — record_chunks table and methods
# ---------------------------------------------------------------------------


def test_record_chunks_table_exists(store: VaultStore) -> None:
    """After init_db(), record_chunks table must exist."""
    cur = store._con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='record_chunks'"
    )
    assert cur.fetchone() is not None


def test_upsert_and_retrieve_chunk_embedding(store: VaultStore) -> None:
    r = _rec()
    store.upsert_record(r)
    vec = [0.1, 0.2, 0.3]
    store.upsert_chunk_embedding(r.id, "test-model", 0, "Hello world.", vec)
    rows = store.get_all_chunk_embeddings("test-model")
    assert len(rows) == 1
    rec_id, cidx, chunk_text, blob = rows[0]
    assert rec_id == r.id
    assert cidx == 0
    assert chunk_text == "Hello world."
    assert len(blob) == len(vec) * 4


def test_chunks_multiple_chunks_per_record(store: VaultStore) -> None:
    r = _rec()
    store.upsert_record(r)
    for i, text in enumerate(["Sentence one.", "Sentence two.", "Sentence three."]):
        store.upsert_chunk_embedding(r.id, "test-model", i, text, [float(i)])
    rows = store.get_all_chunk_embeddings("test-model")
    assert len(rows) == 3
    idxs = [row[1] for row in rows]
    assert idxs == [0, 1, 2]  # ordered by chunk_index


def test_get_chunks_for_record(store: VaultStore) -> None:
    r = _rec()
    store.upsert_record(r)
    for i in range(3):
        store.upsert_chunk_embedding(r.id, "test-model", i, f"chunk {i}", [float(i)])
    chunks = store.get_chunks_for_record(r.id, "test-model")
    assert len(chunks) == 3
    assert chunks[0][0] == 0
    assert chunks[1][1] == "chunk 1"


def test_get_records_without_chunks(store: VaultStore) -> None:
    r1 = _rec(body="first")
    r2 = _rec(body="second")
    store.upsert_record(r1)
    store.upsert_record(r2)
    pending = store.get_records_without_chunks("test-model")
    assert r1.id in pending
    assert r2.id in pending
    # Add a chunk for r1
    store.upsert_chunk_embedding(r1.id, "test-model", 0, "text", [0.5])
    pending2 = store.get_records_without_chunks("test-model")
    assert r1.id not in pending2
    assert r2.id in pending2


def test_chunks_upsert_replaces_existing(store: VaultStore) -> None:
    r = _rec()
    store.upsert_record(r)
    store.upsert_chunk_embedding(r.id, "test-model", 0, "original", [0.1])
    store.upsert_chunk_embedding(r.id, "test-model", 0, "replaced", [0.9])
    chunks = store.get_chunks_for_record(r.id, "test-model")
    assert len(chunks) == 1
    assert chunks[0][1] == "replaced"


def test_schema_version_is_6(store: VaultStore) -> None:
    from egovault.core.store import _SCHEMA_VERSION
    assert _SCHEMA_VERSION == 6
    version = store.get_setting("schema_version")
    assert version == "6"
