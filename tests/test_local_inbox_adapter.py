"""Unit tests for LocalInboxAdapter + end-to-end pipeline validation."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from egovault.adapters.local_inbox import LocalInboxAdapter
from egovault.config import Settings, LLMSettings
from egovault.core.enrichment import EnrichmentPipeline, _parse_response, _parse_gems
from egovault.core.schema import NormalizedRecord
from egovault.core.store import VaultStore
from egovault.output.markdown import generate_markdown

FIXTURES = Path(__file__).parent / "fixtures" / "local_inbox"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def store() -> VaultStore:
    s = VaultStore(":memory:")
    s.init_db()
    yield s
    s.close()


@pytest.fixture()
def adapter(store: VaultStore) -> LocalInboxAdapter:
    return LocalInboxAdapter(store=store)


# ---------------------------------------------------------------------------
# can_handle()
# ---------------------------------------------------------------------------


def test_can_handle_fixture_dir(adapter: LocalInboxAdapter) -> None:
    assert adapter.can_handle(FIXTURES) is True


def test_can_handle_rejects_file(adapter: LocalInboxAdapter, tmp_path: Path) -> None:
    f = tmp_path / "file.md"
    f.write_text("hello")
    # A file path, not a directory
    assert adapter.can_handle(f) is False


def test_can_handle_empty_dir(adapter: LocalInboxAdapter, tmp_path: Path) -> None:
    assert adapter.can_handle(tmp_path) is False


def test_can_handle_dir_with_unsupported_files(adapter: LocalInboxAdapter, tmp_path: Path) -> None:
    (tmp_path / "archive.xyz").write_bytes(b"garbage")
    assert adapter.can_handle(tmp_path) is False


# ---------------------------------------------------------------------------
# ingest() — field mapping
# ---------------------------------------------------------------------------


def _ingest_all(adapter: LocalInboxAdapter, path: Path) -> list[NormalizedRecord]:
    return list(adapter.ingest(path))


def test_ingest_yields_records_for_fixture_dir(adapter: LocalInboxAdapter) -> None:
    records = _ingest_all(adapter, FIXTURES)
    assert len(records) >= 3  # .md + .txt + .html


def test_ingest_md_record_type(adapter: LocalInboxAdapter) -> None:
    records = _ingest_all(adapter, FIXTURES)
    md_recs = [r for r in records if r.mime_type == "text/markdown"]
    assert md_recs, "Expected at least one .md record"
    r = md_recs[0]
    assert r.record_type == "note"
    assert r.platform == "local"
    assert r.sender_id == "user"
    assert "meeting" in r.body.lower() or r.body  # has content


def test_ingest_txt_record_type(adapter: LocalInboxAdapter) -> None:
    records = _ingest_all(adapter, FIXTURES)
    txt_recs = [r for r in records if r.mime_type == "text/plain"]
    assert txt_recs
    r = txt_recs[0]
    assert r.record_type == "note"
    assert "reading" in r.body.lower() or r.body


def test_ingest_html_extracts_text(adapter: LocalInboxAdapter) -> None:
    records = _ingest_all(adapter, FIXTURES)
    html_recs = [r for r in records if r.mime_type == "text/html"]
    assert html_recs
    r = html_recs[0]
    # HTML tags should be stripped
    assert "<html>" not in r.body
    assert "<p>" not in r.body
    assert "local-first" in r.body.lower() or "local" in r.body.lower()


def test_ingest_file_path_set(adapter: LocalInboxAdapter) -> None:
    records = _ingest_all(adapter, FIXTURES)
    for r in records:
        assert r.file_path is not None
        assert Path(r.file_path).exists()


def test_ingest_attachment_is_file_path(adapter: LocalInboxAdapter) -> None:
    records = _ingest_all(adapter, FIXTURES)
    for r in records:
        assert len(r.attachments) == 1
        assert r.attachments[0] == r.file_path


def test_ingest_id_is_deterministic(adapter: LocalInboxAdapter) -> None:
    r1 = _ingest_all(adapter, FIXTURES)
    r2 = _ingest_all(LocalInboxAdapter(), FIXTURES)
    ids1 = {r.id for r in r1}
    ids2 = {r.id for r in r2}
    assert ids1 == ids2


# ---------------------------------------------------------------------------
# Change detection (ingested_files)
# ---------------------------------------------------------------------------


def test_change_detection_skips_known_files(adapter: LocalInboxAdapter, store: VaultStore) -> None:
    # First ingest — all files are new
    first_run = _ingest_all(adapter, FIXTURES)
    assert len(first_run) >= 3

    # Simulate the CLI tracking step: mark all files as ingested
    for rec in first_run:
        store.upsert_record(rec)
        fid = rec.raw.get("file_id")
        if fid and rec.file_path:
            store.upsert_ingested_file(
                file_id=str(fid),
                path=rec.file_path,
                mtime=float(rec.raw["mtime"]),
                size_bytes=int(rec.raw["size_bytes"]),
                platform="local",
            )

    # Second ingest — same files, should be skipped entirely
    second_run = _ingest_all(adapter, FIXTURES)
    assert second_run == []


def test_dedup_via_store_upsert(adapter: LocalInboxAdapter, store: VaultStore) -> None:
    records = _ingest_all(adapter, FIXTURES)
    for r in records:
        store.upsert_record(r)
    # Re-run without file tracking
    adapter2 = LocalInboxAdapter()  # no store → no skip logic
    records2 = _ingest_all(adapter2, FIXTURES)
    for r in records2:
        was_new = store.upsert_record(r)
        assert was_new is False, "Re-ingest should be a no-op via INSERT OR IGNORE"


# ---------------------------------------------------------------------------
# Enrichment — unit tests (mocked LLM)
# ---------------------------------------------------------------------------

_MOCK_RESPONSE = """\
SUMMARY: This meeting covered Q1 planning and produced several action items.
GEMS:
- [Decision] We agreed to move the weekly sync to Fridays at 10am.
- [Action] Bob will send the project report by end of March.
- [Recommendation] Alice recommended reading Getting Things Done.
- [Link] https://example.com/project-dashboard — shared by Alice
"""


def test_parse_response_extracts_summary() -> None:
    summary, gems_raw = _parse_response(_MOCK_RESPONSE)
    assert "Q1 planning" in summary


def test_parse_response_extracts_gems_raw() -> None:
    _, gems_raw = _parse_response(_MOCK_RESPONSE)
    assert "Decision" in gems_raw
    assert "Action" in gems_raw


def test_parse_gems_returns_structured_list() -> None:
    _, gems_raw = _parse_response(_MOCK_RESPONSE)
    gems = _parse_gems(gems_raw)
    types = {g["gem_type"] for g in gems}
    assert "decision" in types
    assert "action" in types
    assert "link" in types


def test_parse_gems_link_has_url() -> None:
    _, gems_raw = _parse_response(_MOCK_RESPONSE)
    gems = _parse_gems(gems_raw)
    link_gems = [g for g in gems if g["gem_type"] == "link"]
    assert link_gems
    assert link_gems[0]["url"] == "https://example.com/project-dashboard"


def _make_settings() -> Settings:
    return Settings(
        vault_db=":memory:",
        output_dir="/tmp/egovault_test_output",
        llm=LLMSettings(
            model="test-model",
            base_url="http://localhost:11434",
            timeout_seconds=10,
        ),
    )


def test_enrich_record_success(store: VaultStore) -> None:
    from datetime import datetime, timezone

    record = NormalizedRecord(
        platform="local",
        record_type="note",
        timestamp=datetime(2024, 3, 15, tzinfo=timezone.utc),
        sender_id="user",
        sender_name="user",
        thread_id="/inbox",
        thread_name="inbox",
        body="We agreed to meet Fridays. Bob will send report. https://example.com",
    )
    store.upsert_record(record)

    settings = _make_settings()
    pipeline = EnrichmentPipeline(store, settings)

    with patch(
        "egovault.core.enrichment._call_llm_simple", return_value=_MOCK_RESPONSE
    ):
        ok = pipeline.enrich_record(record)

    assert ok is True
    # enriched flag should be 1
    results = store.get_records({"enriched": 1})
    assert any(r.id == record.id for r in results)


def test_enrich_record_failure_marks_failed(store: VaultStore) -> None:
    from datetime import datetime, timezone

    record = NormalizedRecord(
        platform="local",
        record_type="note",
        timestamp=datetime(2024, 3, 16, tzinfo=timezone.utc),
        sender_id="user",
        sender_name="user",
        thread_id="/inbox",
        thread_name="inbox",
        body="Test failure record",
    )
    store.upsert_record(record)
    settings = _make_settings()
    pipeline = EnrichmentPipeline(store, settings)

    with patch(
        "egovault.core.enrichment._call_llm_simple",
        side_effect=OSError("connection refused"),
    ), patch("egovault.core.enrichment.time.sleep"):
        ok = pipeline.enrich_record(record)

    assert ok is False
    failed = store.get_records({"enriched": 2})
    assert any(r.id == record.id for r in failed)


# ---------------------------------------------------------------------------
# Markdown output
# ---------------------------------------------------------------------------


def test_generate_markdown_creates_file(store: VaultStore, tmp_path: Path) -> None:
    from datetime import datetime, timezone

    record = NormalizedRecord(
        platform="local",
        record_type="note",
        timestamp=datetime(2024, 3, 15, tzinfo=timezone.utc),
        sender_id="user",
        sender_name="user",
        thread_id="/inbox",
        thread_name="inbox",
        body="Test body content",
    )
    store.upsert_record(record)
    store.insert_enrichment_result(
        record_id=record.id,
        model="test-model",
        summary="A test summary.",
        gems_raw="- [Decision] Test decision",
    )
    store.insert_gem(record.id, "decision", "Test decision")
    store.mark_enriched(record.id, 1)

    out_path = generate_markdown(record, store, tmp_path)

    assert out_path is not None
    assert out_path.exists()
    text = out_path.read_text(encoding="utf-8")
    assert "platform: local" in text
    assert "## Summary" in text
    assert "A test summary." in text
    assert "## Gems" in text
    assert "Test decision" in text
    assert "## Raw Context" in text
    assert "Test body content" in text


def test_generate_markdown_no_enrichment_returns_none(store: VaultStore, tmp_path: Path) -> None:
    from datetime import datetime, timezone

    record = NormalizedRecord(
        platform="local",
        record_type="note",
        timestamp=datetime(2024, 3, 15, tzinfo=timezone.utc),
        sender_id="user",
        sender_name="user",
        thread_id="/inbox",
        thread_name="inbox",
        body="Not enriched",
    )
    store.upsert_record(record)
    result = generate_markdown(record, store, tmp_path)
    assert result is None


# ---------------------------------------------------------------------------
# End-to-end: ingest → enrich → export
# ---------------------------------------------------------------------------


def test_end_to_end_pipeline(store: VaultStore, tmp_path: Path) -> None:
    # 1. Ingest
    adapter = LocalInboxAdapter(store=store)
    records_ingested = []
    for rec in adapter.ingest(FIXTURES):
        if store.upsert_record(rec):
            records_ingested.append(rec)

    assert len(records_ingested) >= 3

    # 2. Enrich (mocked)
    settings = _make_settings()
    settings.output_dir = str(tmp_path / "output")
    pipeline = EnrichmentPipeline(store, settings)

    with patch(
        "egovault.core.enrichment._call_llm_simple", return_value=_MOCK_RESPONSE
    ):
        ok, fail = pipeline.enrich_all()

    assert ok >= 3
    assert fail == 0

    # 3. Export
    from egovault.output.markdown import MarkdownGenerator
    gen = MarkdownGenerator(store, settings)
    paths = gen.generate_all()

    assert len(paths) >= 3
    for p in paths:
        text = p.read_text(encoding="utf-8")
        assert "## Summary" in text
        assert "platform: local" in text


def test_dedup_re_ingest_no_new_records(store: VaultStore) -> None:
    adapter = LocalInboxAdapter(store=store)

    # First ingest
    first = []
    for rec in adapter.ingest(FIXTURES):
        was_new = store.upsert_record(rec)
        if was_new:
            fid = rec.raw.get("file_id")
            if fid and rec.file_path:
                store.upsert_ingested_file(
                    str(fid), rec.file_path,
                    float(rec.raw["mtime"]), int(rec.raw["size_bytes"]), "local",
                    content_hash=rec.raw.get("content_hash"),
                )
            first.append(rec)

    assert len(first) >= 3

    # Second ingest — change detection should skip all files
    second = list(adapter.ingest(FIXTURES))
    assert second == [], "Re-ingest on unchanged files should yield nothing"


def test_content_hash_dedup_skips_renamed_file(store: VaultStore, tmp_path: Path) -> None:
    """A file copied to a new path with a different mtime is still skipped by content hash."""
    import shutil

    src = FIXTURES / "meeting_notes.md"
    copy_a = tmp_path / "a" / "meeting_notes.md"
    copy_b = tmp_path / "b" / "renamed_copy.md"
    copy_a.parent.mkdir()
    copy_b.parent.mkdir()
    shutil.copy2(src, copy_a)
    shutil.copy2(src, copy_b)

    adapter = LocalInboxAdapter(store=store)

    # Ingest copy_a and record it in ingested_files
    records_a = list(adapter.ingest(copy_a.parent))
    assert len(records_a) == 1
    rec = records_a[0]
    store.upsert_record(rec)
    store.upsert_ingested_file(
        file_id=str(rec.raw["file_id"]),
        path=rec.file_path,
        mtime=float(rec.raw["mtime"]),
        size_bytes=int(rec.raw["size_bytes"]),
        platform="local",
        content_hash=rec.raw.get("content_hash"),
    )

    # Ingest copy_b — same content but different path → should be skipped
    records_b = list(adapter.ingest(copy_b.parent))
    assert records_b == [], "Renamed copy with identical content should be skipped via content hash"
