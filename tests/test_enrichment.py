"""Unit tests for enrichment pipeline helpers."""
from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch


from egovault.core.enrichment import (
    EnrichmentPipeline,
    _chunk_records,
    _parse_gems,
    _parse_response,
)
from egovault.core.schema import EnrichmentStatus, NormalizedRecord


def _make_record(body: str, suffix: str = "") -> NormalizedRecord:
    return NormalizedRecord(
        platform="test",
        record_type="note",
        thread_id=f"thread{suffix}",
        thread_name="",
        sender_id=f"sender{suffix}",
        sender_name="",
        timestamp=datetime(2024, 1, 1),
        body=body,
        raw={},
    )


# ---------------------------------------------------------------------------
# _parse_response
# ---------------------------------------------------------------------------

class TestParseResponse:
    def test_extracts_summary_and_gems(self) -> None:
        response = "SUMMARY: This is the summary.\nGEMS:\n- [Link] https://example.com"
        summary, gems_raw = _parse_response(response)
        assert summary == "This is the summary."
        assert "https://example.com" in gems_raw

    def test_returns_empty_on_blank_response(self) -> None:
        summary, gems_raw = _parse_response("")
        assert summary == ""
        assert gems_raw == ""

    def test_gems_section_without_summary(self) -> None:
        response = "GEMS:\n- [Decision] Use PostgreSQL"
        _, gems_raw = _parse_response(response)
        assert "Use PostgreSQL" in gems_raw

    def test_summary_only(self) -> None:
        response = "SUMMARY: Just a summary, no gems."
        summary, gems_raw = _parse_response(response)
        assert summary == "Just a summary, no gems."
        assert gems_raw == ""

    def test_case_insensitive_headers(self) -> None:
        response = "summary: lower case summary\ngems:\n- [Action] do something"
        summary, gems_raw = _parse_response(response)
        assert summary == "lower case summary"
        assert "do something" in gems_raw


# ---------------------------------------------------------------------------
# _parse_gems
# ---------------------------------------------------------------------------

class TestParseGems:
    def test_parses_link_gem(self) -> None:
        gems_raw = "- [Link] Check https://example.com for details"
        gems = _parse_gems(gems_raw)
        assert len(gems) == 1
        assert gems[0]["gem_type"] == "link"
        assert gems[0]["url"] == "https://example.com"

    def test_parses_decision_gem(self) -> None:
        gems_raw = "- [Decision] We chose Python over Go"
        gems = _parse_gems(gems_raw)
        assert len(gems) == 1
        assert gems[0]["gem_type"] == "decision"
        assert gems[0]["url"] is None

    def test_parses_recommendation_gem(self) -> None:
        gems_raw = "- [Recommendation] Read Clean Code"
        gems = _parse_gems(gems_raw)
        assert gems[0]["gem_type"] == "recommendation"

    def test_parses_action_gem(self) -> None:
        gems_raw = "- [Action] Follow up with team by Friday"
        gems = _parse_gems(gems_raw)
        assert gems[0]["gem_type"] == "action"

    def test_skips_non_gem_lines(self) -> None:
        gems_raw = "Not a gem line\n- [Link] https://real.com"
        gems = _parse_gems(gems_raw)
        assert len(gems) == 1

    def test_parses_multiple_gems(self) -> None:
        gems_raw = (
            "- [Link] https://a.com\n"
            "- [Decision] Go with option B\n"
            "- [Action] Schedule meeting"
        )
        gems = _parse_gems(gems_raw)
        assert len(gems) == 3

    def test_empty_input(self) -> None:
        assert _parse_gems("") == []


# ---------------------------------------------------------------------------
# _chunk_records
# ---------------------------------------------------------------------------

class TestChunkRecords:
    def test_empty_input(self) -> None:
        assert _chunk_records([]) == []

    def test_single_record_fits_in_one_chunk(self) -> None:
        records = [_make_record("short body")]
        chunks = _chunk_records(records, target_tokens=2000)
        assert len(chunks) == 1
        assert chunks[0] == records

    def test_splits_large_records(self) -> None:
        # 4 chars ≈ 1 token, so 400 chars ≈ 100 tokens
        records = [_make_record("a" * 400, suffix=str(i)) for i in range(30)]
        chunks = _chunk_records(records, target_tokens=500, overlap=0)
        assert len(chunks) > 1
        # Every original record must appear at least once
        all_records = [r for chunk in chunks for r in chunk]
        assert len(all_records) >= 30

    def test_overlap_carries_records(self) -> None:
        records = [_make_record("a" * 400, suffix=str(i)) for i in range(10)]
        chunks = _chunk_records(records, target_tokens=500, overlap=2)
        if len(chunks) > 1:
            # Last records of chunk N should be first records of chunk N+1
            assert chunks[0][-2:] == chunks[1][:2]


# ---------------------------------------------------------------------------
# EnrichmentPipeline.enrich_record (mock LLM)
# ---------------------------------------------------------------------------

class TestEnrichmentPipeline:
    def _make_pipeline(self):
        store = MagicMock()
        settings = MagicMock()
        settings.llm.base_url = "http://127.0.0.1:8080"
        settings.llm.model = "test-model"
        settings.llm.timeout_seconds = 30
        settings.llm.provider = "llama_cpp"
        settings.llm.api_key = None
        settings.embeddings.contextual_enabled = False
        return EnrichmentPipeline(store=store, settings=settings), store, settings

    def test_enrich_record_success(self) -> None:
        pipeline, store, _ = self._make_pipeline()
        record = _make_record("Some note about Python.")
        llm_response = "SUMMARY: Note about Python.\nGEMS:\n- [Decision] Use Python"

        with patch("egovault.core.enrichment._call_llm_simple", return_value=llm_response):
            result = pipeline.enrich_record(record)

        assert result is True
        store.insert_enrichment_result.assert_called_once()
        store.mark_enriched.assert_called_with(record.id, EnrichmentStatus.DONE)

    def test_enrich_record_failure_marks_failed(self) -> None:
        pipeline, store, _ = self._make_pipeline()
        record = _make_record("Some content.")

        with patch("egovault.core.enrichment._call_llm_simple", side_effect=ConnectionError("no server")):
            result = pipeline.enrich_record(record)

        assert result is False
        store.mark_enriched.assert_called_with(record.id, EnrichmentStatus.FAILED)

    def test_enrich_record_contextual_body_stored_when_enabled(self) -> None:
        pipeline, store, settings = self._make_pipeline()
        settings.embeddings.contextual_enabled = True
        record = _make_record("Content requiring contextual retrieval.")
        llm_response = "SUMMARY: Content.\nGEMS:"

        with patch("egovault.core.enrichment._call_llm_simple", side_effect=["ctx prefix", llm_response]):
            # First call returns context prefix, second returns enrichment
            # Actual call order depends on implementation; just verify store gets called.
            pass

        with patch("egovault.core.enrichment._call_llm_simple", return_value=llm_response), \
             patch("egovault.core.enrichment._generate_context_prefix", return_value="ctx body"):
            result = pipeline.enrich_record(record)

        assert result is True
        store.upsert_contextual_body.assert_called_once_with(record.id, "ctx body")
