"""Unit tests for GmailAdapter (Google Takeout .mbox imports)."""
from __future__ import annotations

import mailbox
from pathlib import Path

import pytest

from egovault.adapters.gmail import (
    GmailAdapter,
    _clean_subject,
    _extract_email_address,
    _parse_timestamp,
)
from egovault.core.schema import NormalizedRecord
from egovault.core.store import VaultStore

FIXTURES = Path(__file__).parent / "fixtures" / "gmail"
MBOX_FILE = FIXTURES / "test_export.mbox"


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
def adapter(store: VaultStore) -> GmailAdapter:
    return GmailAdapter(store=store)


def _ingest_all(adapter: GmailAdapter, path: Path) -> list[NormalizedRecord]:
    return list(adapter.ingest(path))


# ---------------------------------------------------------------------------
# can_handle()
# ---------------------------------------------------------------------------

class TestCanHandle:
    def test_accepts_mbox_file(self, adapter: GmailAdapter) -> None:
        assert adapter.can_handle(MBOX_FILE) is True

    def test_rejects_directory(self, adapter: GmailAdapter, tmp_path: Path) -> None:
        assert adapter.can_handle(tmp_path) is False

    def test_rejects_wrong_extension(self, adapter: GmailAdapter, tmp_path: Path) -> None:
        f = tmp_path / "messages.txt"
        f.write_text("From alice@example.com Mon Jan 01 00:00:00 2024\n")
        assert adapter.can_handle(f) is False

    def test_rejects_mbox_extension_without_from_line(
        self, adapter: GmailAdapter, tmp_path: Path
    ) -> None:
        f = tmp_path / "empty.mbox"
        f.write_text("this is not an mbox file\n")
        assert adapter.can_handle(f) is False

    def test_rejects_nonexistent_file(self, adapter: GmailAdapter, tmp_path: Path) -> None:
        assert adapter.can_handle(tmp_path / "missing.mbox") is False


# ---------------------------------------------------------------------------
# ingest() — message count and None filtering
# ---------------------------------------------------------------------------

class TestIngestRecordCount:
    def test_yields_three_records_skips_empty(self, adapter: GmailAdapter) -> None:
        # Fixture has 4 messages; 1 has an empty body and must be skipped
        records = _ingest_all(adapter, MBOX_FILE)
        assert len(records) == 3

    def test_all_records_are_normalized_record_instances(
        self, adapter: GmailAdapter
    ) -> None:
        records = _ingest_all(adapter, MBOX_FILE)
        assert all(isinstance(r, NormalizedRecord) for r in records)


# ---------------------------------------------------------------------------
# ingest() — field mapping
# ---------------------------------------------------------------------------

class TestFieldMapping:
    def _get_plain_text_record(self, adapter: GmailAdapter) -> NormalizedRecord:
        records = _ingest_all(adapter, MBOX_FILE)
        # First record: plain text message from alice
        return records[0]

    def test_platform_is_gmail(self, adapter: GmailAdapter) -> None:
        r = self._get_plain_text_record(adapter)
        assert r.platform == "gmail"

    def test_record_type_is_message(self, adapter: GmailAdapter) -> None:
        r = self._get_plain_text_record(adapter)
        assert r.record_type == "message"

    def test_sender_id_is_normalized_email(self, adapter: GmailAdapter) -> None:
        r = self._get_plain_text_record(adapter)
        assert r.sender_id == "alice@example.com"

    def test_sender_name_extracted(self, adapter: GmailAdapter) -> None:
        r = self._get_plain_text_record(adapter)
        assert r.sender_name == "Alice Smith"

    def test_thread_id_from_gmail_header(self, adapter: GmailAdapter) -> None:
        r = self._get_plain_text_record(adapter)
        assert r.thread_id == "17f1a2b3c4d5e6f7"

    def test_thread_name_is_cleaned_subject(self, adapter: GmailAdapter) -> None:
        r = self._get_plain_text_record(adapter)
        assert r.thread_name == "Project kickoff meeting"

    def test_reply_thread_name_strips_re_prefix(self, adapter: GmailAdapter) -> None:
        records = _ingest_all(adapter, MBOX_FILE)
        reply = records[1]  # "Re: Project kickoff meeting"
        assert reply.thread_name == "Project kickoff meeting"

    def test_reply_shares_thread_id_with_original(self, adapter: GmailAdapter) -> None:
        records = _ingest_all(adapter, MBOX_FILE)
        assert records[0].thread_id == records[1].thread_id

    def test_timestamp_is_timezone_aware(self, adapter: GmailAdapter) -> None:
        r = self._get_plain_text_record(adapter)
        assert r.timestamp.tzinfo is not None

    def test_timestamp_year_correct(self, adapter: GmailAdapter) -> None:
        r = self._get_plain_text_record(adapter)
        assert r.timestamp.year == 2026

    def test_body_contains_content(self, adapter: GmailAdapter) -> None:
        r = self._get_plain_text_record(adapter)
        assert "kickoff" in r.body.lower()

    def test_mime_type(self, adapter: GmailAdapter) -> None:
        r = self._get_plain_text_record(adapter)
        assert r.mime_type == "message/rfc822"

    def test_file_path_is_mbox_path(self, adapter: GmailAdapter) -> None:
        r = self._get_plain_text_record(adapter)
        assert r.file_path == str(MBOX_FILE)

    def test_raw_contains_labels(self, adapter: GmailAdapter) -> None:
        r = self._get_plain_text_record(adapter)
        assert "Important" in r.raw["labels"]

    def test_raw_contains_message_id(self, adapter: GmailAdapter) -> None:
        r = self._get_plain_text_record(adapter)
        assert r.raw["message_id"] == "<msg001@example.com>"

    def test_multipart_body_prefers_plain_text(self, adapter: GmailAdapter) -> None:
        records = _ingest_all(adapter, MBOX_FILE)
        newsletter = records[2]  # multipart message
        assert "dataclasses" in newsletter.body.lower()


# ---------------------------------------------------------------------------
# Body extraction — HTML fallback
# ---------------------------------------------------------------------------

class TestHtmlFallback:
    def test_html_only_message_has_text_body(self, tmp_path: Path) -> None:
        mbox_path = tmp_path / "html_only.mbox"
        mb = mailbox.mbox(str(mbox_path), create=True)
        msg = mailbox.mboxMessage()
        msg["From"] = "sender@example.com"
        msg["To"] = "me@gmail.com"
        msg["Subject"] = "HTML only test"
        msg["Date"] = "Mon, 05 Jan 2026 09:00:00 +0000"
        msg["X-Gmail-Labels"] = "Inbox"
        msg["X-Gmail-Thread-Id"] = "aabbccdd11223344"
        msg.set_type("text/html")
        msg.set_payload("<html><body><p>Hello from HTML</p></body></html>")
        mb.add(msg)
        mb.flush()
        mb.close()

        adapter = GmailAdapter()
        records = list(adapter.ingest(mbox_path))
        assert len(records) == 1
        assert "hello from html" in records[0].body.lower()


# ---------------------------------------------------------------------------
# Deduplication — re-importing same mbox is a safe no-op
# ---------------------------------------------------------------------------

class TestDeduplication:
    def test_reimport_produces_no_new_records(self, store: VaultStore) -> None:
        adapter = GmailAdapter(store=store)
        first_pass = _ingest_all(adapter, MBOX_FILE)
        for rec in first_pass:
            store.upsert_record(rec)

        second_pass = _ingest_all(adapter, MBOX_FILE)
        new_insertions = sum(1 for rec in second_pass if store.upsert_record(rec))
        assert new_insertions == 0


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------

class TestCleanSubject:
    def test_strips_re_prefix(self) -> None:
        assert _clean_subject("Re: Hello world") == "Hello world"

    def test_strips_fwd_prefix(self) -> None:
        assert _clean_subject("Fwd: Hello world") == "Hello world"

    def test_strips_multiple_prefixes(self) -> None:
        assert _clean_subject("Re: Re: Hello") == "Hello"

    def test_strips_re_with_count(self) -> None:
        assert _clean_subject("Re[3]: Hello") == "Hello"

    def test_empty_subject_returns_placeholder(self) -> None:
        assert _clean_subject("") == "(no subject)"

    def test_no_prefix_unchanged(self) -> None:
        assert _clean_subject("Meeting notes") == "Meeting notes"


class TestExtractEmailAddress:
    def test_parses_display_name_and_address(self) -> None:
        name, addr = _extract_email_address("Alice Smith <alice@example.com>")
        assert name == "Alice Smith"
        assert addr == "alice@example.com"

    def test_bare_address_falls_back_name_to_addr(self) -> None:
        name, addr = _extract_email_address("alice@example.com")
        assert addr == "alice@example.com"
        assert name == "alice@example.com"

    def test_none_returns_unknown(self) -> None:
        name, addr = _extract_email_address(None)
        assert name == "unknown"

    def test_empty_string_returns_unknown(self) -> None:
        _, addr = _extract_email_address("")
        assert addr == "unknown@unknown"

    def test_address_lowercased(self) -> None:
        _, addr = _extract_email_address("Alice <ALICE@EXAMPLE.COM>")
        assert addr == "alice@example.com"


class TestParseTimestamp:
    def test_valid_date_parsed(self) -> None:
        dt = _parse_timestamp("Mon, 05 Jan 2026 09:00:00 +0000")
        assert dt.year == 2026
        assert dt.month == 1
        assert dt.day == 5

    def test_result_is_timezone_aware(self) -> None:
        dt = _parse_timestamp("Mon, 05 Jan 2026 09:00:00 +0000")
        assert dt.tzinfo is not None

    def test_none_falls_back_to_now(self) -> None:
        dt = _parse_timestamp(None)
        assert dt.tzinfo is not None
        assert dt.year >= 2026

    def test_invalid_string_falls_back_to_now(self) -> None:
        dt = _parse_timestamp("not a date")
        assert dt.tzinfo is not None
