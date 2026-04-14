"""Tests for GmailImapAdapter and gmail_imap utilities.

The IMAP connection is fully mocked — no network calls, no credentials.
"""
from __future__ import annotations

import email as _stdlib_email
import imaplib
import json
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from egovault.adapters.gmail_imap_adapter import GmailImapAdapter
from egovault.core.store import VaultStore
from egovault.utils.gmail_imap import (
    CREDENTIALS_FILENAME,
    IMAP_HOST,
    IMAP_PORT,
    get_credentials_path,
    imap_before_date,
    imap_since_date,
    load_credentials,
    save_credentials,
    verify_connection,
)


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
def adapter(store: VaultStore) -> GmailImapAdapter:
    return GmailImapAdapter(store=store)


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    return tmp_path


def _make_raw_imap_message(
    subject: str = "Test email",
    from_: str = "Alice <alice@example.com>",
    to: str = "me@gmail.com",
    date_str: str = "Mon, 05 Jan 2026 09:00:00 +0000",
    body: str = "Hello from IMAP",
    message_id: str = "<msg001@example.com>",
    thread_id: str = "",
) -> bytes:
    """Build a minimal RFC 2822 email as raw bytes (like IMAP RFC822 fetch)."""
    msg = _stdlib_email.message.Message()
    msg["From"] = from_
    msg["To"] = to
    msg["Subject"] = subject
    msg["Date"] = date_str
    msg["Message-ID"] = message_id
    if thread_id:
        msg["X-Gmail-Thread-Id"] = thread_id
    msg.set_payload(body)
    return msg.as_bytes()


def _make_imap_mock(
    messages: list[bytes],
    since: str = "",
    folder_status: str = "OK",
    search_status: str = "OK",
    fetch_status: str = "OK",
) -> MagicMock:
    """Return a mock IMAP4_SSL that yields *messages* as RFC822 fetch results."""
    mail = MagicMock(spec=imaplib.IMAP4_SSL)

    # select()
    mail.select.return_value = (folder_status, [b"1"])

    # search() — return message sequence numbers 1..N
    ids = b" ".join(str(i + 1).encode() for i in range(len(messages)))
    mail.search.return_value = (search_status, [ids])

    # fetch() — return the raw bytes for each message number
    def _fetch(msg_num, _fmt):
        idx = int(msg_num) - 1
        if 0 <= idx < len(messages):
            return (fetch_status, [(b"FLAGS", messages[idx])])
        return ("NO", [None])

    mail.fetch.side_effect = _fetch
    return mail


# ---------------------------------------------------------------------------
# gmail_imap utilities
# ---------------------------------------------------------------------------


class TestImapSinceDate:
    def test_converts_iso_to_imap_format(self):
        assert imap_since_date("2025-01-15") == "15-Jan-2025"

    def test_leading_zero_for_single_digit_day(self):
        assert imap_since_date("2025-03-05") == "05-Mar-2025"

    def test_december(self):
        assert imap_since_date("2026-12-31") == "31-Dec-2026"


class TestSaveAndLoadCredentials:
    def test_round_trip(self, data_dir: Path):
        save_credentials(data_dir, "me@gmail.com", "abcd1234efgh5678")
        result = load_credentials(data_dir)
        assert result == ("me@gmail.com", "abcd1234efgh5678")

    def test_credentials_file_name(self, data_dir: Path):
        save_credentials(data_dir, "a@b.com", "pass123")
        assert (data_dir / CREDENTIALS_FILENAME).exists()

    def test_load_returns_none_when_missing(self, data_dir: Path):
        assert load_credentials(data_dir) is None

    def test_load_returns_none_when_malformed(self, data_dir: Path):
        (data_dir / CREDENTIALS_FILENAME).write_text("not json")
        assert load_credentials(data_dir) is None

    def test_load_returns_none_when_missing_keys(self, data_dir: Path):
        (data_dir / CREDENTIALS_FILENAME).write_text(json.dumps({"only_email": "x"}))
        assert load_credentials(data_dir) is None

    def test_get_credentials_path(self, data_dir: Path):
        assert get_credentials_path(data_dir) == data_dir / CREDENTIALS_FILENAME

    def test_creates_parent_directories(self, tmp_path: Path):
        deep_dir = tmp_path / "a" / "b" / "c"
        save_credentials(deep_dir, "x@gmail.com", "pass")
        assert (deep_dir / CREDENTIALS_FILENAME).exists()


class TestVerifyConnection:
    def test_successful_connection(self):
        with patch("egovault.utils.gmail_imap.imaplib.IMAP4_SSL") as mock_class:
            mock_mail = MagicMock()
            mock_class.return_value = mock_mail
            verify_connection("a@gmail.com", "apppass")
            mock_class.assert_called_once_with(IMAP_HOST, IMAP_PORT)
            mock_mail.login.assert_called_once_with("a@gmail.com", "apppass")
            mock_mail.logout.assert_called_once()

    def test_raises_on_login_failure(self):
        with patch("egovault.utils.gmail_imap.imaplib.IMAP4_SSL") as mock_class:
            mock_mail = MagicMock()
            mock_mail.login.side_effect = imaplib.IMAP4.error("invalid credentials")
            mock_class.return_value = mock_mail
            with pytest.raises(imaplib.IMAP4.error):
                verify_connection("bad@gmail.com", "wrongpass")

    def test_logout_called_even_on_login_failure(self):
        with patch("egovault.utils.gmail_imap.imaplib.IMAP4_SSL") as mock_class:
            mock_mail = MagicMock()
            mock_mail.login.side_effect = imaplib.IMAP4.error("fail")
            mock_class.return_value = mock_mail
            with pytest.raises(imaplib.IMAP4.error):
                verify_connection("a@gmail.com", "bad")
            mock_mail.logout.assert_called_once()


# ---------------------------------------------------------------------------
# GmailImapAdapter._message_to_record
# ---------------------------------------------------------------------------


class TestMessageToRecord:
    def test_basic_conversion(self):
        raw = _make_raw_imap_message()
        msg = _stdlib_email.message_from_bytes(raw)
        record = GmailImapAdapter._message_to_record(msg)
        assert record is not None
        assert record.platform == "gmail"
        assert record.record_type == "message"
        assert record.sender_id == "alice@example.com"
        assert record.sender_name == "Alice"
        assert "Hello from IMAP" in record.body

    def test_subject_cleaned(self):
        raw = _make_raw_imap_message(subject="Re: Re: Meeting tomorrow")
        msg = _stdlib_email.message_from_bytes(raw)
        record = GmailImapAdapter._message_to_record(msg)
        assert record is not None
        assert record.thread_name == "Meeting tomorrow"

    def test_returns_none_for_empty_body(self):
        raw = _make_raw_imap_message(body="")
        msg = _stdlib_email.message_from_bytes(raw)
        record = GmailImapAdapter._message_to_record(msg)
        assert record is None

    def test_thread_id_from_x_gmail_thread_id_header(self):
        raw = _make_raw_imap_message(thread_id="thread999")
        msg = _stdlib_email.message_from_bytes(raw)
        record = GmailImapAdapter._message_to_record(msg)
        assert record is not None
        assert record.thread_id == "thread999"

    def test_thread_id_fallback_when_no_header(self):
        raw = _make_raw_imap_message(subject="Unique Subject XYZ", thread_id="")
        msg = _stdlib_email.message_from_bytes(raw)
        record = GmailImapAdapter._message_to_record(msg)
        assert record is not None
        assert "Unique Subject XYZ" in record.thread_id

    def test_message_id_in_raw(self):
        raw = _make_raw_imap_message(message_id="<unique-msg@domain.com>")
        msg = _stdlib_email.message_from_bytes(raw)
        record = GmailImapAdapter._message_to_record(msg)
        assert record is not None
        assert record.raw["message_id"] == "<unique-msg@domain.com>"

    def test_mime_type(self):
        raw = _make_raw_imap_message()
        msg = _stdlib_email.message_from_bytes(raw)
        record = GmailImapAdapter._message_to_record(msg)
        assert record is not None
        assert record.mime_type == "message/rfc822"

    def test_timestamp_parsed_from_date_header(self):
        raw = _make_raw_imap_message(date_str="Mon, 05 Jan 2026 09:00:00 +0000")
        msg = _stdlib_email.message_from_bytes(raw)
        record = GmailImapAdapter._message_to_record(msg)
        assert record is not None
        assert record.timestamp.year == 2026
        assert record.timestamp.month == 1
        assert record.timestamp.day == 5


# ---------------------------------------------------------------------------
# GmailImapAdapter.ingest_from_imap
# ---------------------------------------------------------------------------


class TestIngestFromImap:
    def _patch_connect(self, mock_mail: MagicMock):
        return patch("egovault.adapters.gmail_imap_adapter.connect", return_value=mock_mail)

    def test_yields_records_for_valid_messages(self, adapter: GmailImapAdapter):
        msgs = [
            _make_raw_imap_message(subject="Email 1", body="Body 1"),
            _make_raw_imap_message(subject="Email 2", body="Body 2", message_id="<m2@x.com>"),
        ]
        mock_mail = _make_imap_mock(msgs)
        with self._patch_connect(mock_mail):
            records = list(adapter.ingest_from_imap("me@gmail.com", "pass"))
        assert len(records) == 2

    def test_skips_empty_body_messages(self, adapter: GmailImapAdapter):
        msgs = [
            _make_raw_imap_message(body=""),
            _make_raw_imap_message(body="Has content"),
        ]
        mock_mail = _make_imap_mock(msgs)
        with self._patch_connect(mock_mail):
            records = list(adapter.ingest_from_imap("me@gmail.com", "pass"))
        assert len(records) == 1

    def test_since_parameter_used_in_search(self, adapter: GmailImapAdapter):
        mock_mail = _make_imap_mock([_make_raw_imap_message()])
        with self._patch_connect(mock_mail):
            list(adapter.ingest_from_imap("me@gmail.com", "pass", since="2025-06-01"))
        mock_mail.search.assert_called_once_with(None, 'SINCE "01-Jun-2025"')

    def test_no_since_searches_all(self, adapter: GmailImapAdapter):
        mock_mail = _make_imap_mock([_make_raw_imap_message()])
        with self._patch_connect(mock_mail):
            list(adapter.ingest_from_imap("me@gmail.com", "pass", since=""))
        mock_mail.search.assert_called_once_with(None, "ALL")

    def test_max_results_caps_results(self, adapter: GmailImapAdapter):
        msgs = [_make_raw_imap_message(body=f"Body {i}", message_id=f"<m{i}@x.com>") for i in range(10)]
        mock_mail = _make_imap_mock(msgs)
        with self._patch_connect(mock_mail):
            records = list(adapter.ingest_from_imap("me@gmail.com", "pass", max_results=3))
        assert len(records) == 3

    def test_selects_all_mail_folder_by_default(self, adapter: GmailImapAdapter):
        mock_mail = _make_imap_mock([_make_raw_imap_message()])
        with self._patch_connect(mock_mail):
            list(adapter.ingest_from_imap("me@gmail.com", "pass"))
        # Folder name is pre-quoted to work around imaplib not quoting brackets.
        mock_mail.select.assert_called_once_with('"[Gmail]/All Mail"', readonly=True)

    def test_falls_back_to_inbox_on_select_error(self, adapter: GmailImapAdapter):
        mock_mail = _make_imap_mock([_make_raw_imap_message()], folder_status="NO")
        with self._patch_connect(mock_mail):
            list(adapter.ingest_from_imap("me@gmail.com", "pass"))
        assert any(
            c == call("INBOX", readonly=True) for c in mock_mail.select.call_args_list
        )

    def test_logout_called_after_sync(self, adapter: GmailImapAdapter):
        mock_mail = _make_imap_mock([_make_raw_imap_message()])
        with self._patch_connect(mock_mail):
            list(adapter.ingest_from_imap("me@gmail.com", "pass"))
        mock_mail.logout.assert_called_once()

    def test_logout_called_even_on_search_failure(self, adapter: GmailImapAdapter):
        mock_mail = _make_imap_mock([], search_status="NO")
        with self._patch_connect(mock_mail):
            list(adapter.ingest_from_imap("me@gmail.com", "pass"))
        mock_mail.logout.assert_called_once()

    def test_empty_vault_all_records_inserted(
        self, adapter: GmailImapAdapter, store: VaultStore
    ):
        msgs = [
            _make_raw_imap_message(subject="A", body="Body A"),
            _make_raw_imap_message(subject="B", body="Body B", message_id="<b@x.com>"),
        ]
        mock_mail = _make_imap_mock(msgs)
        with self._patch_connect(mock_mail):
            for record in adapter.ingest_from_imap("me@gmail.com", "pass"):
                store.upsert_record(record)
        assert len(store.get_records()) == 2

    def test_duplicate_emails_not_inserted_twice(
        self, adapter: GmailImapAdapter, store: VaultStore
    ):
        raw = _make_raw_imap_message()
        mock_mail = _make_imap_mock([raw])
        with self._patch_connect(mock_mail):
            for r in adapter.ingest_from_imap("me@gmail.com", "pass"):
                store.upsert_record(r)
        # Second sync of the same message
        mock_mail2 = _make_imap_mock([raw])
        with self._patch_connect(mock_mail2):
            for r in adapter.ingest_from_imap("me@gmail.com", "pass"):
                store.upsert_record(r)
        assert len(store.get_records()) == 1


# ---------------------------------------------------------------------------
# imap_before_date utility
# ---------------------------------------------------------------------------


class TestImapBeforeDate:
    def test_converts_iso_to_imap_format(self):
        assert imap_before_date("2025-01-15") == "15-Jan-2025"

    def test_leading_zero_for_single_digit_day(self):
        assert imap_before_date("2025-03-05") == "05-Mar-2025"

    def test_december(self):
        assert imap_before_date("2024-12-31") == "31-Dec-2024"


# ---------------------------------------------------------------------------
# Dual-frontier helpers
# ---------------------------------------------------------------------------


def _make_dual_pass_imap_mock(
    fwd_messages: list[bytes],
    bwd_messages: list[bytes],
) -> MagicMock:
    """Return a mock IMAP4_SSL with separate message sets for SINCE vs BEFORE searches.

    Forward messages get seq IDs 1..N.  Backward messages get seq IDs 1000..M so
    the two sets never collide in the dedup logic.
    """
    mail = MagicMock(spec=imaplib.IMAP4_SSL)
    mail.select.return_value = ("OK", [b"1"])

    fwd_ids = b" ".join(str(i + 1).encode() for i in range(len(fwd_messages)))
    bwd_ids = b" ".join(str(1000 + i).encode() for i in range(len(bwd_messages)))

    def _search(_, criteria):
        if "BEFORE" in criteria:
            return ("OK", [bwd_ids])
        # SINCE <date> or ALL → forward ids
        return ("OK", [fwd_ids])

    mail.search.side_effect = _search

    all_messages: dict[int, bytes] = {}
    for i, msg in enumerate(fwd_messages):
        all_messages[i + 1] = msg
    for i, msg in enumerate(bwd_messages):
        all_messages[1000 + i] = msg

    def _fetch(msg_num, _fmt):
        idx = int(msg_num)
        if idx in all_messages:
            return ("OK", [(b"FLAGS", all_messages[idx])])
        return ("NO", [None])

    mail.fetch.side_effect = _fetch
    return mail


# ---------------------------------------------------------------------------
# Dual-frontier: GmailImapAdapter with since + before
# ---------------------------------------------------------------------------


class TestDualFrontierSync:
    def _patch_connect(self, mock_mail: MagicMock):
        return patch("egovault.adapters.gmail_imap_adapter.connect", return_value=mock_mail)

    def test_two_search_calls_when_both_frontiers_set(self, adapter: GmailImapAdapter):
        """Both SINCE and BEFORE searches are issued when both frontiers are active."""
        fwd = [_make_raw_imap_message(subject="New 1", message_id="<n1@x.com>")]
        bwd = [_make_raw_imap_message(subject="Old 1", message_id="<o1@x.com>")]
        mock_mail = _make_dual_pass_imap_mock(fwd, bwd)
        with self._patch_connect(mock_mail):
            list(adapter.ingest_from_imap(
                "me@gmail.com", "pass",
                since="2026-04-01",
                before="2026-01-01",
            ))
        assert mock_mail.search.call_count == 2

    def test_since_and_before_criteria_strings(self, adapter: GmailImapAdapter):
        """SINCE and BEFORE receive correctly formatted IMAP date strings."""
        mock_mail = _make_dual_pass_imap_mock([], [])
        with self._patch_connect(mock_mail):
            list(adapter.ingest_from_imap(
                "me@gmail.com", "pass",
                since="2026-04-01",
                before="2026-01-01",
            ))
        calls = [str(c) for c in mock_mail.search.call_args_list]
        assert any("SINCE\"" in c or 'SINCE "' in c for c in calls)
        assert any("BEFORE\"" in c or 'BEFORE "' in c for c in calls)

    def test_yields_records_from_both_passes(self, adapter: GmailImapAdapter):
        """Records from both forward and backward passes are yielded."""
        fwd = [_make_raw_imap_message(subject="New", message_id="<n@x.com>")]
        bwd = [_make_raw_imap_message(subject="Old", message_id="<o@x.com>")]
        mock_mail = _make_dual_pass_imap_mock(fwd, bwd)
        with self._patch_connect(mock_mail):
            records = list(adapter.ingest_from_imap(
                "me@gmail.com", "pass",
                since="2026-04-01",
                before="2026-01-01",
                max_results=10,
            ))
        assert len(records) == 2

    def test_budget_split_evenly_across_both_passes(self, adapter: GmailImapAdapter):
        """With budget=6, each pass gets at most 3 messages."""
        fwd = [_make_raw_imap_message(body=f"New {i}", message_id=f"<n{i}@x.com>") for i in range(5)]
        bwd = [_make_raw_imap_message(body=f"Old {i}", message_id=f"<o{i}@x.com>") for i in range(5)]
        mock_mail = _make_dual_pass_imap_mock(fwd, bwd)
        with self._patch_connect(mock_mail):
            records = list(adapter.ingest_from_imap(
                "me@gmail.com", "pass",
                since="2026-04-01",
                before="2026-01-01",
                max_results=6,
            ))
        assert len(records) == 6

    def test_backward_budget_uses_remainder_after_forward(self, adapter: GmailImapAdapter):
        """If forward pass finds fewer messages than its quota, backward gets the leftover."""
        fwd = [_make_raw_imap_message(body="New 1", message_id="<n1@x.com>")]  # only 1
        bwd = [_make_raw_imap_message(body=f"Old {i}", message_id=f"<o{i}@x.com>") for i in range(5)]
        mock_mail = _make_dual_pass_imap_mock(fwd, bwd)
        with self._patch_connect(mock_mail):
            records = list(adapter.ingest_from_imap(
                "me@gmail.com", "pass",
                since="2026-04-01",
                before="2026-01-01",
                max_results=6,  # fwd_quota=3, gets 1; bwd_quota=5, gets 5
            ))
        assert len(records) == 6

    def test_only_one_search_when_before_is_empty(self, adapter: GmailImapAdapter):
        """No backward pass when before=''; all budget goes to forward."""
        msgs = [_make_raw_imap_message()]
        mock_mail = _make_imap_mock(msgs)
        with self._patch_connect(mock_mail):
            list(adapter.ingest_from_imap(
                "me@gmail.com", "pass",
                since="2026-04-01",
                before="",
            ))
        assert mock_mail.search.call_count == 1

    def test_full_budget_to_forward_when_no_before(self, adapter: GmailImapAdapter):
        """All max_results slots go to the forward pass when before is empty."""
        msgs = [_make_raw_imap_message(body=f"Body {i}", message_id=f"<m{i}@x.com>") for i in range(10)]
        mock_mail = _make_imap_mock(msgs)
        with self._patch_connect(mock_mail):
            records = list(adapter.ingest_from_imap(
                "me@gmail.com", "pass",
                since="2026-04-01",
                before="",
                max_results=10,
            ))
        assert len(records) == 10

    def test_no_duplicate_records_across_passes(
        self, adapter: GmailImapAdapter, store: VaultStore
    ):
        """Message IDs from different passes do not collide in the vault."""
        fwd = [_make_raw_imap_message(subject="New", message_id="<n@x.com>", body="new body")]
        bwd = [_make_raw_imap_message(subject="Old", message_id="<o@x.com>", body="old body")]
        mock_mail = _make_dual_pass_imap_mock(fwd, bwd)
        with self._patch_connect(mock_mail):
            for r in adapter.ingest_from_imap(
                "me@gmail.com", "pass",
                since="2026-04-01",
                before="2026-01-01",
                max_results=10,
            ):
                store.upsert_record(r)
        assert len(store.get_records()) == 2

    def test_oldest_synced_setting_updated_after_backward_pass(
        self, adapter: GmailImapAdapter, store: VaultStore
    ):
        """After syncing old emails, gmail_oldest_synced is pushed back in time."""
        # Old email dated 2025-01-10
        bwd = [_make_raw_imap_message(
            subject="Old", message_id="<o@x.com>",
            date_str="Fri, 10 Jan 2025 08:00:00 +0000",
        )]
        mock_mail = _make_dual_pass_imap_mock([], bwd)
        with self._patch_connect(mock_mail):
            for r in adapter.ingest_from_imap(
                "me@gmail.com", "pass",
                since="",
                before="2025-02-01",
                max_results=10,
            ):
                was_new = store.upsert_record(r)
                if was_new and r.timestamp:
                    current = store.get_setting("gmail_oldest_synced") or ""
                    ts_str = r.timestamp.strftime("%Y-%m-%d")
                    if not current or ts_str < current:
                        store.set_setting("gmail_oldest_synced", ts_str)
        assert store.get_setting("gmail_oldest_synced") == "2025-01-10"
