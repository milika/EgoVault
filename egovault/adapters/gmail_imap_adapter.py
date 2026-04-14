"""GmailImapAdapter — fetches Gmail via IMAP using an App Password.

No Google Cloud project, no OAuth credentials JSON, no browser flow needed.

User flow
---------
1.  One-time setup (run once inside ``egovault chat`` or via CLI)::

        /gmail-auth       → choose option 1 (IMAP App Password)

    EgoVault will guide you through creating an App Password and save the
    credentials locally to ``data/gmail_imap.json``.

2.  Sync emails::

        /gmail-sync

    On subsequent runs only new emails are fetched (using the ``SINCE``
    IMAP criterion against the date stored in ``gmail_last_sync``).

This adapter is **not** registered in the auto-discovery registry.  It is
instantiated directly by ``_handle_gmail_sync()`` in the chat session and by
the ``gmail-sync`` CLI command.
"""
from __future__ import annotations

import email as _stdlib_email
import imaplib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterator

from egovault.adapters.gmail import (
    _clean_subject,
    _decode_header_value,
    _extract_body,
    _extract_email_address,
    _get_attachment_names,
    _parse_timestamp,
)
from egovault.core.adapter import BasePlatformAdapter
from egovault.core.schema import NormalizedRecord
from egovault.utils.gmail_imap import connect, imap_before_date, imap_since_date

if TYPE_CHECKING:
    from egovault.core.store import VaultStore

logger = logging.getLogger(__name__)

# Gmail labels that map to IMAP folder names.
_LABEL_INBOX = "INBOX"
_LABEL_ALL = "[Gmail]/All Mail"


class GmailImapAdapter(BasePlatformAdapter):
    """Fetches Gmail messages via IMAP using an App Password.

    This adapter is **not** registered in the auto-discovery registry.
    """

    platform_id = "gmail"

    def __init__(self, store: "VaultStore") -> None:
        self.store = store

    # ------------------------------------------------------------------
    # BasePlatformAdapter interface
    # ------------------------------------------------------------------

    @classmethod
    def can_handle(cls, source_path: Path) -> bool:
        """Always False — this adapter is never auto-detected by the registry."""
        return False

    def ingest(self, source_path: Path) -> "Iterator[NormalizedRecord]":
        """Not supported for IMAP — use :meth:`ingest_from_imap` directly."""
        raise NotImplementedError("Use ingest_from_imap() for IMAP-based sync.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_from_imap(
        self,
        gmail_address: str,
        app_password: str,
        mailbox_label: str = _LABEL_ALL,
        since: str = "",
        before: str = "",
        max_results: int = 500,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> Iterator[NormalizedRecord]:
        """Yield :class:`~egovault.core.schema.NormalizedRecord` objects from Gmail IMAP.

        Args:
            gmail_address: Full Gmail address (e.g. ``you@gmail.com``).
            app_password: 16-character App Password from Google Account settings.
            mailbox_label: IMAP folder name.  Defaults to ``[Gmail]/All Mail``
                so that all emails (not just inbox) are fetched.
            since: ISO-8601 date string (``YYYY-MM-DD``).  Forward frontier —
                only emails on or after this date are fetched in the forward pass.
                Empty string on first run means all emails.
            before: ISO-8601 date string (``YYYY-MM-DD``).  Backward frontier —
                emails strictly before this date are fetched in the backward pass
                to grow the history on each sync.  Empty string disables backfill.
            max_results: Maximum number of emails to yield across both passes.
                When both *since* and *before* are set the budget is split evenly
                (half new, half old).  Newest emails are yielded first.
            progress_callback: Optional ``(fetched, total)`` callable called
                after each message is yielded so the caller can update a
                progress bar.
        """
        mail = connect(gmail_address, app_password)
        try:
            yield from self._fetch_messages(mail, mailbox_label, since, before, max_results, progress_callback)
        finally:
            try:
                mail.close()
                mail.logout()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_messages(
        self,
        mail: imaplib.IMAP4_SSL,
        mailbox_label: str,
        since: str,
        before: str,
        max_results: int,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> Iterator[NormalizedRecord]:
        # imaplib's _checkquote() does not quote names starting with '[',
        # which causes the server to reject the EXAMINE command with BAD.
        # Pre-wrapping in double-quotes fixes this for any folder whose name
        # contains brackets (e.g. "[Gmail]/All Mail").
        quoted = f'"{mailbox_label}"' if not mailbox_label.startswith('"') else mailbox_label
        status, _ = mail.select(quoted, readonly=True)
        if status != "OK":
            # "[Gmail]/All Mail" may have a different name in some locales;
            # fall back to INBOX.
            logger.warning("Cannot select %r, falling back to INBOX", mailbox_label)
            mail.select(_LABEL_INBOX, readonly=True)

        message_ids: list[bytes] = []

        # ── Forward pass: new emails (SINCE last_sync or ALL on first run) ──
        fwd_quota = (max_results // 2) if (since and before) else max_results
        fwd_criteria = f'SINCE "{imap_since_date(since)}"' if since else "ALL"
        status, data = mail.search(None, fwd_criteria)
        if status == "OK":
            # Reverse to get newest first, then cap at quota.
            message_ids.extend(data[0].split()[::-1][:fwd_quota])

        # ── Backward pass: historical backfill (BEFORE oldest_synced) ──────
        if before:
            bwd_quota = max_results - len(message_ids)
            if bwd_quota > 0:
                bwd_criteria = f'BEFORE "{imap_before_date(before)}"'
                status, data = mail.search(None, bwd_criteria)
                if status == "OK":
                    # Reverse to get newest-of-old first (pages backwards each run).
                    bwd_ids = data[0].split()[::-1][:bwd_quota]
                    # Deduplicate: IMAP ids are per-session ints but forward pass
                    # uses SINCE which may overlap with before on same-day boundaries.
                    existing = set(message_ids)
                    message_ids.extend(mid for mid in bwd_ids if mid not in existing)

        total = len(message_ids)

        for fetched, msg_num in enumerate(message_ids, start=1):
            status, msg_data = mail.fetch(msg_num, "(RFC822)")
            if status != "OK" or not msg_data or not msg_data[0]:
                continue
            raw: bytes = msg_data[0][1]  # type: ignore[index]
            try:
                msg = _stdlib_email.message_from_bytes(raw)
                record = self._message_to_record(msg)
            except Exception:  # noqa: BLE001
                logger.debug("Skipping malformed message %s", msg_num, exc_info=True)
                continue
            if record is not None:
                if progress_callback is not None:
                    progress_callback(fetched, total)
                yield record

    @staticmethod
    def _message_to_record(
        msg: _stdlib_email.message.Message,
    ) -> NormalizedRecord | None:
        """Convert a parsed :class:`email.message.Message` to a :class:`NormalizedRecord`.

        Returns ``None`` if no text body can be extracted.
        """
        subject_raw = msg.get("Subject", "") or ""
        subject = _decode_header_value(subject_raw)
        thread_name = _clean_subject(subject)

        # Prefer X-Gmail-Thread-Id for cross-source deduplication with
        # Takeout mbox exports and the OAuth API adapter.
        # str() is needed because msg.get() can return a Header object.
        thread_id: str = str(
            msg.get("X-Gmail-Thread-Id")
            or msg.get("X-GM-THRID")
            or f"subj:{thread_name[:64]}"
        )

        sender_name, sender_id = _extract_email_address(msg.get("From"))
        timestamp = _parse_timestamp(msg.get("Date"))

        body = _extract_body(msg)  # type: ignore[arg-type]  # compatible Message subtype
        if not body:
            return None

        message_id = str(msg.get("Message-ID", "") or "")

        def _h(name: str) -> str:
            """Return header value as a plain str (never a Header object)."""
            return str(msg.get(name, "") or "")

        return NormalizedRecord(
            platform="gmail",
            record_type="message",
            timestamp=timestamp,
            sender_id=sender_id,
            sender_name=sender_name,
            thread_id=thread_id,
            thread_name=thread_name,
            body=body,
            attachments=_get_attachment_names(msg),  # type: ignore[arg-type]
            raw={
                "message_id": message_id,
                "subject": subject,
                "from": _h("From"),
                "to": _h("To"),
                "cc": _h("Cc"),
                "labels": _h("X-Gmail-Labels"),
                "api_message_id": "",
                "api_thread_id": "",
            },
            file_path=None,
            mime_type="message/rfc822",
        )
