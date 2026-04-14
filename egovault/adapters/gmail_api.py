"""GmailApiAdapter — fetches Gmail via OAuth2 + Gmail REST API.

User flow
---------
1.  One-time setup::

        egovault gmail-auth --credentials ~/Downloads/client_secret_*.json

    A browser window opens, the user grants permission, and the token is
    saved to ``data/gmail_token.json``.

2.  Sync emails::

        egovault gmail-sync                    # fetch up to 500 most recent
        egovault gmail-sync --max-results 2000 # fetch more
        egovault gmail-sync --since 2025-01-01 # only emails after a date
        egovault gmail-sync --query "from:boss@acme.com"  # Gmail search query

The adapter is **not** registered in the auto-discovery registry and is
**not** triggered by ``egovault ingest``.  It is intentionally separate
because it requires a live network connection and OAuth credentials while
``ingest`` is designed for offline file imports.

Re-running ``gmail-sync`` is always safe — records are deduplicated by their
deterministic SHA-256 id, so the same email is never inserted twice.
"""
from __future__ import annotations

import base64
import email as _stdlib_email
import logging
import mailbox
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

if TYPE_CHECKING:
    from egovault.core.store import VaultStore

logger = logging.getLogger(__name__)

# Gmail API page size — max allowed by the API is 500; keep a comfortable default.
_PAGE_SIZE: int = 100


class GmailApiAdapter(BasePlatformAdapter):
    """Fetches Gmail messages via the Gmail REST API (OAuth2).

    This adapter is **not** registered in the auto-discovery registry —
    it is instantiated directly by the ``gmail-sync`` CLI command.

    ``ingest(source_path)`` accepts the OAuth token file path as
    *source_path* so the class conforms to the ``BasePlatformAdapter``
    interface.  For richer control (query, max_results, progress), use
    :meth:`ingest_from_api` directly.
    """

    platform_id = "gmail"

    def __init__(self, store: "VaultStore | None" = None) -> None:
        self._store = store

    # ------------------------------------------------------------------
    # BasePlatformAdapter interface
    # ------------------------------------------------------------------

    @classmethod
    def can_handle(cls, source_path: Path) -> bool:
        """Always False — this adapter is never auto-detected by the registry."""
        return False

    def ingest(self, source_path: Path) -> Iterator[NormalizedRecord]:
        """Conform to ``BasePlatformAdapter.ingest``.

        *source_path* is interpreted as the OAuth token file path.
        Use :meth:`ingest_from_api` for full option control.
        """
        yield from self.ingest_from_api(token_path=source_path)

    # ------------------------------------------------------------------
    # Primary API-backed sync method
    # ------------------------------------------------------------------

    def ingest_from_api(
        self,
        token_path: Path,
        query: str = "",
        max_results: int = 500,
        progress_callback: Callable[[int], None] | None = None,
    ) -> Iterator[NormalizedRecord]:
        """Fetch emails from the Gmail API and yield NormalizedRecords.

        Args:
            token_path: Path to the saved OAuth2 token JSON (created by
                ``egovault gmail-auth``).
            query: Gmail search query string.  Defaults to
                ``"-in:spam -in:trash"`` (all mail except spam and trash).
                Gmail query syntax: https://support.google.com/mail/answer/7190
            max_results: Maximum number of emails to fetch.  Gmail API
                pagination is handled automatically.
            progress_callback: Optional callable invoked with the count of
                emails fetched so far after each message.

        Yields:
            One :class:`~egovault.core.schema.NormalizedRecord` per email
            that has an extractable text body.  Emails with no text content
            (e.g. calendar invites with only HTML) are silently skipped.

        Raises:
            RuntimeError: If the token file is absent or invalid (user must
                re-run ``egovault gmail-auth``).
            ImportError: If Gmail dependencies are missing from the current
                environment (install with ``pip install egovault`` or
                reinstall Gmail deps manually).
        """
        from egovault.utils.gmail_auth import build_service, load_credentials

        creds = load_credentials(token_path)
        if creds is None:
            raise RuntimeError(
                "Not authenticated with Gmail.\n"
                "Run:  egovault gmail-auth --credentials path/to/client_secret.json\n"
                f"Token path: {token_path}"
            )

        service = build_service(creds)
        effective_query = query.strip() or "-in:spam -in:trash"
        logger.debug("Gmail API sync: query=%r max_results=%d", effective_query, max_results)

        fetched = 0
        for msg_id, thread_id_api in self._list_message_ids(
            service, effective_query, max_results
        ):
            record = self._fetch_and_convert(service, msg_id, thread_id_api)
            if record is not None:
                yield record
            fetched += 1
            if progress_callback is not None:
                progress_callback(fetched)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _list_message_ids(
        self,
        service: object,
        query: str,
        max_results: int,
    ) -> Iterator[tuple[str, str]]:
        """Yield ``(message_id, thread_id)`` pairs from the Gmail API.

        Pages through API results automatically until *max_results* is
        reached or there are no more messages.
        """
        fetched = 0
        page_token: str | None = None

        while fetched < max_results:
            batch_size = min(_PAGE_SIZE, max_results - fetched)
            params: dict = {
                "userId": "me",
                "q": query,
                "maxResults": batch_size,
            }
            if page_token:
                params["pageToken"] = page_token

            result = service.users().messages().list(**params).execute()  # type: ignore[attr-defined]
            messages = result.get("messages", [])
            for m in messages:
                yield m["id"], m.get("threadId", "")
                fetched += 1
                if fetched >= max_results:
                    return

            page_token = result.get("nextPageToken")
            if not page_token:
                break

    def _fetch_and_convert(
        self,
        service: object,
        msg_id: str,
        thread_id_api: str,
    ) -> NormalizedRecord | None:
        """Fetch one message in raw RFC 2822 format and return a NormalizedRecord.

        Returns ``None`` if the message has no usable text body or if the
        API call fails.
        """
        try:
            result = service.users().messages().get(  # type: ignore[attr-defined]
                userId="me", id=msg_id, format="raw"
            ).execute()
        except Exception as exc:
            logger.warning("Failed to fetch Gmail message %s: %s", msg_id, exc)
            return None

        # The raw field is base64url-encoded RFC 2822 bytes.
        # Pad to a multiple of 4 to satisfy base64.urlsafe_b64decode.
        raw_b64 = result["raw"].encode("ascii")
        raw_bytes = base64.urlsafe_b64decode(raw_b64 + b"=" * (-len(raw_b64) % 4))

        msg = _stdlib_email.message_from_bytes(raw_bytes)
        mbox_msg = mailbox.mboxMessage(msg)

        return self._message_to_record(mbox_msg, msg_id, thread_id_api or None)

    def _message_to_record(
        self,
        msg: mailbox.mboxMessage,
        api_msg_id: str,
        thread_id_override: str | None = None,
    ) -> NormalizedRecord | None:
        """Convert an mboxMessage to a NormalizedRecord.

        Returns ``None`` if no text body can be extracted.
        """
        subject_raw = msg.get("Subject", "") or ""
        subject = _decode_header_value(subject_raw)
        thread_name = _clean_subject(subject)

        # Prefer the Gmail API thread ID (matches X-Gmail-Thread-Id header in
        # Takeout exports, so mbox and API records for the same email deduplicate).
        thread_id = (
            thread_id_override
            or msg.get("X-Gmail-Thread-Id")
            or msg.get("X-GM-THRID")
            or f"subj:{thread_name[:64]}"
        )

        sender_name, sender_id = _extract_email_address(msg.get("From"))
        timestamp = _parse_timestamp(msg.get("Date"))

        body = _extract_body(msg)
        if not body:
            return None

        return NormalizedRecord(
            platform="gmail",
            record_type="message",
            timestamp=timestamp,
            sender_id=sender_id,
            sender_name=sender_name,
            thread_id=str(thread_id),
            thread_name=thread_name,
            body=body,
            attachments=_get_attachment_names(msg),
            raw={
                "message_id": msg.get("Message-ID", "") or api_msg_id,
                "subject": subject,
                "from": msg.get("From", ""),
                "to": msg.get("To", ""),
                "cc": msg.get("Cc", ""),
                "labels": msg.get("X-Gmail-Labels", ""),
                "api_message_id": api_msg_id,
                "api_thread_id": thread_id_override or "",
            },
            file_path=None,
            mime_type="message/rfc822",
        )
