"""GmailAdapter — ingests Gmail exports from Google Takeout (.mbox format).

Usage
-----
1. Go to https://takeout.google.com and request a Gmail export.
2. Download and extract the archive — locate ``*.mbox`` file(s) inside
   the ``Mail/`` folder (e.g. ``All mail Including Spam and Trash.mbox``).
3. Run:  egovault ingest "path/to/All mail Including Spam and Trash.mbox"

No OAuth, no API key, no internet connection required at ingest time.
All processing is local.
"""
from __future__ import annotations

import email.header
import email.utils
import mailbox
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

from egovault.core.adapter import BasePlatformAdapter
from egovault.core.registry import register
from egovault.core.schema import NormalizedRecord

if TYPE_CHECKING:
    from egovault.core.store import VaultStore


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_mbox_file(path: Path) -> bool:
    """Return True if *path* looks like a valid mbox file.

    Checks that the file starts with the ``From `` separator line that
    every standard mbox file must begin with (RFC 4155).
    """
    try:
        with path.open("rb") as fh:
            head = fh.read(512).decode("utf-8", errors="replace")
    except OSError:
        return False
    return head.startswith("From ")


def _decode_header_value(raw: str | None) -> str:
    """Decode an RFC 2047-encoded email header value to a plain string."""
    if not raw:
        return ""
    parts: list[str] = []
    for segment, charset in email.header.decode_header(raw):
        if isinstance(segment, bytes):
            parts.append(segment.decode(charset or "utf-8", errors="replace"))
        else:
            parts.append(segment)
    return " ".join(parts)


_RE_REPLY_PREFIX = re.compile(
    r"^(Re|Fwd|FW|RE|FWD|Fw)\s*(\[\d+\])?\s*:\s*",
    flags=re.IGNORECASE,
)
_RE_WHITESPACE = re.compile(r"\s+")


def _clean_subject(subject: str) -> str:
    """Strip Re:/Fwd: prefixes and collapse whitespace."""
    subject = subject.strip()
    while _RE_REPLY_PREFIX.match(subject):
        subject = _RE_REPLY_PREFIX.sub("", subject).strip()
    return _RE_WHITESPACE.sub(" ", subject) or "(no subject)"


def _extract_email_address(header_value: str | None) -> tuple[str, str]:
    """Return ``(display_name, email_address)`` parsed from a From/To header."""
    if not header_value:
        return ("unknown", "unknown@unknown")
    name, addr = email.utils.parseaddr(header_value)
    addr = addr.lower().strip() or "unknown@unknown"
    name = name.strip() or addr
    return name, addr


def _parse_timestamp(date_str: str | None) -> datetime:
    """Parse an email ``Date`` header into a timezone-aware ``datetime``.

    Falls back to UTC now if the header is absent or malformed.
    Always returns a timezone-aware instance — naive datetimes from malformed
    Date headers are coerced to UTC.
    """
    if date_str:
        try:
            dt = email.utils.parsedate_to_datetime(date_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            pass
    return datetime.now(tz=timezone.utc)


def _extract_body(msg: mailbox.mboxMessage) -> str:
    """Return the best plain-text body from an email message.

    Preference order:
    1. ``text/plain`` parts (concatenated)
    2. ``text/html`` parts stripped via BeautifulSoup (or fallback regex)
    3. Empty string (message will be skipped by the adapter)
    """
    plain_parts: list[str] = []
    html_parts: list[str] = []

    parts = list(msg.walk()) if msg.is_multipart() else [msg]
    for part in parts:
        content_type = part.get_content_type()
        content_disposition = str(part.get("Content-Disposition", ""))
        if "attachment" in content_disposition:
            continue

        charset = part.get_content_charset() or "utf-8"
        try:
            payload = part.get_payload(decode=True)
        except Exception:
            continue
        if payload is None:
            continue

        try:
            text = payload.decode(charset, errors="replace")
        except (LookupError, UnicodeDecodeError):
            text = payload.decode("utf-8", errors="replace")

        if content_type == "text/plain":
            plain_parts.append(text)
        elif content_type == "text/html":
            html_parts.append(text)

    if plain_parts:
        return "\n".join(plain_parts).strip()

    if html_parts:
        combined = "\n".join(html_parts)
        try:
            from bs4 import BeautifulSoup  # type: ignore[import-untyped]
            return BeautifulSoup(combined, "html.parser").get_text(
                separator="\n", strip=True
            )
        except ImportError:
            # Minimal fallback: strip HTML tags with a regex
            return re.sub(r"<[^>]+>", " ", combined).strip()

    return ""


def _get_attachment_names(msg: mailbox.mboxMessage) -> list[str]:
    """Return filenames of all attachment parts in the message."""
    names: list[str] = []
    for part in msg.walk():
        if "attachment" in str(part.get("Content-Disposition", "")):
            filename = part.get_filename()
            if filename:
                names.append(_decode_header_value(filename))
    return names


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

@register
class GmailAdapter(BasePlatformAdapter):
    """Adapter for Gmail exports from Google Takeout (.mbox files).

    Detection
    ---------
    Matches any ``.mbox`` file whose first 512 bytes start with the standard
    ``From `` mbox separator line (RFC 4155).  The ``.mbox`` extension alone
    is sufficient to distinguish Gmail Takeout files from Thunderbird exports
    (which have no extension by default).

    Field mapping
    -------------
    =================== ================================
    Email field         NormalizedRecord field
    =================== ================================
    platform            ``"gmail"``
    record_type         ``"message"``
    ``From``            ``sender_name``, ``sender_id``
    ``Date``            ``timestamp``
    ``X-Gmail-Thread-Id`` / ``X-GM-THRID`` ``thread_id``
    ``Subject`` (cleaned) ``thread_name``
    ``text/plain`` body ``body``
    attachment names    ``attachments``
    ``X-Gmail-Labels``, ``Message-ID``, ``To``, ``Cc`` ``raw``
    =================== ================================
    """

    platform_id = "gmail"

    def __init__(self, store: "VaultStore | None" = None) -> None:
        self._store = store

    @classmethod
    def can_handle(cls, source_path: Path) -> bool:
        """Return True for ``.mbox`` files that start with the standard mbox header."""
        if not source_path.is_file():
            return False
        if source_path.suffix.lower() != ".mbox":
            return False
        return _is_mbox_file(source_path)

    def ingest(self, source_path: Path) -> Iterator[NormalizedRecord]:
        """Yield one :class:`NormalizedRecord` per email in the mbox file.

        Empty messages (no extractable body) are silently skipped.
        The standard record-level SHA-256 deduplication (``INSERT OR IGNORE``)
        in the store ensures that re-running on the same or an updated export
        never inserts duplicate records.
        """
        mbox = mailbox.mbox(str(source_path), factory=None, create=False)
        try:
            for msg in mbox:
                record = self._message_to_record(msg, source_path)
                if record is not None:
                    yield record
        finally:
            mbox.close()

    def _message_to_record(
        self,
        msg: mailbox.mboxMessage,
        source_path: Path,
    ) -> NormalizedRecord | None:
        """Convert a single :class:`mailbox.mboxMessage` to a NormalizedRecord.

        Returns ``None`` if the message has no extractable body.
        """
        subject_raw = msg.get("Subject", "") or ""
        subject = _decode_header_value(subject_raw)
        thread_name = _clean_subject(subject)

        # Prefer Gmail's own thread ID header for correct thread grouping
        thread_id = (
            msg.get("X-Gmail-Thread-Id")
            or msg.get("X-GM-THRID")
            or f"subj:{thread_name[:64]}"
        )

        sender_name, sender_id = _extract_email_address(msg.get("From"))
        timestamp = _parse_timestamp(msg.get("Date"))

        body = _extract_body(msg)
        if not body:
            return None  # nothing useful to store

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
                "message_id": msg.get("Message-ID", ""),
                "subject": subject,
                "from": msg.get("From", ""),
                "to": msg.get("To", ""),
                "cc": msg.get("Cc", ""),
                "labels": msg.get("X-Gmail-Labels", ""),
                "mbox_path": str(source_path),
            },
            file_path=str(source_path),
            mime_type="message/rfc822",
        )
