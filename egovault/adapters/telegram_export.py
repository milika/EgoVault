"""TelegramExportAdapter — ingests Telegram chat history exports.

Usage
-----
1. In Telegram desktop: Settings → Advanced → Export Telegram Data
   → select "Personal chats", "Group chats", or all
   → Format: Machine-readable JSON
   → Download the archive and extract it
2. Locate the ``result.json`` file (or the export root folder).
3. Run:  egovault ingest "path/to/Telegram Desktop/result.json"
   or:   egovault ingest "path/to/Telegram Desktop/"   (auto-detected)

Each conversation becomes one record per message (same as Gmail adapter).
Media captions are included as body text.  File/photo attachments are
recorded in the attachments list.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from egovault.core.adapter import BasePlatformAdapter
from egovault.core.registry import register
from egovault.core.schema import NormalizedRecord

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_dt(value: str | int | None) -> datetime:
    """Parse a Telegram timestamp string (ISO 8601) or Unix int."""
    if value is None:
        return datetime.now(tz=timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(int(value), tz=timezone.utc)
    # Telegram exports use "2026-04-13T13:21:00" (no tz) — treat as UTC
    try:
        dt = datetime.fromisoformat(str(value))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return datetime.now(tz=timezone.utc)


def _text_content(text_field) -> str:  # type: ignore[return]
    """Telegram's ``text`` field can be a plain string or a list of
    mixed string/entity dicts.  Flatten to plain text.
    """
    if isinstance(text_field, str):
        return text_field
    if isinstance(text_field, list):
        parts: list[str] = []
        for item in text_field:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                # Entity objects have a "text" key
                parts.append(item.get("text", ""))
        return "".join(parts)
    return str(text_field) if text_field else ""


def _attachment_name(msg: dict) -> str | None:
    """Return a human-readable attachment filename if the message has media."""
    # photo or sticker stored as a relative path
    for key in ("photo", "file", "thumbnail", "sticker", "video_file", "audio_file",
                "voice_file", "video_message"):
        val = msg.get(key)
        if val and isinstance(val, str):
            return Path(val).name
    return None


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

@register
class TelegramExportAdapter(BasePlatformAdapter):
    """Adapter for Telegram Desktop JSON exports (result.json)."""

    @property
    def platform_id(self) -> str:
        return "telegram"

    @classmethod
    def can_handle(cls, source_path: Path) -> bool:
        """Accept result.json directly, or a directory containing one."""
        if source_path.is_file():
            return source_path.name == "result.json" and _is_telegram_export(source_path)
        if source_path.is_dir():
            candidate = source_path / "result.json"
            return candidate.exists() and _is_telegram_export(candidate)
        return False

    def ingest(self, source_path: Path) -> Iterator[NormalizedRecord]:
        json_path = (
            source_path if source_path.is_file() else source_path / "result.json"
        )
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("Could not read Telegram export %s: %s", json_path, exc)
            return

        # Export owner name — used as sender for "out" messages
        owner_name: str = data.get("personal_information", {}).get("first_name", "")
        if last := data.get("personal_information", {}).get("last_name", ""):
            owner_name = f"{owner_name} {last}".strip()
        owner_name = owner_name or "me"

        chats = data.get("chats", {}).get("list", [])
        if not chats:
            # Some Telegram exports put everything under a top-level "messages" key
            # when a single conversation is exported.
            if "messages" in data:
                chats = [data]
            else:
                logger.warning("No chats found in %s", json_path)
                return

        for chat in chats:
            yield from _ingest_chat(chat, owner_name, json_path)


def _is_telegram_export(path: Path) -> bool:
    """Quick check that result.json is actually a Telegram export."""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            head = f.read(512)
        return '"chats"' in head or '"messages"' in head
    except OSError:
        return False


def _ingest_chat(chat: dict, owner_name: str, json_path: Path) -> Iterator[NormalizedRecord]:
    """Yield one NormalizedRecord per message in a single chat object."""
    chat_name: str = chat.get("name") or chat.get("title") or "Unknown Chat"
    chat_id: str = str(chat.get("id") or chat_name)
    chat_type: str = chat.get("type", "personal_chat")  # personal_chat | saved_messages | …

    messages: list[dict] = chat.get("messages", [])
    if not messages:
        return

    for msg in messages:
        msg_type = msg.get("type", "")
        # Skip service messages (people joining, calls, etc.) — no conversational content
        if msg_type == "service":
            continue

        raw_text = _text_content(msg.get("text", ""))
        caption = _text_content(msg.get("caption", ""))
        body = raw_text or caption

        # Collect attachment name if present
        att = _attachment_name(msg)
        attachments = [att] if att else []

        # For pure media messages with no caption, use a placeholder body
        # so the record is not empty and still gets indexed.
        if not body and att:
            body = f"[{att}]"

        # Skip completely empty messages
        if not body:
            continue

        # Determine sender
        sender_name: str = msg.get("from") or owner_name
        sender_id: str = str(msg.get("from_id") or sender_name)

        ts = _parse_dt(msg.get("date") or msg.get("date_unixtime"))

        msg_id = str(msg.get("id", ""))

        yield NormalizedRecord(
            platform="telegram",
            record_type="message",
            timestamp=ts,
            sender_id=sender_id,
            sender_name=sender_name,
            thread_id=chat_id,
            thread_name=chat_name,
            body=body,
            attachments=attachments,
            raw={
                "chat_type": chat_type,
                "message_id": msg_id,
                "export_file": str(json_path),
            },
        )
