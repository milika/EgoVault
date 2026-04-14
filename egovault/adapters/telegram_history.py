"""TelegramHistoryAdapter — fetches Telegram message history via Telethon MTProto.

This adapter is NOT registered in the auto-discovery registry.
It is instantiated directly by the telegram-sync CLI command.

User flow
---------
1.  One-time setup (run once)::

        egovault telegram-auth

    Guides you through getting api_id + api_hash from my.telegram.org/apps,
    authenticates with your phone number + Telegram verification code,
    and saves the session locally.

2.  Sync messages::

        egovault telegram-sync

    Fetches all personal chats and group messages since the last sync.
    On subsequent runs only new messages are fetched (incremental).
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator, TYPE_CHECKING

from egovault.core.adapter import BasePlatformAdapter
from egovault.core.schema import NormalizedRecord

if TYPE_CHECKING:
    from egovault.core.store import VaultStore

logger = logging.getLogger(__name__)

# Dialog types to skip by default (broadcast channels = noisy newsletters)
_SKIP_BROADCAST_CHANNELS = True


class TelegramHistoryAdapter(BasePlatformAdapter):
    """Fetches Telegram message history via Telethon MTProto.

    This adapter is **not** registered in the auto-discovery registry.
    """

    platform_id = "telegram"

    def __init__(self, store: "VaultStore") -> None:
        self.store = store

    @classmethod
    def can_handle(cls, source_path: Path) -> bool:
        return False  # never auto-detected

    def ingest(self, source_path: Path) -> "Iterator[NormalizedRecord]":
        raise NotImplementedError("Use ingest_from_api() for Telegram MTProto sync.")

    def ingest_from_api(
        self,
        api_id: int,
        api_hash: str,
        phone: str,
        session_path: Path,
        since: str = "",
        max_messages: int = 5000,
        skip_channels: bool = True,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> Iterator[NormalizedRecord]:
        """Fetch messages from Telegram and yield NormalizedRecords.

        Args:
            api_id: Telegram API ID from my.telegram.org/apps.
            api_hash: Telegram API hash from my.telegram.org/apps.
            phone: Phone number associated with the Telegram account.
            session_path: Path to the Telethon .session file (without extension).
            since: ISO date string (YYYY-MM-DD) — only fetch messages after this date.
            max_messages: Maximum messages to fetch per dialog.
            skip_channels: Skip broadcast channels (default True — avoids newsletter noise).
            progress_callback: Called as (fetched, estimated_total) during sync.
        """
        records = asyncio.run(
            _fetch_all(
                api_id=api_id,
                api_hash=api_hash,
                phone=phone,
                session_path=session_path,
                since=since,
                max_messages=max_messages,
                skip_channels=skip_channels,
                progress_callback=progress_callback,
            )
        )
        yield from records


# ---------------------------------------------------------------------------
# Async fetch logic (runs inside asyncio.run())
# ---------------------------------------------------------------------------

async def _fetch_all(
    api_id: int,
    api_hash: str,
    phone: str,
    session_path: Path,
    since: str,
    max_messages: int,
    skip_channels: bool,
    progress_callback: Callable[[int, int], None] | None,
) -> list[NormalizedRecord]:
    try:
        from telethon import TelegramClient
        from telethon.tl.types import (  # noqa: F401
            Channel, Chat, User,
            MessageMediaDocument, MessageMediaPhoto,
        )
    except ImportError as exc:
        raise ImportError(
            "Telethon is not installed. Run: pip install telethon"
        ) from exc

    since_dt: datetime | None = None
    if since:
        try:
            since_dt = datetime.fromisoformat(since).replace(tzinfo=timezone.utc)
        except ValueError:
            logger.warning("Invalid since date %r — fetching all messages.", since)

    # Telethon session path: pass the string without .session extension
    session_str = str(session_path)

    records: list[NormalizedRecord] = []
    fetched_total = 0

    async with TelegramClient(session_str, api_id, api_hash) as client:
        # Build list of dialogs first to get a total count
        dialogs = []
        async for dialog in client.iter_dialogs():
            entity = dialog.entity
            # Skip broadcast channels if requested
            if skip_channels and isinstance(entity, Channel) and entity.broadcast:
                continue
            dialogs.append(dialog)

        estimated = len(dialogs) * min(max_messages, 100)  # rough estimate

        # Owner info (me)
        me = await client.get_me()
        owner_name = _full_name(me) if me else "me"
        owner_id = str(me.id) if me else "0"

        for dialog in dialogs:
            entity = dialog.entity
            dialog_name = dialog.name or _entity_name(entity) or "Unknown"
            dialog_id = str(dialog.id)

            async for msg in client.iter_messages(
                entity,
                limit=max_messages,
                offset_date=since_dt,
                reverse=False,  # newest first; we'll still get correct timestamps
            ):
                if not hasattr(msg, "message"):
                    continue  # skip service messages

                # Text body
                body = msg.message or ""

                # For media messages pick up the caption; add attachment name
                attachment: str | None = None
                if msg.media:
                    caption = getattr(msg, "message", "") or ""
                    if isinstance(msg.media, MessageMediaPhoto):
                        attachment = "photo.jpg"
                        if not body:
                            body = caption or "[photo]"
                    elif isinstance(msg.media, MessageMediaDocument):
                        doc = msg.media.document
                        # Try to get original filename from attributes
                        fname = _doc_filename(doc)
                        attachment = fname or "file"
                        if not body:
                            body = caption or f"[{attachment}]"

                if not body:
                    continue

                # Sender
                if msg.out:
                    sender_id = owner_id
                    sender_name = owner_name
                else:
                    sender = await msg.get_sender()
                    sender_name = _entity_name(sender) or "unknown"
                    sender_id = str(getattr(sender, "id", sender_name))

                ts = msg.date or datetime.now(tz=timezone.utc)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)

                attachments = [attachment] if attachment else []

                try:
                    record = NormalizedRecord(
                        platform="telegram",
                        record_type="message",
                        timestamp=ts,
                        sender_id=sender_id,
                        sender_name=sender_name,
                        thread_id=dialog_id,
                        thread_name=dialog_name,
                        body=body,
                        attachments=attachments,
                        raw={"message_id": str(msg.id), "dialog_id": dialog_id},
                    )
                    records.append(record)
                except ValueError as exc:
                    logger.debug("Skipping message %s: %s", msg.id, exc)
                    continue

                fetched_total += 1
                if progress_callback and fetched_total % 50 == 0:
                    progress_callback(fetched_total, max(estimated, fetched_total))

    if progress_callback:
        progress_callback(fetched_total, fetched_total)

    return records


# ---------------------------------------------------------------------------
# Async auth helper (called by egovault telegram-auth)
# ---------------------------------------------------------------------------

async def _run_auth(
    api_id: int,
    api_hash: str,
    phone: str,
    session_path: Path,
    code_callback: Callable[[], str],
    password_callback: Callable[[], str],
) -> str:
    """Authenticate and save the session.  Returns the display name of the account."""
    try:
        from telethon import TelegramClient
    except ImportError as exc:
        raise ImportError("Telethon is not installed. Run: pip install telethon") from exc

    session_str = str(session_path)

    async with TelegramClient(session_str, api_id, api_hash) as client:
        await client.start(
            phone=phone,
            code_callback=code_callback,
            password=password_callback,
        )
        me = await client.get_me()
        return _full_name(me) if me else phone


def run_auth(
    api_id: int,
    api_hash: str,
    phone: str,
    session_path: Path,
    code_callback: Callable[[], str],
    password_callback: Callable[[], str],
) -> str:
    """Synchronous wrapper around :func:`_run_auth`."""
    return asyncio.run(
        _run_auth(api_id, api_hash, phone, session_path, code_callback, password_callback)
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _full_name(entity) -> str:  # type: ignore[return]
    first = getattr(entity, "first_name", "") or ""
    last = getattr(entity, "last_name", "") or ""
    return f"{first} {last}".strip() or getattr(entity, "username", "") or "me"


def _entity_name(entity) -> str:
    if entity is None:
        return "unknown"
    # User
    name = _full_name(entity)
    if name:
        return name
    # Chat / Channel
    return getattr(entity, "title", "") or getattr(entity, "username", "") or "unknown"


def _doc_filename(doc) -> str | None:
    """Extract original filename from a Telegram document's attributes."""
    if doc is None:
        return None
    attrs = getattr(doc, "attributes", []) or []
    for attr in attrs:
        fname = getattr(attr, "file_name", None)
        if fname:
            return fname
    return None
