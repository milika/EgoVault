"""Telegram MTProto API helpers — credential storage and session management.

One-time setup (~2 minutes):
  1. Go to https://my.telegram.org/apps  (login with your Telegram phone number)
  2. Click "Create application" — fill any name, short name is fine
  3. Copy the api_id (integer) and api_hash (hex string)
  4. Run: egovault telegram-auth

No server, no OAuth, no browser flow beyond my.telegram.org.
Everything runs over Telethon's MTProto connection directly to Telegram.
"""
from __future__ import annotations

import json
from pathlib import Path

CREDENTIALS_FILENAME = "telegram_api.json"
SESSION_FILENAME = "telegram_session"   # Telethon appends .session automatically


def get_credentials_path(data_dir: str | Path) -> Path:
    return Path(data_dir) / CREDENTIALS_FILENAME


def get_session_path(data_dir: str | Path) -> Path:
    """Return the base session path (without .session extension) for Telethon."""
    return Path(data_dir) / SESSION_FILENAME


def save_credentials(
    data_dir: str | Path,
    api_id: int,
    api_hash: str,
    phone: str,
) -> Path:
    """Persist API credentials to *data_dir/telegram_api.json*."""
    path = get_credentials_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"api_id": api_id, "api_hash": api_hash, "phone": phone}),
        encoding="utf-8",
    )
    try:
        path.chmod(0o600)
    except NotImplementedError:
        pass
    return path


def load_credentials(data_dir: str | Path) -> dict | None:
    """Return ``{"api_id": int, "api_hash": str, "phone": str}`` or None."""
    path = get_credentials_path(data_dir)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {
            "api_id": int(data["api_id"]),
            "api_hash": str(data["api_hash"]),
            "phone": str(data["phone"]),
        }
    except (json.JSONDecodeError, KeyError, ValueError):
        return None
