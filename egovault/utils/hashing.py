"""Deterministic SHA-256 ID generation for records and files."""
from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path


def compute_record_id(
    platform: str,
    thread_id: str,
    timestamp: datetime | str,
    sender_id: str,
    body: str,
    file_path: str | None = None,
) -> str:
    """Return a deterministic hex digest for a record.

    For file records (file_path is not None):
        sha256(platform + file_path + timestamp + body[:128])

    For chat/browser records:
        sha256(platform + thread_id + timestamp + sender_id + body[:128])
    """
    ts = timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp
    if file_path is not None:
        key = "\x00".join([platform, file_path, ts, body[:128]])
    else:
        key = "\x00".join([platform, thread_id, ts, sender_id, body[:128]])
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def compute_file_id(path: str, mtime: float, size: int) -> str:
    """Return a deterministic hex digest for a local file.

    Used by the ingested_files change-detection table.
    Key input: sha256(absolute_path + mtime + size_bytes)
    """
    key = "\x00".join([path, str(mtime), str(size)])
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def compute_content_hash(path: str | Path) -> str:
    """Return a SHA-256 hex digest of the raw file bytes at *path*.

    Reads the file in 64 KiB chunks to handle large files efficiently.
    Used for content-based deduplication so identical files at different
    paths (or with a refreshed mtime) are never scanned twice.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
