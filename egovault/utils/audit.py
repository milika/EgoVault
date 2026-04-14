"""JSONL audit logger for the EgoVault agentic tool-calling loop.

Each tool invocation is appended as a single JSON line to ``data/audit.jsonl``:

    {
        "ts":    "2026-04-14T10:30:00",   # ISO-8601 UTC timestamp
        "tool":  "vault_search",           # tool name
        "args":  {"query": "arduino", …},  # sanitised tool arguments
        "ok":    true,                     # false when tool raised an exception
        "ms":    42,                       # wall-clock duration in milliseconds
        "size":  1234                      # len(result) or -1 on error
    }

Usage::

    record_tool_call(
        tool_name="vault_search",
        args={"query": "..."},
        result="...",
        error=None,
        elapsed_ms=42,
        data_dir=Path("./data"),
    )

The file is rotated: when it exceeds ``_MAX_BYTES`` bytes the oldest half
is silently dropped (in-place truncation, no external dependency needed).
If any I/O error occurs it is silently logged via ``logging`` so it never
interrupts the chat pipeline.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Rotate audit log when it exceeds this size (default 5 MB).
_MAX_BYTES = 5 * 1024 * 1024


def record_tool_call(
    tool_name: str,
    args: dict,
    result: str | None,
    error: BaseException | None,
    elapsed_ms: float,
    data_dir: Path,
) -> None:
    """Append a single tool-call record to ``data/audit.jsonl``.

    Never raises — all I/O errors are swallowed after logging.
    """
    audit_path = data_dir / "audit.jsonl"
    entry: dict = {
        "ts": datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "tool": tool_name,
        "args": _sanitise_args(args),
        "ok": error is None,
        "ms": round(elapsed_ms),
        "size": len(result) if result is not None else -1,
    }
    line = json.dumps(entry, ensure_ascii=False)
    try:
        _rotate_if_needed(audit_path)
        with audit_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception as exc:  # noqa: BLE001
        logger.debug("audit: write failed: %s", exc)


def _sanitise_args(args: dict) -> dict:
    """Return a copy of *args* with sensitive fields redacted."""
    _REDACT = frozenset({"password", "token", "api_key", "secret", "credential"})
    return {
        k: "***" if k.lower() in _REDACT else v
        for k, v in args.items()
    }


def _rotate_if_needed(path: Path) -> None:
    """If *path* exceeds _MAX_BYTES, discard the oldest half of lines."""
    try:
        if not path.exists() or path.stat().st_size <= _MAX_BYTES:
            return
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines(keepends=True)
        keep_from = len(lines) // 2
        path.write_text("".join(lines[keep_from:]), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        logger.debug("audit: rotation failed: %s", exc)
