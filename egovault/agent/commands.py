"""Centralized slash-command handler used by all EgoVault frontends.

All frontends (TUI, Streamlit web, Telegram) call ``handle_command()`` for
every common slash command so that help text, status output, and command
semantics are defined *once* and behave identically everywhere.

Commands that require frontend-specific I/O (``/scan``, ``/gmail-*``,
``/schedule``, ``/open``) return ``None`` so the frontend can handle them
directly.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from egovault.config import Settings

# ---------------------------------------------------------------------------
# Shim imports — kept for backwards-compat while Cycle-7.5 migration runs.
# ruff: noqa: F401
# ---------------------------------------------------------------------------
from egovault.chat.session import (
    _handle_gmail_auth,
    _setup_imap_auth,
    _handle_gmail_sync,
    _explain_imap_error,
    _sync_via_imap,
    _finish_sync,
    _handle_scan,
    _handle_telegram_auth,
    _handle_telegram_sync,
)

# ---------------------------------------------------------------------------
# Help text — single source of truth (Markdown, renders in all frontends)
# ---------------------------------------------------------------------------

HELP_MD = """\
**EgoVault commands**

| Command | Description |
|---------|-------------|
| `/help` | Show this help |
| `/clear` | Clear conversation history |
| `/restart` | Reset conversation history |
| `/sources` | Show sources from last answer |
| `/profile` | Show owner profile |
| `/profile --refresh` | Re-extract owner profile |
| `/status` | LLM server + vault stats + WAN URL |
| `/top N` | Set retrieval depth (1–50, default 10) |
| `/scan <folder>` | Scan a folder and add files to vault |
| `/scan --list` | Show known folder aliases |
| `/open` | Open last saved file with default app |
| `/gmail-auth` | Connect Gmail (one-time setup) |
| `/gmail-sync` | Import emails from Gmail |
| `/exit` | End the session / stop the bot |
| `/schedule --list` | List scheduled tasks |
| `/schedule /gmail-sync every day at 19:05` | Schedule Gmail sync |
| `/schedule /scan inbox every 30min` | Schedule inbox scan |
| `/schedule --cancel <id>` | Cancel a scheduled task |
| `/telegram-auth` | Authenticate with Telegram (one-time setup) |
| `/telegram-sync` | Import Telegram message history into vault |
"""

# ---------------------------------------------------------------------------
# CommandResult
# ---------------------------------------------------------------------------

@dataclass
class CommandResult:
    """Return value from ``handle_command()``.

    *text* is always a Markdown-formatted string.
    *action* signals side-effects the frontend must apply:

    - ``"exit"``           — close/stop the session
    - ``"clear"``          — reset UI (screen + history)
    - ``"restart"``        — reset history only
    - ``"top_n"``          — set top_n to *value*
    - ``"refresh_profile"``— re-extract owner profile (frontend calls LLM)
    """
    text: str
    action: str | None = None
    value: Any = None

# ---------------------------------------------------------------------------
# Status helper — shared by all frontends
# ---------------------------------------------------------------------------

def _fmt_bytes(n: int) -> str:
    if n >= 1_073_741_824:
        return f"{n / 1_073_741_824:.1f} GB"
    if n >= 1_048_576:
        return f"{n / 1_048_576:.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"


def _compute_status(settings: "Settings", ctx: dict) -> str:
    """Return a Markdown string with LLM server + vault stats.

    *ctx* may contain optional keys: ``top_n``, ``bg_threads``, ``scheduler``.
    """
    import urllib.request

    llm = settings.llm
    server_ok = False
    for path in ("/health", "/v1/models"):
        try:
            with urllib.request.urlopen(llm.base_url.rstrip("/") + path, timeout=3) as r:
                if r.status == 200:
                    server_ok = True
                    break
        except Exception:
            pass

    status_icon = "✅ reachable" if server_ok else "❌ unreachable"
    lines: list[str] = [
        f"**Model:** `{llm.model}`",
        f"**Server:** `{llm.base_url}` — {status_icon}",
    ]

    # Vault record count
    try:
        from egovault.core.store import VaultStore
        _vs = VaultStore(settings.vault_db)
        _vs.init_db()
        try:
            stats = _vs.count_records()
            lines.append(f"**Vault records:** {stats.get('total', 0)}")
            for row in stats.get("breakdown", []):
                lines.append(f"- {row['platform']}: {row['count']}")
        finally:
            _vs.close()
    except Exception as exc:
        lines.append(f"_Vault query failed: {exc}_")

    # Loaded models (Ollama /api/ps or llama-server — graceful fallback)
    try:
        import json as _json
        import urllib.request as _ur
        _ps_url = llm.base_url.rstrip("/") + "/api/ps"
        with _ur.urlopen(_ps_url, timeout=3) as _r:  # noqa: S310
            _models = _json.loads(_r.read()).get("models", [])
        if _models:
            lines.append("")
            lines.append("**Loaded models:**")
            for m in _models:
                sz = m.get("size", 0)
                vram = m.get("size_vram", 0)
                pct = int(vram / sz * 100) if sz else 0
                lines.append(f"- {m.get('name', '?')}: {_fmt_bytes(sz)} total, {_fmt_bytes(vram)} VRAM ({pct}%)")
    except Exception:
        pass

    # Optional context extras
    top_n = ctx.get("top_n")
    if top_n is not None:
        lines.append(f"\n**Top-N chunks:** {top_n}")

    bg_threads = ctx.get("bg_threads", [])
    active_bg = [t for t in bg_threads if t.is_alive()]
    if active_bg:
        lines.append("**Background tasks:** " + ", ".join(t.name.replace("bg-", "") for t in active_bg))

    scheduler = ctx.get("scheduler")
    if scheduler is not None:
        tasks = scheduler.list_tasks()
        if tasks:
            lines.append(f"**Scheduled tasks:** {len(tasks)}")

    wan = os.environ.get("EGOVAULT_WAN_URL", "")
    if wan:
        lines.append(f"\n⚡ **WAN:** [{wan}]({wan})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Central dispatcher
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Central dispatcher
# ---------------------------------------------------------------------------

def _run_capturing(fn, *args, **kwargs) -> str:
    """Run a handler that writes to the ``egovault.chat.session`` global console.

    Temporarily swaps the global ``console`` for a StringIO-backed Rich Console
    (no colours, no live widgets), executes ``fn(*args, **kwargs)``, restores the
    original, and returns the captured plain text.

    Intentionally not thread-safe — only one capturing call should run at a
    time (the Telegram executor serialises calls for a single-user bot).
    """
    from io import StringIO
    import re as _re
    from rich.console import Console as _RC
    import egovault.chat.session as _sess

    buf = StringIO()
    cap = _RC(file=buf, no_color=True, width=100, highlight=False, force_terminal=False)
    orig = _sess.console
    _sess.console = cap
    try:
        fn(*args, **kwargs)
    finally:
        _sess.console = orig
    # Strip residual ANSI codes (shouldn't be any with no_color=True, but be safe)
    text = _re.sub(r"\x1b\[[0-9;]*m", "", buf.getvalue()).strip()
    return text or "(no output)"


def handle_command(cmd: str, ctx: dict) -> "CommandResult | None":
    """Dispatch a slash command and return a ``CommandResult``, or ``None``.

    Returns ``None`` for commands that need frontend-specific I/O (``/scan``,
    ``/gmail-*``, ``/schedule``, ``/open``); the frontend handles those itself.

    *ctx* is a plain dict. Recognised keys:

    - ``settings``      — :class:`egovault.config.Settings` instance
    - ``sources``       — ``list[str]`` source attributions from last answer
    - ``owner_profile`` — ``str`` cached owner profile
    - ``top_n``         — ``int`` current retrieval depth
    - ``bg_threads``    — list of background :class:`threading.Thread` objects
    - ``bg_progress``   — ``dict[str, BgProgress]``
    - ``scheduler``     — :class:`egovault.utils.scheduler.Scheduler` instance
    """
    lower = cmd.strip().lower()
    settings: "Settings | None" = ctx.get("settings")

    if lower in ("/exit", "/quit"):
        return CommandResult(text="Goodbye.", action="exit")

    if lower == "/clear":
        return CommandResult(text="", action="clear")

    if lower == "/restart":
        return CommandResult(text="Conversation history cleared.", action="restart")

    if lower == "/help":
        return CommandResult(text=HELP_MD)

    if lower == "/sources":
        sources: list[str] = ctx.get("sources", [])
        text = "\n".join(f"- {s}" for s in sources) if sources else "_No sources from the previous answer._"
        return CommandResult(text=text)

    if lower == "/profile":
        profile: str = ctx.get("owner_profile", "")
        text = profile if profile else "_No profile extracted yet — try `/scan <folder>` first._"
        return CommandResult(text=text)

    if lower == "/profile --refresh":
        return CommandResult(text="", action="refresh_profile")

    if lower == "/status":
        if settings is None:
            return CommandResult(text="_Status unavailable: settings not in context._")
        return CommandResult(text=_compute_status(settings, ctx))

    if lower.startswith("/top "):
        parts = lower.split()
        if len(parts) == 2 and parts[1].isdigit():
            n = max(1, min(50, int(parts[1])))
            return CommandResult(
                text=f"Retrieving up to **{n}** records per query.",
                action="top_n",
                value=n,
            )
        return CommandResult(text="Usage: `/top <number>` (1–50)")

    # Unknown — frontend handles it (/scan, /gmail-*, /schedule, /open …)
    return None

