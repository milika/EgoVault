"""Level-0 (Ego) chat session — interactive REPL over the vault."""
from __future__ import annotations

import json
import os
import queue
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from egovault.utils.scheduler import Scheduler
    from egovault.config import Settings

# prompt_toolkit is a REPL-only dependency — imported lazily so that web.py
# can import _call_llm / _call_llm_agent without requiring it to be installed.
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.rule import Rule

from egovault.chat.rag import RetrievedChunk, assemble_context, build_prompt, extract_owner_profile, retrieve, source_attribution, vault_summary_context
from egovault.config import Settings, load_agent_prompts
from egovault.core.store import VaultStore
from egovault.utils.folders import list_known_folders, resolve_folder

console = Console(highlight=False)

# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------
_DIR_LIST_CAP = 200         # max directory entries to emit in list_directory tool
_EMBED_BATCH_LIMIT = 5_000  # max records per embed-on-ingest background batch
_HYDE_TIMEOUT_CAP = 15      # max seconds for HyDE LLM calls (fast fallback)

# Tools that are irreversible — require a confirmation step before execution.
# Set session_ctx["_confirmed_tool"] = tool_name to bypass for the current turn.
_REQUIRES_CONFIRMATION: frozenset[str] = frozenset({"send_email"})




class LLMKwargs(TypedDict):
    """Keyword arguments passed to ``_call_llm`` / ``_call_llm_agent``."""

    base_url: str
    model: str
    timeout: int | float
    provider: str
    api_key: str


class _SessionCtxOptional(TypedDict, total=False):
    last_file: str  # path to the last file written by a tool


class SessionCtx(_SessionCtxOptional):
    """Per-turn ephemeral context dict passed through the agentic loop."""

    settings: "Settings"
    last_sources: list[str]
    owner_profile: str
    owner_profile_ref: dict  # mutable ref — set dirty=True when scan happens
    call_llm_fn: Callable
    hyde_llm_kwargs: "LLMKwargs"
    scheduler: "Scheduler | None"
    notice_queue: "queue.Queue | None"



# File-classification sets — defined once at module level to avoid re-creation inside hot loops.
_IMAGE_EXTS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".gif", ".webp", ".heic", ".bmp"})
_FILE_EXTS: frozenset[str] = frozenset({"pdf", "doc", "docx", "txt", "jpg", "jpeg",
                                         "png", "gif", "csv", "xlsx", "xls", "zip",
                                         "mp3", "mp4", "mov", "avi", "md"})

def _get_version() -> str:
    try:
        from importlib.metadata import version
        return version("egovault")
    except Exception:
        return "?"


_BANNER = "[bold cyan]EgoVault[/bold cyan] — your personal data vault  |  type [dim]/help[/dim] for commands"

_HELP = (
    "[bold]Commands:[/bold]\n"
    "  [cyan]/exit[/cyan]  or  [cyan]/quit[/cyan]  — end the session\n"
    "  [cyan]/clear[/cyan]                       — clear the screen\n"
    "  [cyan]/restart[/cyan]                     — clear screen and reset conversation history\n"
    "  [cyan]/sources[/cyan]                     — show sources used in last answer\n"
    "  [cyan]/status[/cyan]                      — show LLM server info\n"
    "  [cyan]/top N[/cyan]                       — set how many records to retrieve (default 10)\n"
    "  [cyan]/scan <folder>[/cyan]               — scan a folder once and add files to vault\n"
    "  [cyan]/scan --list[/cyan]                 — show well-known folder aliases (desktop, downloads…)\n"
    "  [cyan]/profile[/cyan]                     — show / re-extract your personal profile from the vault\n"
    "  [cyan]/open[/cyan]                        — open last saved/single-result file with default app\n"
    "  [cyan]/gmail-auth[/cyan]                  — connect Gmail (App Password, one-time setup)\n"
    "  [cyan]/gmail-sync[/cyan]                  — import emails from Gmail (also: /gmail-sync --since 2025-01-01)\n"
    "  [cyan]/schedule[/cyan]                    — manage scheduled tasks\n"
    "    [dim]/schedule /gmail-sync in 5min[/dim]\n"
    "    [dim]/schedule /gmail-sync every day at 19:05[/dim]\n"
    "    [dim]/schedule /scan inbox every 30min[/dim]\n"
    "    [dim]/schedule --list[/dim]              — show all scheduled tasks\n"
    "    [dim]/schedule --cancel <id>[/dim]       — remove a scheduled task\n"
    "  [cyan]/help[/cyan]                        — show this help"
)

# Natural-language phrases that map to slash commands.
# Each entry is (pattern, command_or_callable).
# A plain string is returned as-is; a callable receives the Match and returns the full command.
_INTENT_MAP: list[tuple[re.Pattern[str], str | Callable[[re.Match[str]], str]]] = [
    (re.compile(r"\b(status|how.{0,10}model|gpu|vram|memory usage)\b"), "/status"),
    (re.compile(r"\b(sources?|where.*come from|which.*source|show.*source|list.*source)\b"), "/sources"),
    (re.compile(r"\b(help|what can you do|show commands?|list commands?)\b"), "/help"),
    (re.compile(r"\b(clear|clean.{0,10}screen|reset.{0,10}screen)\b"), "/clear"),
    (re.compile(r"\b(restart|start over|new conversation|reset (chat|history|conversation))\b"), "/restart"),
    (re.compile(r"\b(exit|quit|bye|goodbye|stop|end session)\b"), "/exit"),
    # Gmail intents
    (re.compile(r"\b(connect|auth|link|setup|sign.?in).{0,20}gmail\b"), "/gmail-auth"),
    (re.compile(r"\bgmail.{0,20}(connect|auth|link|setup|sign.?in)\b"), "/gmail-auth"),
    (re.compile(r"\b(sync|import|fetch|download|pull).{0,20}(gmail|emails?|mail)\b"), "/gmail-sync"),
    (re.compile(r"\b(gmail|emails?|mail).{0,20}(sync|import|fetch|download|pull)\b"), "/gmail-sync"),
    # scan-folder intent — captures the folder alias as group 2
    (re.compile(r"\b(scan|index|import|ingest).{0,20}\b(inbox|desktop|documents|downloads|pictures|music|videos|movies|home)\b"),
     lambda m: f"/scan {m.group(2)}"),
    # bare "scan inbox" / "index inbox" without a trailing alias word
    (re.compile(r"\b(scan|index|import|ingest).{0,10}(my\s+)?inbox\b"),
     "/scan inbox"),
    # open-file intent
    (re.compile(r"\bopen\b.{0,30}\b(file|result|it|that|this)\b"), "/open"),
]

# Regex matching any scheduling time expression in natural language.
_SCHEDULE_TIME_RE = re.compile(
    r"\b(?:"
    r"in\s+\d+\s*(?:sec(?:ond)?s?|min(?:ute)?s?|hr?s?|hour?s?|days?)"
    r"|every\s+(?:\d+\s*)?(?:sec(?:ond)?s?|min(?:ute)?s?|hr?s?|hour?s?|days?|morning|evening|night)"
    r"(?:\s+at\s+\d{1,2}:\d{2})?"
    r"|(?:daily|every\s+day)\s+at\s+\d{1,2}:\d{2}"
    r")\b",
    re.IGNORECASE,
)


def _make_prompt_session() -> "PromptSession | None":
    """Create a PromptSession with persistent file history.

    Returns None when stdout is not a real TTY (e.g. piped/redirected),
    in which case the REPL falls back to plain input().
    """
    import sys
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return None
    history_dir = Path.home() / ".config" / "egovault"
    history_dir.mkdir(parents=True, exist_ok=True)
    history_file = history_dir / "chat_history"
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory
        return PromptSession(history=FileHistory(str(history_file)))
    except Exception:
        return None


def _resolve_intent(text: str) -> str | None:
    """Return the full slash command if *text* matches a known intent, else None."""
    lower = text.lower()
    # Check schedule intents first (they are more specific).
    # Pass original text so the chat: scheduler preserves case.
    scheduled = _resolve_schedule_intent(lower, text)
    if scheduled:
        return scheduled
    for pattern, command in _INTENT_MAP:
        m = pattern.search(lower)
        if m:
            return command(m) if callable(command) else command
    return None


def _resolve_schedule_intent(lower: str, original: str = "") -> str | None:
    """Detect if *lower* is a scheduling request and return a /schedule command.

    Examples::

        "check my mail in 5 minutes"  →  "/schedule /gmail-sync in 5min"
        "sync gmail every day at 19:05"  →  "/schedule /gmail-sync every day at 19:05"
        "scan inbox every 30 minutes"  →  "/schedule /scan inbox every 30min"
        "do web search for X in 1min"  →  "/schedule chat: do web search for X in 1min"
    """
    time_m = _SCHEDULE_TIME_RE.search(lower)
    if not time_m:
        return None

    time_expr = time_m.group(0)

    # Detect the target command from surrounding context.
    if re.search(r"\b(gmail|mail|email)\b", lower):
        cmd = "/gmail-sync"
    elif re.search(r"\b(scan|inbox|folder)\b", lower):
        cmd = "/scan inbox"
    else:
        # General scheduling: use the original-case text as the prompt so the
        # scheduled chat agent runs the full request (web search, file write, etc.).
        source = original if original else lower
        prompt = re.sub(r"\s{2,}", " ", _SCHEDULE_TIME_RE.sub("", source).strip().strip(",").strip())
        if len(prompt) > 5:
            return f"/schedule chat: {prompt} {time_expr}"
        return None

    return f"/schedule {cmd} {time_expr}"


def _fmt_bytes(n: int) -> str:
    """Format a byte count as a human-readable string (KB / MB / GB)."""
    if n >= 1_073_741_824:
        return f"{n / 1_073_741_824:.1f} GB"
    if n >= 1_048_576:
        return f"{n / 1_048_576:.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"


def _ollama_ps(base_url: str) -> list[dict]:
    """Query Ollama GET /api/ps and return the list of loaded models.

    Returns an empty list if the endpoint is unavailable or returns no models.
    Each dict has at least 'name', 'size', and 'size_vram' keys (Ollama format).
    """
    import urllib.request as _ur
    import urllib.error as _ue
    url = base_url.rstrip("/") + "/api/ps"
    try:
        with _ur.urlopen(url, timeout=5) as resp:  # noqa: S310
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("models", [])
    except (_ue.URLError, OSError, json.JSONDecodeError):
        return []


def _call_llm(
    base_url: str,
    model: str,
    messages: list[dict],
    timeout: int,
    provider: str = "llama_cpp",
    api_key: str = "",
) -> tuple[str, dict]:
    """Call the configured LLM endpoint. Returns (content, raw_response_data).

    Delegates HTTP dispatch to ``egovault.utils.llm.call_llm_chat``, which is
    the single source of truth for LLM wire protocol.  This wrapper preserves
    the ``(content, data)`` tuple contract expected by the chat session callers.
    """
    from egovault.utils.llm import call_llm_chat as _dispatch

    content = _dispatch(
        base_url=base_url,
        model=model,
        messages=messages,
        timeout=timeout,
        provider=provider,
        api_key=api_key,
    )
    # Return an empty dict as raw data — callers that need it access content only.
    return content, {}




def _open_with_default_app(path: str) -> str:
    """Open *path* with the system-default application. Returns a status message."""
    import subprocess
    import sys as _sys
    try:
        if _sys.platform == "win32":
            os.startfile(path)  # noqa: S606
        elif _sys.platform == "darwin":
            subprocess.run(["open", path], check=False)  # noqa: S603, S607
        else:
            subprocess.run(["xdg-open", path], check=False)  # noqa: S603, S607
        return f"Opened: {path}"
    except Exception as exc:
        return f"Could not open {Path(path).name}: {exc}"


def _resolve_write_target_path(raw_path: str) -> Path:
    """Resolve model-provided output paths to a usable local filesystem path.

    On Windows, LLMs sometimes emit Linux-like paths such as
    ``/home/user/Desktop/file.txt``. Those would otherwise resolve to
    ``<drive>:\\home\\user\\Desktop\\...``. Map these to the real Desktop.
    """
    expanded = os.path.expandvars(os.path.expanduser(raw_path.strip()))

    if os.name == "nt":
        normalized = expanded.replace("\\", "/")
        m = re.match(r"^/(?:home|users)/[^/]+/desktop(?:/(.*))?$", normalized, re.IGNORECASE)
        if m:
            suffix = (m.group(1) or "").lstrip("/")
            try:
                desktop = resolve_folder("desktop")
            except ValueError:
                desktop = (Path.home() / "Desktop").resolve()
            return desktop / Path(suffix) if suffix else desktop

    return Path(expanded)


def _handle_gmail_auth(store: VaultStore, settings: Settings) -> None:
    """Execute an inline /gmail-auth command from the chat REPL.

    Uses IMAP App Password — no Google Cloud project or browser OAuth needed.
    Credentials are saved to ``data/gmail_imap.json`` and reused automatically.
    Re-running is safe: reports if already connected.
    """
    from egovault.utils.gmail_imap import (
        get_credentials_path as get_imap_path,
        load_credentials as load_imap,
    )

    data_dir = Path(settings.vault_db).parent
    imap_path = get_imap_path(data_dir)

    if load_imap(data_dir) is not None:
        console.print(
            f"[green]✓[/green] Already connected — credentials at [cyan]{imap_path}[/cyan]\n"
            "[dim]Run [bold]/gmail-sync[/bold] to import your emails.[/dim]"
        )
        return

    _setup_imap_auth(data_dir)


def _setup_imap_auth(data_dir: Path) -> None:
    """Prompt for IMAP App Password credentials and verify them."""
    from egovault.utils.gmail_imap import verify_connection

    console.print(
        "\n[bold cyan]Gmail Setup — IMAP App Password[/bold cyan]\n"
        "─────────────────────────────────────────────────────────────\n"
        "[bold]Step 1 — Enable IMAP in Gmail[/bold]\n"
        "  1. In [link=https://mail.google.com]Gmail[/link], click [bold]Settings ⚙[/bold] → [bold]See all settings[/bold]\n"
        "  2. Click the [bold]Forwarding and POP/IMAP[/bold] tab\n"
        "  3. In the [bold]IMAP Access[/bold] section, select [bold]Enable IMAP[/bold]\n"
        "  4. Click [bold]Save Changes[/bold]\n\n"
        "[bold]Step 2 — Create an App Password[/bold]  [dim](requires 2-Step Verification)[/dim]\n"
        "  1. Visit [link=https://myaccount.google.com/apppasswords]https://myaccount.google.com/apppasswords[/link]\n"
        "  2. Enter a name (e.g. [italic]EgoVault[/italic]) and click [bold]Create[/bold]\n"
        "  3. Copy the 16-character code shown\n"
    )

    try:
        email_addr = input("Gmail address: ").strip()
        if not email_addr:
            console.print("[dim]Cancelled.[/dim]")
            return
        if "@" not in email_addr:
            email_addr = f"{email_addr}@gmail.com"
        app_pwd = input("App Password (16 chars, spaces ok): ").replace(" ", "").strip()
        if not app_pwd:
            console.print("[dim]Cancelled.[/dim]")
            return
    except (EOFError, KeyboardInterrupt):
        console.print("[dim]Cancelled.[/dim]")
        return

    console.print("[dim]Verifying credentials…[/dim]")
    try:
        verify_connection(email_addr, app_pwd)
    except Exception as exc:  # noqa: BLE001
        console.print(
            f"[red]Connection failed:[/red] {exc}\n"
            "[dim]Check your email address, App Password, and that IMAP is enabled in Gmail settings.[/dim]"
        )
        return

    from egovault.utils.gmail_imap import save_credentials as _save
    creds_path = _save(data_dir, email_addr, app_pwd)
    console.print(
        f"[green]✓[/green] Connected — credentials saved to [cyan]{creds_path}[/cyan]\n"
        "[dim]Run [bold]/gmail-sync[/bold] to import your emails.[/dim]"
    )


def _handle_gmail_sync(user_input: str, store: VaultStore, settings: Settings) -> None:
    """Execute an inline /gmail-sync command from the chat REPL (IMAP only).

    Usage:
        /gmail-sync                      fetch up to 500 recent emails
        /gmail-sync --since 2025-01-01   only emails after a date
        /gmail-sync --max 2000           fetch more emails
    """
    data_dir = Path(settings.vault_db).parent

    import shlex
    try:
        tokens = shlex.split(user_input)[1:]  # drop "/gmail-sync"
    except ValueError:
        tokens = user_input.split()[1:]

    since = ""
    max_results = 500
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in ("--since", "-s") and i + 1 < len(tokens):
            since = tokens[i + 1]
            i += 2
        elif tok in ("--max", "-m", "--max-results") and i + 1 < len(tokens):
            try:
                max_results = int(tokens[i + 1])
            except ValueError:
                pass
            i += 2
        else:
            i += 1

    if not since:
        since = store.get_setting("gmail_last_sync") or ""

    oldest_synced = store.get_setting("gmail_oldest_synced") or ""
    # Bootstrap: if we have a last-sync date but have never tracked the oldest
    # email, start the backward pass from before the last-sync date so existing
    # vaults (created before dual-frontier was added) begin backfilling immediately.
    if not oldest_synced and since:
        oldest_synced = since

    from egovault.utils.gmail_imap import load_credentials as load_imap
    imap_creds = load_imap(data_dir)

    if imap_creds is None:
        console.print("[dim]Not connected to Gmail yet — running setup…[/dim]")
        _handle_gmail_auth(store, settings)
        imap_creds = load_imap(data_dir)
        if imap_creds is None:
            return

    _sync_via_imap(imap_creds, since, oldest_synced, max_results, store)


def _explain_imap_error(exc: Exception) -> None:
    """Print a user-friendly explanation for common IMAP errors."""
    _IMAP_GUIDE = (
        "  1. In [link=https://mail.google.com]Gmail[/link], click [bold]Settings ⚙[/bold] → [bold]See all settings[/bold]\n"
        "  2. Click the [bold]Forwarding and POP/IMAP[/bold] tab\n"
        "  3. In the [bold]IMAP Access[/bold] section, select [bold]Enable IMAP[/bold]\n"
        "  4. Click [bold]Save Changes[/bold]"
    )
    msg = str(exc).lower()
    if "bad" in msg or "examine" in msg or "select" in msg:
        console.print(
            "[red]IMAP error:[/red] Gmail rejected the connection.\n\n"
            "[bold]IMAP access is not enabled on your Gmail account.[/bold]\n"
            "To enable it:\n"
            + _IMAP_GUIDE +
            "\n\nThen run [bold]/gmail-sync[/bold] again."
        )
    elif "authentication" in msg or "invalid credentials" in msg or "login" in msg:
        console.print(
            "[red]IMAP error:[/red] Authentication failed.\n\n"
            "[bold]Your App Password may be incorrect or revoked.[/bold]\n"
            "To fix it:\n"
            "  1. Visit [link=https://myaccount.google.com/apppasswords]https://myaccount.google.com/apppasswords[/link]\n"
            "  2. Delete the old EgoVault password and create a new one\n"
            "  3. Run [bold]/gmail-auth[/bold] to save the updated password\n\n"
            "Also confirm that IMAP is enabled:\n"
            + _IMAP_GUIDE
        )
    elif "connection" in msg or "timeout" in msg or "network" in msg:
        console.print(
            "[red]IMAP error:[/red] Could not connect to Gmail (imap.gmail.com:993).\n\n"
            "Check your internet connection and try again.\n"
            "Also confirm that IMAP is enabled:\n"
            + _IMAP_GUIDE +
            f"\n\n[dim]Detail: {exc}[/dim]"
        )
    else:
        console.print(
            f"[red]IMAP sync error:[/red] {exc}\n\n"
            "If this persists, confirm IMAP is enabled:\n"
            + _IMAP_GUIDE
        )


def _sync_via_imap(
    imap_creds: tuple[str, str],
    since: str,
    before: str,
    max_results: int,
    store: "VaultStore",
) -> None:
    """Perform a Gmail sync using the IMAP App Password path.

    Runs two passes per sync to grow the database in both directions:
    - Forward pass  (SINCE *since*):  picks up new emails.
    - Backward pass (BEFORE *before*): grows history towards the past.
    """
    from egovault.adapters.gmail_imap_adapter import GmailImapAdapter

    gmail_address, app_password = imap_creds
    direction_note = ""
    if since and before:
        direction_note = f" — new + backfill before {before}"
    elif since:
        direction_note = f" — since {since}"
    elif before:
        direction_note = f" — backfill before {before}"
    console.print(
        f"[dim]Connecting to Gmail IMAP as [bold]{gmail_address}[/bold]"
        f"{direction_note}…[/dim]"
    )

    adapter = GmailImapAdapter(store=store)
    inserted = skipped = 0
    oldest_ts = None

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("[cyan]Fetching emails…", total=None)

        def _on_progress(fetched: int, total: int) -> None:
            progress.update(task, completed=fetched, total=total,
                            description=f"[cyan]Fetching emails… [dim]({inserted} new / {skipped} seen)[/dim]")

        try:
            for record in adapter.ingest_from_imap(
                gmail_address=gmail_address,
                app_password=app_password,
                since=since,
                before=before,
                max_results=max_results,
                progress_callback=_on_progress,
            ):
                was_new = store.upsert_record(record)
                if was_new:
                    inserted += 1
                else:
                    skipped += 1
                # Track oldest timestamp from ALL fetched records (new + already
                # in vault) so the backward frontier advances even when the vault
                # is fully up to date in the current window.
                if record.timestamp and (oldest_ts is None or record.timestamp < oldest_ts):
                    oldest_ts = record.timestamp
        except Exception as exc:  # noqa: BLE001
            _explain_imap_error(exc)
            return

    _finish_sync(store, inserted, skipped, oldest_ts)


def _finish_sync(
    store: "VaultStore",
    inserted: int,
    skipped: int,
    oldest_ts=None,
) -> None:
    from datetime import date
    store.set_setting("gmail_last_sync", date.today().strftime("%Y-%m-%d"))
    # Grow backward frontier: push gmail_oldest_synced further into the past.
    if oldest_ts is not None:
        oldest_str = oldest_ts.strftime("%Y-%m-%d")
        current_oldest = store.get_setting("gmail_oldest_synced") or ""
        if not current_oldest or oldest_str < current_oldest:
            store.set_setting("gmail_oldest_synced", oldest_str)
    console.print(
        f"[green]✓[/green] Gmail sync — "
        f"[bold]{inserted}[/bold] new  [dim]({skipped} already in vault)[/dim]"
    )
    if inserted > 0:
        console.print(
            "[dim]New emails are searchable now. "
            "Run [bold]egovault enrich[/bold] to summarise them.[/dim]"
        )


def _handle_telegram_auth(store: "VaultStore", settings: "Settings") -> None:
    """Interactive /telegram-auth wizard for the TUI (terminal only)."""
    from pathlib import Path

    data_dir = Path(settings.vault_db).parent

    try:
        import telethon  # noqa: F401
    except ImportError:
        console.print("[red]Telethon is not installed.[/red]\nRun: pip install telethon")
        return

    from egovault.utils.telegram_api import (
        get_session_path, save_credentials, load_credentials as _load_tg,
    )
    creds = _load_tg(data_dir)
    session_path = get_session_path(data_dir)
    if creds and session_path.with_suffix(".session").exists():
        console.print("[green]✓ Already authenticated.[/green] Run /telegram-sync to sync.")
        return

    console.print(
        "[bold]Telegram Auth Setup[/bold]\n"
        "Get your [bold]api_id[/bold] and [bold]api_hash[/bold] at: "
        "[cyan]https://my.telegram.org/apps[/cyan]\n"
    )
    try:
        while True:
            api_id_raw = input("api_id (numbers only): ").strip()
            try:
                api_id = int(api_id_raw)
                break
            except ValueError:
                console.print("[red]api_id must be a number.[/red]")
        api_hash = input("api_hash: ").strip()
        if not api_hash:
            console.print("[dim]Cancelled.[/dim]")
            return
        phone = input("Phone number (e.g. +385991234567): ").strip()
        if not phone:
            console.print("[dim]Cancelled.[/dim]")
            return
    except (EOFError, KeyboardInterrupt):
        console.print("[dim]Cancelled.[/dim]")
        return

    console.print("[dim]Connecting to Telegram…[/dim]")

    def _ask_code() -> str:
        try:
            return input("Verification code sent to your Telegram app: ").strip()
        except (EOFError, KeyboardInterrupt):
            return ""

    def _ask_password() -> str:
        import getpass
        try:
            return getpass.getpass("Two-step verification password (blank to skip): ")
        except (EOFError, KeyboardInterrupt):
            return ""

    from egovault.adapters.telegram_history import run_auth
    try:
        display_name = run_auth(
            api_id=api_id,
            api_hash=api_hash,
            phone=phone,
            session_path=session_path,
            code_callback=_ask_code,
            password_callback=_ask_password,
        )
    except Exception as exc:
        console.print(f"[red]Authentication failed:[/red] {exc}")
        return

    save_credentials(data_dir, api_id, api_hash, phone)
    console.print(
        f"[green]✓[/green] Authenticated as [bold]{display_name}[/bold]\n"
        "[dim]Run [bold]/telegram-sync[/bold] to import your messages.[/dim]"
    )


def _handle_telegram_sync(user_input: str, store: "VaultStore", settings: "Settings") -> None:
    """Execute /telegram-sync as a foreground command (TUI / session REPL).

    Usage:
        /telegram-sync                       fetch messages since last sync
        /telegram-sync --since 2025-01-01    only messages after a date
        /telegram-sync --max 10000           limit messages fetched
    """
    from pathlib import Path

    data_dir = Path(settings.vault_db).parent

    try:
        from egovault.utils.telegram_api import get_session_path, load_credentials as _load_tg
    except ImportError:
        console.print("[yellow]⚠ Telethon not installed. Run: pip install telethon[/yellow]")
        return

    creds = _load_tg(data_dir)
    if creds is None:
        console.print(
            "[yellow]⚠ Not authenticated \u2014 run [bold]/telegram-auth[/bold] first.[/yellow]"
        )
        return

    session_path = get_session_path(data_dir)
    if not session_path.with_suffix(".session").exists():
        console.print(
            "[yellow]⚠ Session file missing \u2014 run [bold]/telegram-auth[/bold] "
            "to re-authenticate.[/yellow]"
        )
        return

    import shlex
    try:
        tokens = shlex.split(user_input)[1:]
    except ValueError:
        tokens = user_input.split()[1:]

    since = ""
    max_messages = 5_000
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in ("--since", "-s") and i + 1 < len(tokens):
            since = tokens[i + 1]
            i += 2
        elif tok in ("--max", "-m", "--max-messages") and i + 1 < len(tokens):
            try:
                max_messages = int(tokens[i + 1])
            except ValueError:
                pass
            i += 2
        else:
            i += 1

    if not since:
        since = store.get_setting("telegram_last_sync") or ""

    from egovault.adapters.telegram_history import TelegramHistoryAdapter
    console.print("[dim]Connecting to Telegram\u2026[/dim]")
    adapter = TelegramHistoryAdapter(store=store)
    inserted = skipped = 0
    try:
        for record in adapter.ingest_from_api(
            api_id=creds["api_id"],
            api_hash=creds["api_hash"],
            phone=creds["phone"],
            session_path=session_path,
            since=since,
            max_messages=max_messages,
        ):
            was_new = store.upsert_record(record)
            if was_new:
                inserted += 1
            else:
                skipped += 1
    except Exception as exc:
        console.print(f"[red]\u2717 Telegram sync failed:[/red] {exc}")
        return

    from datetime import date
    store.set_setting("telegram_last_sync", date.today().strftime("%Y-%m-%d"))
    console.print(
        f"[green]\u2713[/green] Telegram sync complete \u2014 "
        f"[bold]{inserted}[/bold] new  [dim]({skipped} already in vault)[/dim]"
    )


def _handle_scan(user_input: str, store: VaultStore, settings: Settings) -> None:
    """Execute an inline /scan command from the chat REPL.

    Usage:
        /scan --list                  list well-known aliases
        /scan <folder-alias-or-path>  one-shot folder scan
    """
    from egovault.adapters.local_inbox import LocalInboxAdapter

    # Parse the argument after "/scan"
    parts = user_input.split(None, 1)
    arg = parts[1].strip() if len(parts) > 1 else ""

    if not arg or arg.lower() in ("--list", "-l", "list"):
        console.print("[bold]Well-known folder aliases on this system:[/bold]")
        console.print(f"  [cyan]{'inbox':<12}[/cyan] {Path(settings.inbox_dir).expanduser().resolve()}")
        for alias, path in list_known_folders():
            if path is not None:
                console.print(f"  [cyan]{alias:<12}[/cyan] {path}")
            else:
                console.print(f"  [cyan]{alias:<12}[/cyan] [dim]not found[/dim]")
        if not arg:
            console.print(
                "\n[dim]Usage: /scan <alias-or-path>   e.g.  /scan inbox  or  /scan downloads[/dim]"
            )
        return

    # Resolve the built-in 'inbox' alias to the vault's configured drop-inbox dir.
    if re.search(r'^[/\\]?inbox[/\\]?$', arg.strip(), re.IGNORECASE) or \
            re.search(r'\b(my|the)\s+inbox\b', arg.strip(), re.IGNORECASE):
        arg = str(Path(settings.inbox_dir).expanduser().resolve())

    try:
        src = resolve_folder(arg)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        return

    if not src.exists() or not src.is_dir():
        console.print(f"[red]Error:[/red] Not an existing directory: {src}")
        return

    adapter = LocalInboxAdapter(store=store)
    if not adapter.can_handle(src):
        console.print(
            f"[yellow]No supported files found in[/yellow] [cyan]{src}[/cyan]"
        )
        return

    # Pre-collect files and filter out ones already known by metadata hash.
    # This uses only stat() calls (fast, no file reads) so the progress total
    # matches exactly what the adapter will actually process.
    from egovault.adapters.local_inbox import SUPPORTED_SUFFIXES
    from egovault.utils.hashing import compute_file_id

    all_files = sorted(
        f for f in src.rglob("*")
        if f.is_file() and f.suffix.lower() in SUPPORTED_SUFFIXES
    )
    found_total = len(all_files)

    # Stage-1 pre-filter: skip files whose metadata hash is already in ingested_files
    # UNLESS the stored record has an empty body (needs re-extraction).
    pending: list[Path] = []
    already_known_count = 0
    for f in all_files:
        st = f.stat()
        fid = compute_file_id(str(f), st.st_mtime, st.st_size)
        if store.is_file_known(fid) and not store.record_needs_body_update(str(f)):
            already_known_count += 1
        else:
            pending.append(f)

    total = len(pending)

    if already_known_count:
        console.print(
            f"[dim]{already_known_count} of {found_total} files already in vault — "
            f"scanning {total} new/changed files.[/dim]"
        )

    if total == 0:
        console.print(f"[green]✓[/green] All {found_total} files are already in the vault.")
        return

    inserted = skipped = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("[cyan]ETA[/cyan]"),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as progress:
        # Row 1 — overall progress with ETA
        overall = progress.add_task(
            f"[bold]Scanning [cyan]{src.name}/[/cyan] ({total} files)[/bold]",
            total=total,
        )
        # Row 2 — current file ticker
        current = progress.add_task("[dim]…[/dim]", total=total)

        for record in adapter.ingest(src):
            fname = Path(record.file_path).name if record.file_path else ""
            rel = str(Path(record.file_path).relative_to(src)) if record.file_path else fname
            progress.update(current, description=f"[dim]{rel}[/dim]")
            was_new = store.upsert_record(record)
            if was_new:
                inserted += 1
            else:
                # Re-extracted empty-body record — update the existing row
                if record.file_path and record.body:
                    store.update_body_by_file_path(record.file_path, record.body)
                skipped += 1
            file_id = record.raw.get("file_id")
            if file_id and record.file_path:
                store.upsert_ingested_file(
                    file_id=str(file_id),
                    path=record.file_path,
                    mtime=float(record.raw.get("mtime", 0)),
                    size_bytes=int(record.raw.get("size_bytes", 0)),
                    platform="local",
                    content_hash=record.raw.get("content_hash"),
                )
            progress.advance(overall)
            progress.advance(current)

        # Update the file ticker to show the final summary inline
        progress.update(
            current,
            description=f"[green]{inserted} new[/green]  [dim]{skipped} content-dupes[/dim]",
        )
    console.print(
        f"[green]✓[/green] Scanned [bold]{inserted}[/bold] new records "
        f"([dim]{skipped} already known[/dim]) from [cyan]{src}[/cyan]"
    )
    if inserted > 0:
        console.print("[dim]New files are now searchable — try asking about them.[/dim]")


def _build_file_export(
    args: dict,
    chunks: list,
    settings: "Settings",
) -> str | None:
    """Called from _execute_tool when write_file has empty/stub content but search already ran.

    Inspects the collected chunks to extract real file paths, formats them as the
    requested file type, and returns populated content — or None if not applicable.
    """
    path = args.get("path", "")
    if not path or not chunks:
        return None
    ext = Path(path).suffix.lstrip(".").lower() or "csv"
    unique_paths = list(dict.fromkeys(c.record.file_path for c in chunks if c.record.file_path))
    if not unique_paths:
        return None

    import csv as _csv
    import io as _io
    import json as _json
    if ext == "csv":
        buf = _io.StringIO()
        w = _csv.writer(buf)
        w.writerow(["path"])
        for p in unique_paths:
            w.writerow([p])
        return buf.getvalue()
    if ext == "tsv":
        return "path\n" + "\n".join(unique_paths) + "\n"
    if ext == "json":
        return _json.dumps(unique_paths, indent=2, ensure_ascii=False)
    if ext in ("md", "markdown"):
        rows = ["| Path |", "|------|"] + [f"| {p} |" for p in unique_paths]
        return "\n".join(rows) + "\n"
    # txt / fallback
    return "\n".join(unique_paths) + "\n"


# ---------------------------------------------------------------------------
# Agentic tool-calling loop
# ---------------------------------------------------------------------------

_VAULT_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_vault",
            "description": (
                "Search the personal data vault for emails, messages, files, notes, code projects, "
                "and any other content stored on the user's computer. "
                "Use this for ANY question about: emails from a person, conversations, documents, "
                "files, projects, trips, purchases, or anything the user may have stored. "
                "IMPORTANT: for exhaustive searches ('search all files', 'find every mention') "
                "set max_results to 50. Call multiple times with different keyword variations "
                "to ensure complete coverage. Never conclude a keyword is absent until you have "
                "searched with max_results=50. "
                "When the query has BOTH a topic AND a time constraint (e.g. 'what did Gary "
                "want last week', 'emails from Alice yesterday'), supply BOTH query AND since/until "
                "together — e.g. query='gary', since='2026-04-05', until='2026-04-11'. "
                "When the user asks to VIEW, SHOW, or LIST records for a date range with NO topic "
                "(e.g. 'show yesterday's emails', 'list emails from last week'), pass query='' and "
                "supply since/until/platform filters only. "
                "For relative dates resolve against [TODAY] in the system prompt. "
                "IMAGES/ATTACHMENTS: When the user asks about images, photos, screenshots, or "
                "attachments they sent or received, ALWAYS include the person's name AND media "
                "keywords together — e.g. query='gary,image,photo,jpg,png,screenshot'. "
                "Never search only by person name when the question is about shared media."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Comma-separated topic keywords to search for (not meta-words like "
                            "'search' or 'find'). Known abbreviations (mt, db, py, uk, etc.) are "
                            "automatically expanded to full terms. "
                            "ALWAYS combine person + topic keywords for media queries: "
                            "e.g. 'gary,image,photo,jpg,png' for images sent to/from Gary. "
                            "For best results use full words: "
                            "'malta,maltese'  |  'arduino,sketch'  |  'invoice,payment'. "
                            "Pass empty string when filtering by date/platform only."
                        ),
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of records to return (default 10, max 50). Use 50 for exhaustive searches.",
                    },
                    "file_type": {
                        "type": "string",
                        "description": (
                            "Optional file-extension filter. Restrict results to files of this type. "
                            "Examples: 'pdf', 'txt', 'md', 'py', 'xlsx'. "
                            "Use this whenever the user says 'in pdfs', 'in txt files', 'in spreadsheets', etc."
                        ),
                    },
                    "platform": {
                        "type": "string",
                        "description": "Filter results to a specific platform, e.g. 'gmail', 'local_inbox'. Omit to search all.",
                    },
                    "since": {
                        "type": "string",
                        "description": "ISO date (YYYY-MM-DD) — only return records on or after this date.",
                    },
                    "until": {
                        "type": "string",
                        "description": "ISO date (YYYY-MM-DD) — only return records on or before this date.",
                    },
                    "sort_by": {
                        "type": "string",
                        "description": "Sort order: 'relevance' (default, best match first) or 'date' (newest first). Use 'date' when the user asks for recent or latest items.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "count_records",
            "description": (
                "Count records stored in the vault, optionally filtered by platform and/or date range. "
                "Use this for any counting or statistics question: 'how many emails this week', "
                "'how many files last month', 'how many gmail messages in April'. "
                "For relative dates ('this week', 'yesterday', 'last month'), resolve them using [TODAY] "
                "from the system prompt. 'This week' starts on Monday. Returns total count and per-platform breakdown."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "platform": {
                        "type": "string",
                        "description": "Filter by platform, e.g. 'gmail', 'local_inbox'. Omit to count all platforms.",
                    },
                    "since": {
                        "type": "string",
                        "description": "ISO date (YYYY-MM-DD) lower bound, inclusive. E.g. '2026-04-06' for this week's Monday.",
                    },
                    "until": {
                        "type": "string",
                        "description": "ISO date (YYYY-MM-DD) upper bound, inclusive. E.g. '2026-04-11' for today.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_vault_stats",
            "description": (
                "Return a summary of what is stored in the vault: total record count, "
                "date range, and list of sources. Use this when the user asks what is in "
                "the vault, or before searching to understand available data."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read the full text content of a specific file on disk. "
                "Use this after search_vault finds a relevant file path to get the complete content, "
                "or when the user asks to open / show a specific file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file to read.",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Maximum characters to return (default 4000). Increase for large files.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scan_folder",
            "description": (
                "Scan a folder on disk and add all its files to the personal vault so they "
                "become searchable. Use this when the user asks to index, import, or add a "
                "folder of files, or when you need to access files not yet in the vault. "
                "Use 'inbox' to scan the vault's own drop-inbox folder. "
                "After scanning you can call search_vault to find the newly added content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Absolute path or well-known alias of the folder to scan. "
                            "Aliases: 'inbox' (vault drop-inbox), 'desktop', 'documents', "
                            "'downloads', 'pictures', 'music', 'home'. "
                            "IMPORTANT: when the user says 'inbox', pass the string 'inbox' exactly, not '/inbox'."
                        ),
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_sources",
            "description": (
                "Return the list of vault sources that informed the last answer. "
                "Use this when the user asks where an answer came from, or to cite sources."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_status",
            "description": (
                "Return status of the currently loaded LLM model: name, VRAM usage, GPU layers. "
                "Use this when the user asks about the model, GPU, or memory."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_profile",
            "description": (
                "Return the owner's personal profile extracted from their vault and config files: "
                "name, email, username, location, occupation. Use this when you need to confirm "
                "personal identity details."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": (
                "List files and sub-folders inside a directory on disk. "
                "Use this to explore the user's folder structure, find projects, "
                "or when the user asks what files are in a folder."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the directory to list.",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Optional glob pattern to filter results, e.g. '*.ino' or '*.py'.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_attachment",
            "description": (
                "Download an image or file attachment from a Gmail email and save it locally "
                "so it can be shown in the chat. "
                "Use this when the user asks to see, view, show, or download images/photos/files "
                "that are attached to emails. "
                "First call search_vault to find the emails, then for each attachment call "
                "fetch_attachment with the record_id and the attachment filename. "
                "Returns the local file path on success so the image can be displayed. "
                "Only works for Gmail (platform='gmail') records — local files are already on disk."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "record_id": {
                        "type": "string",
                        "description": "The vault record ID of the email (from search_vault results). Used to look up the Message-ID header.",
                    },
                    "attachment_name": {
                        "type": "string",
                        "description": "The filename of the attachment to download, e.g. 'image.png'. Must match a name in the record's attachments list.",
                    },
                },
                "required": ["record_id", "attachment_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_record",
            "description": (
                "Return the full body of a specific vault record by its ID. "
                "Use this after search_vault finds a relevant email or message and the user wants "
                "to read the full content — 'show me that email', 'what exactly did it say?'. "
                "Works for all record types including Gmail messages which have no on-disk path "
                "and cannot be opened with read_file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "record_id": {
                        "type": "string",
                        "description": "The vault record ID from search_vault results.",
                    },
                },
                "required": ["record_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "gmail_sync",
            "description": (
                "Trigger a live Gmail sync to fetch new emails before searching. "
                "Use this when the user asks 'any new emails?', 'check my mail', 'sync Gmail', "
                "or when search results seem stale. After syncing, call search_vault to find "
                "the newly imported emails. Requires Gmail to be connected via /gmail-auth."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "since": {
                        "type": "string",
                        "description": "ISO date (YYYY-MM-DD) — only fetch emails after this date. Defaults to the last sync date.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum emails to fetch (default 500).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_file",
            "description": (
                "Open a local file with the OS default application, OR download a file from a URL "
                "and save it locally. "
                "Use this after write_file saves a result, when the user says 'open it', "
                "'open the file', 'show me the result', or when the user provides a download "
                "link and asks to download, fetch, or save it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Absolute local file path to open, OR a URL (http:// / https://) to "
                            "download. When a URL is given the file is saved to the vault output "
                            "folder and then opened with the default app."
                        ),
                    },
                    "filename": {
                        "type": "string",
                        "description": (
                            "Optional filename to save the downloaded file as (only used when "
                            "path is a URL). If omitted the name is inferred from the URL."
                        ),
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Write text content to a file on disk. "
                "Use this whenever the user asks to create, save, or export results to a file. "
                "Supports any text format: csv, txt, md, json, html, tsv, yaml, etc. "
                "Always prefer the vault output folder for saving files unless the user specifies a different path. "
                "IMPORTANT: you MUST call search_vault first to collect the actual data, then format it "
                "as the requested file content (e.g. CSV rows), then call write_file with the full populated content. "
                "Never call write_file with placeholder or empty content — only call it once you have real data to write."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Absolute path where the file should be written, including filename and extension. "
                            "Example: 'C:\\Users\\user\\Documents\\results.csv'. "
                            "Use the output_dir from settings when available."
                        ),
                    },
                    "content": {
                        "type": "string",
                        "description": "The full text content to write to the file.",
                    },
                    "overwrite": {
                        "type": "boolean",
                        "description": "If true, overwrite existing file. Defaults to false (appends timestamp suffix to avoid clobbering).",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the public web for current events, news, facts, "
                "software documentation, or anything NOT stored in the personal vault. "
                "Use this when the user asks about: current events, recent news, "
                "prices, weather, public information, how-to guides, or anything where "
                "up-to-date web results are needed. "
                "Do NOT use this for questions about the user's personal data, emails, "
                "files, or messages — use search_vault for those."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query. Be specific and concise.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (1–20, default 5).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "take_screenshot",
            "description": (
                "Capture a screenshot of the current PC screen and save it as a PNG image. "
                "Use this when the user asks to take a screenshot, capture the screen, "
                "show what's on screen, or record the current display. "
                "Returns the saved file path so the image can be displayed in chat."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Optional filename (without extension). Defaults to 'screenshot_<timestamp>'. Example: 'my_screen'.",
                    },
                    "region": {
                        "type": "string",
                        "description": (
                            "Optional screen region to capture as 'x,y,width,height' in pixels. "
                            "Omit to capture the full screen. Example: '0,0,1920,1080'."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_windows",
            "description": (
                "List currently open windows on the PC and optionally capture a screenshot of each. "
                "ALWAYS call this tool fresh — never reuse or echo a previous response from history. "
                "Use `filter` when the user names a specific app: "
                "  'show me task manager' → filter='task manager' "
                "  'show me chrome' → filter='chrome' "
                "Omit `filter` only when the user wants ALL open windows. "
                "Set include_screenshots=false when the user only wants a list of open apps (faster). "
                "Set include_screenshots=true (default) when the user asks to 'see', 'show', or 'capture'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filter": {
                        "type": "string",
                        "description": (
                            "Optional case-insensitive substring filter. "
                            "Only windows whose title contains this string are included. "
                            "Set this to the specific app name when the user asks about one app. "
                            "Omit to list all visible windows."
                        ),
                    },
                    "include_screenshots": {
                        "type": "boolean",
                        "description": (
                            "Capture a screenshot region for each window (default true). "
                            "Set false when the user only wants a list of window titles — no images needed."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_window",
            "description": (
                "Inspect a specific window in detail: dumps all its UI elements (buttons, "
                "menus, text fields, class names, sizes) as a structured text tree, "
                "saves it to a file, and optionally captures a screenshot of that window. "
                "Use when the user wants to 'inspect', 'analyse', or 'get the elements of' a specific app. "
                "Examples: "
                "  'inspect task manager elements' → title_filter='task manager' "
                "  'get the ui tree of notepad' → title_filter='notepad' "
                "  'inspect chrome and save to file' → title_filter='chrome', include_screenshot=true"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title_filter": {
                        "type": "string",
                        "description": "Case-insensitive substring to identify the target window by title. Required.",
                    },
                    "include_screenshot": {
                        "type": "boolean",
                        "description": "Also capture a screenshot of the window (default true).",
                    },
                    "output_file": {
                        "type": "string",
                        "description": (
                            "Optional file path to save the element tree as a .txt file. "
                            "If omitted, the tree is saved to the output directory automatically."
                        ),
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "How deep to walk the element tree (default 5). Reduce to 2-3 for faster results on complex apps.",
                    },
                },
                "required": ["title_filter"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_task",
            "description": (
                "Schedule a task to run at a FUTURE time or on a recurring interval. "
                "ONLY call this tool when the user's message contains an explicit time expression "
                "such as: 'in 5min', 'in 1 hour', 'in 30 seconds', 'every hour', "
                "'every day at 08:00', 'every morning', 'daily at 19:05'. "
                "Do NOT call this tool for immediate requests — if there is no time expression "
                "in the user's message, use the appropriate tool directly "
                "(e.g. web_search, write_file) instead. "
                "Never invent a time expression that the user did not provide. "
                "The `when` parameter MUST be copied verbatim from the user's message. "
                "Examples of requests that REQUIRE schedule_task: "
                "'create a file in 1min with the result of …' → schedule_task(prompt='…', when='in 1min'); "
                "'remind me every morning' → schedule_task(prompt='…', when='every morning'); "
                "'sync gmail every hour' → schedule_task(prompt='/gmail-sync', when='every hour'). "
                "Examples that do NOT use schedule_task (no time expression): "
                "'search web for X and save to desktop' → web_search then write_file; "
                "'take a screenshot' → take_screenshot."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": (
                            "The full action to perform when the task fires. "
                            "This can be any instruction: 'do a web search for X and save to desktop', "
                            "'/gmail-sync', '/scan inbox', etc. "
                            "Preserve the user's original intent exactly."
                        ),
                    },
                    "when": {
                        "type": "string",
                        "description": (
                            "The time expression copied verbatim from the user's message. "
                            "Must be one of: one-shot: 'in 5min', 'in 1 hour', 'in 30 seconds', 'in 2 days'; "
                            "recurring: 'every 30min', 'every hour', 'every day', "
                            "'every day at 19:05', 'every morning', 'every evening'. "
                            "NEVER invent this value — only use what the user explicitly said."
                        ),
                    },
                },
                "required": ["prompt", "when"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": (
                "Send an email via Gmail SMTP using the stored App Password credentials. "
                "ALWAYS call this tool when the user wants to send, compose, write, or email someone. "
                "If to/subject/body are all provided, call immediately without asking. "
                "Only ask for missing parameters if the user did not supply them."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "Recipient email address (or comma-separated list for multiple recipients).",
                    },
                    "subject": {
                        "type": "string",
                        "description": "Email subject line.",
                    },
                    "body": {
                        "type": "string",
                        "description": "Plain-text email body.",
                    },
                    "cc": {
                        "type": "string",
                        "description": "Optional CC recipients (comma-separated email addresses).",
                    },
                    "bcc": {
                        "type": "string",
                        "description": "Optional BCC recipients (comma-separated email addresses).",
                    },
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "system_info",
            "description": (
                "Return a comprehensive snapshot of the PC's system state — CPU usage, RAM, GPU VRAM, "
                "disk drive space, top processes, and network stats. "
                "Use this when the user asks about PC performance, disk space, CPU/GPU usage, "
                "how much RAM is free, what processes are running, or general system health. "
                "Examples: 'how much disk space do I have?', 'what\\'s my CPU usage?', "
                "'show me system info', 'how much RAM is free?', 'what\\'s my GPU usage?', "
                "'how is the system doing?', 'what\\'s running on my PC?', 'PC specs'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sections": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Optional subset of sections to include. "
                            "Valid values: 'os', 'cpu', 'memory', 'gpu', 'disk', 'processes', 'network'. "
                            "Omit or pass ['all'] to return everything."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "launch_frontend",
            "description": (
                "Launch another EgoVault frontend as a detached background process. "
                "ALWAYS call this tool when the user asks to start, open, launch, or run "
                "the web interface, web UI, browser UI, Telegram bot, MCP server, or a new terminal chat. "
                "NEVER describe launching or pretend the interface started — "
                "the frontend is NOT running until this tool returns FRONTEND_LAUNCHED:. "
                "Call immediately with the correct frontend value. "
                "Examples: "
                "  'start the web interface' → frontend='web'; "
                "  'open the web UI' → frontend='web'; "
                "  'open web on port 8888' → frontend='web', port=8888; "
                "  'launch telegram bot' → frontend='telegram'; "
                "  'start MCP server' → frontend='mcp'; "
                "  'open a new terminal chat' → frontend='chat'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "frontend": {
                        "type": "string",
                        "enum": ["web", "telegram", "mcp", "chat"],
                        "description": (
                            "Which interface to start: "
                            "'web' — Streamlit browser UI; "
                            "'telegram' — Telegram bot; "
                            "'mcp' — MCP stdio server; "
                            "'chat' — TUI (new terminal window)."
                        ),
                    },
                    "port": {
                        "type": "integer",
                        "description": "Port for the web frontend (default 8501). Ignored for all other frontends.",
                    },
                },
                "required": ["frontend"],
            },
        },
    },
]


def _expand_abbreviation(term: str) -> str | None:
    """Expand a short abbreviation to full search terms, or return None if unknown.

    Country/territory codes (ISO 3166-1 alpha-2 and alpha-3) are resolved via
    ``pycountry``. Tech abbreviations are handled by a small built-in dict.
    """
    t = term.strip().lower()

    # --- Tech abbreviations (no external DB covers these well) ---
    _TECH: dict[str, str] = {
        # Common informal aliases not in ISO 3166
        "uk": "United Kingdom",
        "us": "United States",
        # Tech abbreviations
        "db": "database",
        "py": "python",
        "js": "javascript",
        "ts": "typescript",
        "cs": "csharp",
        "rb": "ruby",
        "sh": "bash,shell",
        "sql": "sql,database,query",
        "api": "api,endpoint",
        "ui": "interface,frontend",
        "ux": "ux,design",
        "ml": "machine learning",
        "ai": "artificial intelligence",
        "cv": "curriculum vitae,resume",
        "ci": "continuous integration",
        "cd": "continuous deployment",
        "os": "operating system",
        "vm": "virtual machine",
        "ip": "ip address,network",
        "http": "http,https,web,request",
        "pdf": "pdf,document",
        "csv": "csv,spreadsheet",
        "xml": "xml,markup",
        "json": "json,data",
        "yml": "yaml,config",
        "cfg": "config,configuration",
        "env": "environment,config",
        "log": "log,logs",
        "err": "error,exception",
        "tmp": "temp,temporary",
        "src": "source,src",
        "lib": "library,lib",
        "pkg": "package",
        "app": "application,app",
        "exe": "executable,exe",
        "dll": "library,dll",
    }
    if t in _TECH:
        return _TECH[t]

    # --- Country / territory codes via pycountry (ISO 3166) ---
    try:
        import pycountry
        # Try alpha-2 (2-letter) first, then alpha-3 (3-letter)
        country = None
        if len(t) == 2:
            country = pycountry.countries.get(alpha_2=t.upper())
        elif len(t) == 3:
            country = pycountry.countries.get(alpha_3=t.upper())
            if country is None:
                # Could be a subdivision code
                country = pycountry.countries.get(alpha_2=t.upper())
        if country is not None:
            # Return both the English name and common name if different
            names = [country.name]
            if hasattr(country, "common_name") and country.common_name != country.name:
                names.append(country.common_name)
            return ",".join(names)
    except ImportError:
        pass  # pycountry not installed — fall through

    return None


def _rank_search_results(
    lines: list[str],
    query: str,
    llm_kwargs: dict,
    owner_profile: str = "",
) -> list[str]:
    """Ask the LLM to sort *lines* by relevance/importance for *query*.

    Returns the sorted list, or the original order if ranking fails.
    Only called when there are > 2 results (trivial cases need no sorting).
    """
    if len(lines) <= 2:
        return lines

    numbered = "\n".join(f"{i + 1}. {ln}" for i, ln in enumerate(lines))
    profile_hint = f"\nUser context: {owner_profile[:300]}" if owner_profile else ""
    prompt = (
        f"Below are {len(lines)} search results for the query: '{query}'.{profile_hint}\n\n"
        f"Sort these results from most important / most directly relevant to the user "
        f"down to least important. Return ONLY the sorted numbered list in the same "
        f"format (e.g. '1. <text>'). Do not add explanations, headers, or extra text.\n\n"
        f"{numbered}"
    )
    try:
        content, _ = _call_llm(
            base_url=llm_kwargs["base_url"],
            model=llm_kwargs["model"],
            messages=[{"role": "user", "content": prompt}],
            timeout=llm_kwargs["timeout"],
            provider=llm_kwargs.get("provider", "llama_cpp"),
            api_key=llm_kwargs.get("api_key", ""),
        )
        # Parse numbered list from LLM response
        ranked: list[str] = []
        for ln in content.splitlines():
            ln = ln.strip()
            m = re.match(r"^\d+\.\s+(.+)$", ln)
            if m:
                ranked.append(m.group(1))
        if len(ranked) >= len(lines) // 2:
            return ranked
    except Exception:
        pass  # network / parse error — fall back to original order
    return lines


def _extract_matching_lines(context: str, keywords: list[str], max_lines: int = 60) -> list[str]:
    """Extract lines from context matching keywords, using record-aware filtering.

    Single keyword: return all lines containing that keyword (original behavior).
    Multiple keywords: group lines by ``[N]`` record headers, then only extract lines
    from records whose full text contains the FIRST keyword (primary topic).
    This prevents secondary/filter keywords (e.g. "pdf" as a file-type hint) from
    pulling in unrelated records that happen to contain that word.
    Falls back to any-keyword across all records if no record matches the primary keyword.
    """
    kw_lower = [k.lower() for k in keywords]
    if not kw_lower:
        return []

    all_lines = context.splitlines()

    # ── single-keyword: simple scan (original behaviour) ─────────────────────
    if len(kw_lower) == 1:
        seen: set[str] = set()
        results: list[str] = []
        for line in all_lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("[VAULT CONTEXT]") or stripped.startswith("[…"):
                continue
            if kw_lower[0] in stripped.lower():
                key = " ".join(stripped.split())
                if key not in seen:
                    seen.add(key)
                    results.append(stripped)
                    if len(results) >= max_lines:
                        break
        return results

    # ── multi-keyword: record-aware filtering ─────────────────────────────────
    # Group context lines into per-record buckets using [N] header boundaries.
    primary_kw = kw_lower[0]
    records: list[list[str]] = []
    current: list[str] = []
    for line in all_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("[VAULT CONTEXT]"):
            continue
        if re.match(r'^\[\d+\]', stripped):        # start of a new record
            if current:
                records.append(current)
            current = [stripped]
        elif current:
            current.append(stripped)
    if current:
        records.append(current)

    # Only emit lines from records whose BODY (non-header) contains the primary
    # keyword. Checking against header too causes filenames like "Malta.pdf" or
    # "Milika Delic - Curriculum Vitae.pdf" to match a secondary "pdf" keyword,
    # causing the header to appear as a false mention.
    def _record_body_text(rec: list[str]) -> str:
        start = 1 if rec and re.match(r'^\[\d+\]', rec[0]) else 0
        return " ".join(rec[start:]).lower()

    primary_records = [r for r in records if primary_kw in _record_body_text(r)]
    use_records = primary_records if primary_records else records  # graceful fallback

    seen2: set[str] = set()
    results2: list[str] = []   # includes [N] headers as source context
    for record_lines in use_records:
        is_hdr = bool(record_lines and re.match(r'^\[\d+\]', record_lines[0]))
        header_line = record_lines[0] if is_hdr else None
        body_lines = record_lines[1:] if is_hdr else record_lines

        # Match only body lines — never the filename/source header line
        new_matches: list[str] = []
        for stripped in body_lines:
            if stripped.startswith("[…"):
                continue
            if any(kw in stripped.lower() for kw in kw_lower):
                key = " ".join(stripped.split())
                if key not in seen2:
                    seen2.add(key)
                    new_matches.append(stripped)

        if new_matches:
            if header_line:            # prepend header for source attribution
                results2.append(header_line)
            results2.extend(new_matches)
            if len(results2) >= max_lines:
                return results2
    return results2


def _list_windows_cross_platform(filter_str: str = "") -> list[dict]:
    """Return a list of visible windows with title, process name, and bounding box.

    Supports Windows (ctypes / pywin32 / pygetwindow), macOS (AppKit / osascript),
    and Linux (wmctrl).  Each entry is a dict with keys:
    ``title``, ``process``, ``left``, ``top``, ``width``, ``height``.
    """
    import platform as _platform

    _os = _platform.system()

    # ── Windows ──────────────────────────────────────────────────────────────
    if _os == "Windows":
        results: list[dict] = []
        try:
            import ctypes
            import ctypes.wintypes as wt

            # We build the list via EnumWindows + ctypes (zero external deps)
            user32 = ctypes.windll.user32  # type: ignore[attr-defined]
            psapi = None
            try:
                import ctypes
                psapi = ctypes.windll.psapi  # type: ignore[attr-defined]
            except Exception:
                pass

            EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, wt.HWND, ctypes.c_long)

            def _get_process_name(hwnd: int) -> str:
                try:
                    pid = wt.DWORD()
                    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                    proc_handle = ctypes.windll.kernel32.OpenProcess(  # type: ignore[attr-defined]
                        0x0410, False, pid.value
                    )
                    if not proc_handle:
                        return ""
                    buf = ctypes.create_unicode_buffer(260)
                    ctypes.windll.psapi.GetModuleFileNameExW(  # type: ignore[attr-defined]
                        proc_handle, None, buf, 260
                    )
                    ctypes.windll.kernel32.CloseHandle(proc_handle)  # type: ignore[attr-defined]
                    return Path(buf.value).name if buf.value else ""
                except Exception:
                    return ""

            def _callback(hwnd: int, _: int) -> bool:
                # Only visible, non-iconic (not minimised) windows with a title
                if not user32.IsWindowVisible(hwnd):
                    return True
                if user32.IsIconic(hwnd):  # minimised — skip
                    return True
                length = user32.GetWindowTextLengthW(hwnd)
                if length == 0:
                    return True
                buf = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buf, length + 1)
                title = buf.value.strip()
                if not title:
                    return True
                if filter_str and filter_str.lower() not in title.lower():
                    return True
                rect = wt.RECT()
                user32.GetWindowRect(hwnd, ctypes.byref(rect))
                w = rect.right - rect.left
                h = rect.bottom - rect.top
                if w <= 0 or h <= 0:
                    return True
                results.append({
                    "title": title,
                    "process": _get_process_name(hwnd),
                    "left": rect.left,
                    "top": rect.top,
                    "width": w,
                    "height": h,
                })
                return True

            user32.EnumWindows(EnumWindowsProc(_callback), 0)
            return results
        except Exception:
            pass

        # Fallback: pygetwindow
        try:
            import pygetwindow as gw  # type: ignore[import-untyped]
            wins = []
            for w in gw.getAllWindows():
                if not w.title or not w.title.strip():
                    continue
                if filter_str and filter_str.lower() not in w.title.lower():
                    continue
                if w.width <= 0 or w.height <= 0:
                    continue
                wins.append({
                    "title": w.title.strip(),
                    "process": "",
                    "left": w.left,
                    "top": w.top,
                    "width": w.width,
                    "height": w.height,
                })
            return wins
        except Exception:
            return []

    # ── macOS ─────────────────────────────────────────────────────────────────
    if _os == "Darwin":
        try:
            from AppKit import NSWorkspace  # type: ignore[import-untyped]
            import Quartz  # type: ignore[import-untyped]

            running = {
                app.processIdentifier(): app.localizedName()
                for app in NSWorkspace.sharedWorkspace().runningApplications()
                if app.isActive() or True
            }
            cg_windows = Quartz.CGWindowListCopyWindowInfo(
                Quartz.kCGWindowListOptionOnScreenOnly
                | Quartz.kCGWindowListExcludeDesktopElements,
                Quartz.kCGNullWindowID,
            )
            wins = []
            for info in cg_windows:
                title = (
                    info.get("kCGWindowName", "")
                    or info.get("kCGWindowOwnerName", "")
                    or ""
                ).strip()
                if not title:
                    continue
                if filter_str and filter_str.lower() not in title.lower():
                    continue
                bounds = info.get("kCGWindowBounds", {})
                w = int(bounds.get("Width", 0))
                h = int(bounds.get("Height", 0))
                if w <= 0 or h <= 0:
                    continue
                pid = info.get("kCGWindowOwnerPID", 0)
                process = running.get(pid, "")
                wins.append({
                    "title": title,
                    "process": process,
                    "left": int(bounds.get("X", 0)),
                    "top": int(bounds.get("Y", 0)),
                    "width": w,
                    "height": h,
                })
            return wins
        except Exception:
            pass

        # Fallback: pygetwindow
        try:
            import pygetwindow as gw  # type: ignore[import-untyped]
            wins = []
            for w in gw.getAllWindows():
                if not w.title or not w.title.strip():
                    continue
                if filter_str and filter_str.lower() not in w.title.lower():
                    continue
                if w.width <= 0 or w.height <= 0:
                    continue
                wins.append({
                    "title": w.title.strip(),
                    "process": "",
                    "left": w.left,
                    "top": w.top,
                    "width": w.width,
                    "height": w.height,
                })
            return wins
        except Exception:
            return []

    # ── Linux ─────────────────────────────────────────────────────────────────
    import subprocess as _sub
    try:
        out = _sub.check_output(["wmctrl", "-l", "-G"], text=True, timeout=5)
        wins = []
        for line in out.splitlines():
            parts = line.split(None, 8)
            if len(parts) < 9:
                continue
            title = parts[8].strip()
            if not title or title in ("N/A", "(none)"):
                continue
            if filter_str and filter_str.lower() not in title.lower():
                continue
            try:
                x, y, w, h = int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
            except ValueError:
                continue
            if w <= 0 or h <= 0:
                continue
            wins.append({"title": title, "process": "", "left": x, "top": y, "width": w, "height": h})
        return wins
    except Exception:
        return []


def _enumerate_window_elements(hwnd: int, max_depth: int = 5) -> list[str]:
    """Walk the child-window tree of *hwnd* and return an indented text list.

    Uses GetWindow(GW_CHILD/GW_HWNDNEXT) for direct-children traversal, which
    avoids the full-descendants flood of EnumChildWindows.  Works on classic
    Win32 apps; modern XAML/WinUI apps show hosting container nodes only.
    """
    import ctypes as _ct
    import ctypes.wintypes as _wt

    _u32 = _ct.windll.user32  # type: ignore[attr-defined]
    GW_CHILD = 5
    GW_HWNDNEXT = 2

    def _direct_children(parent: int) -> list[int]:
        children: list[int] = []
        child = _u32.GetWindow(parent, GW_CHILD)
        while child:
            children.append(child)
            child = _u32.GetWindow(child, GW_HWNDNEXT)
        return children

    def _walk(h: int, depth: int, lines: list[str]) -> None:
        if depth > max_depth:
            return
        buf = _ct.create_unicode_buffer(512)
        cls = _ct.create_unicode_buffer(256)
        rect = _wt.RECT()
        _u32.GetWindowTextW(h, buf, 512)
        _u32.GetClassNameW(h, cls, 256)
        _u32.GetWindowRect(h, _ct.byref(rect))
        w = rect.right - rect.left
        hh = rect.bottom - rect.top
        # Skip invisible zero-size elements that add noise
        if w == 0 and hh == 0 and not buf.value.strip():
            return
        indent = "  " * depth
        text = buf.value.strip()
        cls_n = cls.value.strip()
        line = f"{indent}[{cls_n}]"
        if text:
            line += f' "{text[:80]}"'
        line += f" {w}x{hh}"
        lines.append(line)
        for child in _direct_children(h):
            _walk(child, depth + 1, lines)

    lines: list[str] = []
    _walk(hwnd, 0, lines)
    return lines


def _execute_tool(
    name: str,
    args: dict,
    store: "VaultStore",
    top_n: int,
    collected_chunks: list,
    session_ctx: dict | None = None,
) -> str:
    """Execute a tool call and return result as a string for the LLM."""
    import time as _time
    _t0 = _time.monotonic()
    _result: str | None = None
    _error: BaseException | None = None
    try:
        _result = _execute_tool_impl(name, args, store, top_n, collected_chunks, session_ctx)
        return _result
    except Exception as exc:
        _error = exc
        raise
    finally:
        _elapsed_ms = (_time.monotonic() - _t0) * 1000
        try:
            from egovault.utils.audit import record_tool_call as _rtc  # noqa: PLC0415
            _data_dir = None
            if session_ctx:
                _s = session_ctx.get("settings")
                if _s:
                    _data_dir = __import__("pathlib").Path(_s.vault_db).parent
            if _data_dir is not None:
                _rtc(name, args, _result, _error, _elapsed_ms, _data_dir)
        except Exception:  # noqa: BLE001
            pass


def _execute_tool_impl(
    name: str,
    args: dict,
    store: "VaultStore",
    top_n: int,
    collected_chunks: list,
    session_ctx: dict | None = None,
) -> str:
    """Execute a tool call and return result as a string for the LLM."""
    _DISPATCH = {
        "search_vault": _tool_search_vault,
        "count_records": _tool_count_records,
        "get_vault_stats": _tool_get_vault_stats,
        "scan_folder": _tool_scan_folder,
        "get_sources": _tool_get_sources,
        "get_status": _tool_get_status,
        "get_profile": _tool_get_profile,
        "read_file": _tool_read_file,
        "list_directory": _tool_list_directory,
        "fetch_attachment": _tool_fetch_attachment,
        "write_file": _tool_write_file,
        "get_record": _tool_get_record,
        "gmail_sync": _tool_gmail_sync,
        "open_file": _tool_open_file,
        "take_screenshot": _tool_take_screenshot,
        "inspect_windows": _tool_inspect_windows,
        "inspect_window": _tool_inspect_window,
        "system_info": _tool_system_info,
        "web_search": _tool_web_search,
        "schedule_task": _tool_schedule_task,
        "send_email": _tool_send_email,
        "launch_frontend": _tool_launch_frontend,
    }
    handler = _DISPATCH.get(name)
    if handler is None:
        return f"Unknown tool: {name}"
    return handler(args, store, top_n, collected_chunks, session_ctx)


def _tool_search_vault(
    args: dict,
    store: "VaultStore",
    top_n: int,
    collected_chunks: list,
    session_ctx: "dict | None",
) -> str:
    query = (args.get("query") or "").strip()
    max_results = min(50, int(args.get("max_results") or top_n))
    platform_filter = (args.get("platform") or "").strip() or None
    since_filter = (args.get("since") or "").strip() or None
    until_filter = (args.get("until") or "").strip() or None

    # ── Date-range / platform browse (no keyword query) ──────────────────
    # When the LLM asks to *show* or *list* records for a time period it
    # passes an empty query + date/platform filters.  Fetch directly from
    # the DB rather than running an empty FTS search.
    if not query:
        if not (platform_filter or since_filter or until_filter):
            return "Error: query parameter is required (or supply since/until/platform to browse by date)."
        rows = store.list_records(
            platform=platform_filter,
            since=since_filter,
            until=until_filter,
            limit=max_results,
        )
        if not rows:
            filter_parts = []
            if platform_filter:
                filter_parts.append(f"platform={platform_filter}")
            if since_filter:
                filter_parts.append(f"since={since_filter}")
            if until_filter:
                filter_parts.append(f"until={until_filter}")
            return f"No records found ({', '.join(filter_parts)})."
        from egovault.core.store import row_to_record as _r2r
        browse_chunks = [
            RetrievedChunk(record=_r2r(row), rank=0.0)
            for row in rows
        ]
        collected_chunks.extend(browse_chunks)
        lines_out: list[str] = []
        for i, c in enumerate(browse_chunks, 1):
            r = c.record
            ts = r.timestamp.strftime("%Y-%m-%d %H:%M") if r.timestamp else ""
            subj = r.thread_name or r.file_path or r.platform
            sender = f" — {r.sender_name}" if r.sender_name else ""
            preview = (r.body or "")[:100].replace("\n", " ").strip()
            preview_str = f": {preview}…" if preview else ""
            lines_out.append(f"  {i}. [{ts}] {subj}{sender}{preview_str}")
        filter_desc = ""
        if since_filter or until_filter:
            filter_desc = f" ({since_filter or ''}–{until_filter or ''})"
        if platform_filter:
            filter_desc = f" [{platform_filter}]{filter_desc}"
        return (
            f"SEARCH RESULTS{filter_desc} — {len(browse_chunks)} record(s). "
            f"Present these to the user as a numbered list:\n\n"
            + "\n".join(lines_out)
        )

    # ── Keyword search (with optional date/platform post-filter) ─────────
    # Normalise: LLMs sometimes use spaces instead of commas as separators.
    # Split by comma when commas are present, else fall back to whitespace.
    if "," in query:
        raw_terms = [t.strip() for t in query.split(",") if t.strip()]
    else:
        raw_terms = [t.strip() for t in query.split() if t.strip()]
    expanded_terms: list[str] = []
    notes: list[str] = []
    for t in raw_terms:
        if len(t) >= 4:
            # If the term looks like a filename or hyphenated compound
            # (contains -, . or _), split it into safe sub-tokens.  This
            # prevents FTS5 from treating `-word` as a NOT filter and from
            # getting confused by dots (e.g. "Yoris-EmiratesTicket.pdf" →
            # ["Yoris", "EmiratesTicket"]).  Drop the extension token if
            # it's a known file suffix (len <= 4) to avoid noise.
            if any(c in t for c in "-._"):
                sub_tokens = [s for s in re.split(r"[-._]", t)
                              if len(s) >= 4 and s.lower() not in _FILE_EXTS]
                if sub_tokens:
                    expanded_terms.extend(sub_tokens)
                else:
                    expanded_terms.append(t)  # fallback: keep as-is
            else:
                expanded_terms.append(t)
        else:
            expansion = _expand_abbreviation(t)
            if expansion:
                expanded_terms.extend(e.strip() for e in expansion.split(","))
                notes.append(f"'{t}' → {expansion}")
            # else: silently drop — too short and unknown
    if not expanded_terms:
        return (
            f"Error: could not expand any keywords in '{query}'. "
            f"All terms were shorter than 4 characters and unknown abbreviations. "
            f"Please use full words (e.g. 'malta' instead of 'mt')."
        )
    planned = " OR ".join(dict.fromkeys(expanded_terms))  # deduplicate, preserve order
    expansion_note = f" (expanded: {', '.join(notes)})" if notes else ""
    _reranker_cfg = (session_ctx or {}).get("settings") and session_ctx["settings"].reranker  # type: ignore[index]
    _embed_cfg = (session_ctx or {}).get("settings") and session_ctx["settings"].embeddings  # type: ignore[index]
    _llm_base_url = ((session_ctx or {}).get("settings") and session_ctx["settings"].llm.base_url) or ""  # type: ignore[index]
    _call_llm_fn = (session_ctx or {}).get("call_llm_fn")
    _hyde_llm_kwargs = (session_ctx or {}).get("hyde_llm_kwargs")
    _crag_cfg = (session_ctx or {}).get("settings") and session_ctx["settings"].crag  # type: ignore[index]
    chunks = retrieve(store, query, top_n=max_results, planned_query=planned, reranker_cfg=_reranker_cfg or None, embed_cfg=_embed_cfg or None, llm_base_url=_llm_base_url, call_llm_fn=_call_llm_fn, llm_kwargs=_hyde_llm_kwargs, crag_cfg=_crag_cfg or None)

    # Apply date/platform post-filters
    if since_filter:
        chunks = [c for c in chunks if c.record.timestamp and c.record.timestamp.isoformat() >= since_filter]
    if until_filter:
        until_end = until_filter if "T" in until_filter else f"{until_filter}T23:59:59"
        chunks = [c for c in chunks if c.record.timestamp and c.record.timestamp.isoformat() <= until_end]
    if platform_filter:
        chunks = [c for c in chunks if c.record.platform == platform_filter]

    # Apply file-type filter when the caller supplied one (e.g. file_type="pdf").
    file_type = args.get("file_type", "").strip().lstrip(".").lower()
    if file_type and chunks:
        chunks = [
            c for c in chunks
            if c.record.file_path and c.record.file_path.lower().endswith(f".{file_type}")
        ]
        if not chunks:
            return f"No {file_type.upper()} files found in vault matching: {planned}{expansion_note}"

    # Apply sort_by — default is relevance (already ranked); 'date' re-sorts newest first.
    sort_by = (args.get("sort_by") or "relevance").strip().lower()
    if sort_by == "date":
        chunks.sort(
            key=lambda c: c.record.timestamp.isoformat() if c.record.timestamp else "",
            reverse=True,
        )

    collected_chunks.extend(chunks)
    if not chunks:
        return f"No results found in vault for: {planned}{expansion_note}"
    ctx = assemble_context(chunks)

    # Pre-extract the exact matching lines so the LLM only sees the answers
    matching_lines = _extract_matching_lines(ctx, expanded_terms)
    if matching_lines:
        lines_block = "\n".join(f"  {i+1}. {ln}" for i, ln in enumerate(matching_lines))
        # Count only content lines; [N] header lines are context, not mentions.
        mention_count = sum(
            1 for ln in matching_lines if not re.match(r'^\[\d+\]', ln)
        )
        # Append a deduplicated list of actual file paths so the LLM can use them
        # when it needs to write a file — without this it only has source labels.
        unique_paths = list(dict.fromkeys(
            c.record.file_path for c in chunks if c.record.file_path
        ))
        paths_block = ""
        if unique_paths:
            paths_list = "\n".join(f"  {p}" for p in unique_paths)
            paths_block = f"\n\nFILE PATHS ({len(unique_paths)} unique files with matches):\n{paths_list}"

        # Attachment call-to-action: tell the LLM which records have image
        # attachments so it will call fetch_attachment next.
        # Only activate when the query explicitly asks for images/photos,
        # otherwise non-image queries that happen to retrieve records with
        # image attachments (e.g. visa scans) would get a wrong response.
        _IMAGE_QUERY_KEYWORDS: frozenset[str] = frozenset(
            {"image", "images", "photo", "photos", "picture", "pictures",
             "img", "jpg", "jpeg", "png", "gif", "screenshot", "screenshots"}
        )
        _query_lower = query.lower()
        _query_has_image_kw = any(kw in _query_lower for kw in _IMAGE_QUERY_KEYWORDS)
        att_cta_lines: list[str] = []
        if _query_has_image_kw:
            for c in chunks:
                atts = c.record.attachments or []
                img_atts = [
                    a for a in atts
                    if any(a.lower().endswith(ext) for ext in _IMAGE_EXTS)
                    or "image" in a.lower()
                ]
                for att in img_atts:
                    ts = c.record.timestamp.strftime("%Y-%m-%d") if c.record.timestamp else ""
                    att_cta_lines.append(
                        f"  record_id={c.record.id!r}, attachment_name={att!r}"
                        f"  (from: {c.record.thread_name or c.record.platform}, {ts})"
                    )
        att_block = ""
        if att_cta_lines:
            deduped = list(dict.fromkeys(att_cta_lines))[:10]
            att_block = (
                "\n\nIMAGE ATTACHMENTS FOUND — call fetch_attachment for each to download and display them:\n"
                + "\n".join(deduped)
            )

        # Auto-download: when image keywords are in the query, download the
        # images directly without requiring a second LLM tool call.
        # Small models (Gemma E2B etc.) won't reliably chain two tool calls.
        _auto_ctx = session_ctx or {}
        _auto_settings = _auto_ctx.get("settings")
        if att_cta_lines and _auto_settings is not None:
            _data_dir = Path(_auto_settings.vault_db).parent
            from egovault.utils.gmail_imap import fetch_attachment_bytes as _fab, load_credentials as _lc
            if _lc(_data_dir) is not None:
                import json as _json
                _att_dir = _data_dir / "attachments"
                _att_dir.mkdir(parents=True, exist_ok=True)
                _downloaded: list[str] = []
                for c in chunks[:5]:  # limit to 5 most relevant records
                    atts = c.record.attachments or []
                    img_atts = [
                        a for a in atts
                        if any(a.lower().endswith(ext) for ext in _IMAGE_EXTS)
                        or "image" in a.lower()
                    ]
                    raw_r = c.record.raw if isinstance(c.record.raw, dict) else _json.loads(c.record.raw or "{}")
                    msg_id = raw_r.get("message_id", "")
                    if not msg_id or c.record.platform != "gmail":
                        continue
                    for att_name in img_atts[:3]:  # max 3 per email
                        safe_name = f"{c.record.id[:12]}_{att_name}"
                        out_path = _att_dir / safe_name
                        if out_path.exists():
                            # already downloaded
                            _downloaded.append(str(out_path))
                            continue
                        try:
                            att_bytes = _fab(_data_dir, msg_id, att_name)
                            if att_bytes:
                                out_path.write_bytes(att_bytes)
                                _downloaded.append(str(out_path))
                        except Exception:
                            pass
                if _downloaded:
                    _auto_ctx.setdefault("saved_attachments", []).extend(_downloaded)
                    att_block += (
                        f"\n\nAUTO-DOWNLOADED {len(_downloaded)} image(s) — "
                        f"they will be displayed in the chat automatically."
                    )

        # When image attachments are present, replace the raw numbered list with
        # a clean thread-level summary so the small LLM writes natural prose instead
        # of echoing the line-by-line data.
        if att_block:
            # Build a deduplicated thread summary: thread → (date, [img filenames])
            seen_thread: dict[str, tuple[str, list[str]]] = {}
            for c in chunks:
                atts = c.record.attachments or []
                img_atts = [
                    a for a in atts
                    if any(a.lower().endswith(ext) for ext in _IMAGE_EXTS)
                    or "image" in a.lower()
                ]
                if not img_atts:
                    continue
                thread = c.record.thread_name or c.record.platform or "unknown"
                ts = c.record.timestamp.strftime("%Y-%m-%d") if c.record.timestamp else ""
                if thread not in seen_thread:
                    seen_thread[thread] = (ts, [])
                seen_thread[thread][1].extend(
                    a for a in img_atts if a not in seen_thread[thread][1]
                )
            thread_lines = "\n".join(
                f"  - {thread} ({date_s}): {len(imgs)} image(s)"
                for thread, (date_s, imgs) in seen_thread.items()
            )
            n_downloaded = len(_auto_ctx.get("saved_attachments", []))
            return (
                f"TOOL RESULT: Found images you sent to Gary in these email threads:\n"
                f"{thread_lines}\n\n"
                f"{n_downloaded} image(s) have been downloaded and will be shown below automatically.\n\n"
                f"Reply in 2-3 sentences: name the threads and dates, say how many images were found, "
                f"and confirm they are displayed below."
            )

        return (
            f"VAULT DATA for '{', '.join(expanded_terms)}'{expansion_note}.\n"
            f"Found {mention_count} mention(s). Answer the user's question using the records below. "
            f"Cite [N] source labels when quoting specific facts. Do NOT repeat the records verbatim.\n\n"
            f"{ctx}"
            f"{paths_block}"
        )
    return f"No matching lines found for '{planned}'{expansion_note} in the vault records."


def _tool_count_records(
    args: dict,
    store: "VaultStore",
    top_n: int,
    collected_chunks: list,
    session_ctx: "dict | None",
) -> str:
    platform_filter = (args.get("platform") or "").strip() or None
    since_filter = (args.get("since") or "").strip() or None
    until_filter = (args.get("until") or "").strip() or None
    result = store.count_records(
        platform=platform_filter, since=since_filter, until=until_filter
    )
    lines: list[str] = []
    filter_parts: list[str] = []
    if platform_filter:
        filter_parts.append(f"platform={platform_filter}")
    if since_filter:
        filter_parts.append(f"since={since_filter}")
    if until_filter:
        filter_parts.append(f"until={until_filter}")
    filter_desc = f" ({', '.join(filter_parts)})" if filter_parts else ""
    lines.append(f"Total records{filter_desc}: {result['total']}")
    if len(result["breakdown"]) > 1:
        lines.append("Breakdown by platform:")
        for item in result["breakdown"]:
            lines.append(f"  {item['platform']}: {item['count']}")
    return "\n".join(lines)


def _tool_get_vault_stats(
    args: dict,
    store: "VaultStore",
    top_n: int,
    collected_chunks: list,
    session_ctx: "dict | None",
) -> str:
    return vault_summary_context(store) or "The vault is empty."


def _tool_scan_folder(
    args: dict,
    store: "VaultStore",
    top_n: int,
    collected_chunks: list,
    session_ctx: "dict | None",
) -> str:
    path_arg = args.get("path", "").strip()
    if not path_arg:
        return "Error: path parameter is required."
    ctx = session_ctx or {}
    settings = ctx.get("settings")
    if settings is None:
        return "Error: scan not available (no settings context)."
    from egovault.adapters.local_inbox import LocalInboxAdapter
    from egovault.utils.folders import resolve_folder
    # Resolve the built-in 'inbox' alias to the vault's configured drop-inbox dir.
    # Strip leading/trailing slashes and whitespace before matching.
    path_normalized = path_arg.strip().strip("/\\").strip()
    if re.search(r'^inbox$', path_normalized, re.IGNORECASE) or \
            re.search(r'\b(my|the)\s+inbox\b', path_arg.strip(), re.IGNORECASE):
        path_arg = str(Path(settings.inbox_dir).expanduser().resolve())
    try:
        src = resolve_folder(path_arg)
    except ValueError as exc:
        return f"Error: {exc}"
    if not src.exists() or not src.is_dir():
        return f"Not an existing directory: {src}"
    adapter = LocalInboxAdapter(store=store)
    if not adapter.can_handle(src):
        return f"No supported files found in {src}"
    # Count total candidate files upfront so the result is accurate even when
    # the adapter filters already-indexed ones before yielding them.
    from egovault.adapters.local_inbox import SUPPORTED_SUFFIXES
    total_files = sum(
        1 for f in src.rglob("*")
        if f.is_file() and f.suffix.lower() in SUPPORTED_SUFFIXES
    )
    inserted = skipped = 0
    for record in adapter.ingest(src):
        try:
            was_new = store.upsert_record(record)
            if was_new:
                inserted += 1
            else:
                if record.file_path and record.body:
                    store.update_body_by_file_path(record.file_path, record.body)
                skipped += 1
        except Exception:
            skipped += 1
    already_indexed = total_files - inserted
    # Signal session to refresh profile after scan
    if "owner_profile_ref" in ctx:
        ctx["owner_profile_ref"]["dirty"] = True
    return (
        f"SCAN COMPLETE — {src}:\n"
        f"New records added: {inserted}\n"
        f"Already known (skipped): {already_indexed}\n"
        f"Total files in folder: {total_files}"
    )


def _tool_get_sources(
    args: dict,
    store: "VaultStore",
    top_n: int,
    collected_chunks: list,
    session_ctx: "dict | None",
) -> str:
    ctx = session_ctx or {}
    sources = ctx.get("last_sources", [])
    if not sources:
        return "No sources from the previous answer (either no vault records were used, or this is the first message)."
    return "Sources used in last answer:\n" + "\n".join(f"  • {s}" for s in sources)


def _tool_get_status(
    args: dict,
    store: "VaultStore",
    top_n: int,
    collected_chunks: list,
    session_ctx: "dict | None",
) -> str:
    ctx = session_ctx or {}
    settings = ctx.get("settings")
    if settings is None:
        return "Status unavailable."
    llm = settings.llm
    # Try llama-server /health or /v1/models for a live status
    import urllib.request as _ur
    server_ok = False
    for _path in ("/health", "/v1/models"):
        try:
            with _ur.urlopen(llm.base_url.rstrip("/") + _path, timeout=3) as _r:
                if _r.status == 200:
                    server_ok = True
                    break
        except Exception:
            pass
    status = "reachable" if server_ok else "unreachable"
    return f"Model: {llm.model}\nServer: {llm.base_url}  ({status})"


def _tool_get_profile(
    args: dict,
    store: "VaultStore",
    top_n: int,
    collected_chunks: list,
    session_ctx: "dict | None",
) -> str:
    ctx = session_ctx or {}
    profile = ctx.get("owner_profile", "") or store.get_owner_profile()
    return profile if profile else "No profile extracted yet. Try asking me to refresh it."


def _tool_read_file(
    args: dict,
    store: "VaultStore",
    top_n: int,
    collected_chunks: list,
    session_ctx: "dict | None",
) -> str:
    path = args.get("path", "").strip()
    if not path:
        return "Error: path parameter is required."
    max_chars = int(args.get("max_chars") or 4000)
    try:
        p = Path(path)
        if not p.exists():
            return f"File not found: {path}"
        if not p.is_file():
            return f"Not a file: {path}"
        content = p.read_text(encoding="utf-8", errors="replace")
        if len(content) > max_chars:
            return content[:max_chars] + f"\n\n[… truncated at {max_chars} chars, file is {len(content)} total]"
        return content
    except (PermissionError, OSError) as exc:
        return f"Could not read file: {exc}"


def _tool_list_directory(
    args: dict,
    store: "VaultStore",
    top_n: int,
    collected_chunks: list,
    session_ctx: "dict | None",
) -> str:
    dir_path = args.get("path", "").strip()
    if not dir_path:
        return "Error: path parameter is required."
    pattern = args.get("pattern", "*") or "*"
    try:
        p = Path(dir_path)
        if not p.exists():
            return f"Directory not found: {dir_path}"
        if not p.is_dir():
            return f"Not a directory: {dir_path}"
        entries = sorted(p.glob(pattern))
        if not entries:
            return f"No files matching '{pattern}' in {dir_path}"
        lines = [f"Contents of {dir_path} (pattern: {pattern}):"]
        for entry in entries[:_DIR_LIST_CAP]:  # cap at _DIR_LIST_CAP entries
            kind = "DIR " if entry.is_dir() else "FILE"
            lines.append(f"  {kind}  {entry.name}")
        if len(entries) > 200:
            lines.append(f"  … and {len(entries) - 200} more")
        return "\n".join(lines)
    except (PermissionError, OSError) as exc:
        return f"Could not list directory: {exc}"


def _tool_fetch_attachment(
    args: dict,
    store: "VaultStore",
    top_n: int,
    collected_chunks: list,
    session_ctx: "dict | None",
) -> str:
    record_id = args.get("record_id", "").strip()
    attachment_name = args.get("attachment_name", "").strip()
    if not record_id or not attachment_name:
        return "Error: record_id and attachment_name are required."

    # Look up the record to get the Message-ID and verify it exists
    row = store._con.execute(
        "SELECT raw, platform, thread_name, sender_name, timestamp FROM normalized_records WHERE id = ?",
        (record_id,),
    ).fetchone()
    if row is None:
        return f"Error: record '{record_id}' not found in vault."
    if row["platform"] != "gmail":
        return "Error: fetch_attachment only supports Gmail records."

    import json as _json
    raw_data = _json.loads(row["raw"] or "{}")
    message_id = raw_data.get("message_id", "")
    if not message_id:
        return "Error: no Message-ID stored for this record — cannot fetch attachment."

    # Get credentials path from settings
    ctx = session_ctx or {}
    settings = ctx.get("settings")
    if settings is None:
        return "Error: no settings context — cannot connect to Gmail."
    data_dir = Path(settings.vault_db).parent

    from egovault.utils.gmail_imap import fetch_attachment_bytes
    try:
        att_bytes = fetch_attachment_bytes(data_dir, message_id, attachment_name)
    except Exception as exc:
        return f"Error fetching attachment: {exc}"

    if att_bytes is None:
        return f"Attachment '{attachment_name}' not found in Gmail message {message_id}."

    # Save to a local attachments folder so Streamlit can display it
    att_dir = Path(settings.vault_db).parent / "attachments"
    att_dir.mkdir(parents=True, exist_ok=True)
    # Unique filename: record_id prefix + original name
    safe_name = f"{record_id[:12]}_{attachment_name}"
    out_path = att_dir / safe_name
    out_path.write_bytes(att_bytes)

    # Record the saved path so the UI can render it
    if session_ctx is not None:
        session_ctx.setdefault("saved_attachments", []).append(str(out_path))

    return f"ATTACHMENT_SAVED:{out_path}"


def _tool_write_file(
    args: dict,
    store: "VaultStore",
    top_n: int,
    collected_chunks: list,
    session_ctx: "dict | None",
) -> str:
    path = args.get("path", "").strip()
    content = args.get("content", "")
    overwrite = bool(args.get("overwrite", False))
    if not path:
        return "Error: path parameter is required."
    # For list-format exports (csv/tsv/json) always build real content from the
    # chunks the search already collected — the model can't reliably extract paths.
    _list_exts = {"csv", "tsv", "json"}
    _path_ext = Path(path).suffix.lstrip(".").lower()
    if _path_ext in _list_exts and collected_chunks:
        rescued = _build_file_export(args, collected_chunks, session_ctx.get("settings") if session_ctx else None)  # type: ignore[arg-type]
        if rescued:
            content = rescued
    # For other formats, fall back to stub detection.
    if not content:
        return (
            "Error: content is required. Call search_vault first to collect data, "
            "then call write_file with the real content."
        )
    if len(content.splitlines()) <= 1 and len(content) < 50:
        if collected_chunks:
            rescued = _build_file_export(args, collected_chunks, session_ctx.get("settings") if session_ctx else None)  # type: ignore[arg-type]
            if rescued:
                content = rescued
            else:
                return "Error: could not extract file paths from search results."
        else:
            return (
                "Error: content appears empty or stub-like. "
                "Call search_vault first to collect data, then write_file with real content."
            )
    try:
        p = _resolve_write_target_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # Avoid clobbering existing files unless overwrite=true
        if p.exists() and not overwrite:
            from datetime import datetime as _dt
            ts = _dt.now().strftime("%Y%m%d_%H%M%S")
            p = p.with_stem(f"{p.stem}_{ts}") if hasattr(p, "with_stem") else p.with_name(f"{p.stem}_{ts}{p.suffix}")
        p.write_text(content, encoding="utf-8")
        if session_ctx is not None:
            session_ctx["last_file"] = str(p)
        return f"FILE WRITTEN: {p}\nSize: {len(content)} chars  |  Path: {p}"
    except (PermissionError, OSError) as exc:
        return f"Could not write file: {exc}"


def _tool_get_record(
    args: dict,
    store: "VaultStore",
    top_n: int,
    collected_chunks: list,
    session_ctx: "dict | None",
) -> str:
    record_id = str(args.get("record_id") or "").strip()
    if not record_id:
        return "Error: record_id parameter is required."
    row = store._con.execute(
        "SELECT * FROM normalized_records WHERE id = ?",
        (record_id,),
    ).fetchone()
    if row is None:
        return f"Error: record '{record_id}' not found in vault."
    from egovault.core.store import row_to_record as _r2r
    rec = _r2r(row)
    ts = rec.timestamp.strftime("%Y-%m-%d %H:%M") if rec.timestamp else ""
    sender = f"From: {rec.sender_name}\n" if rec.sender_name else ""
    subject = f"Subject: {rec.thread_name}\n" if rec.thread_name else ""
    atts = ", ".join(rec.attachments or [])
    att_line = f"Attachments: {atts}\n" if atts else ""
    body = rec.body or "(no body)"
    return (
        f"Platform: {rec.platform}  |  Date: {ts}\n"
        f"{sender}{subject}{att_line}"
        f"\n{body}"
    )


def _tool_gmail_sync(
    args: dict,
    store: "VaultStore",
    top_n: int,
    collected_chunks: list,
    session_ctx: "dict | None",
) -> str:
    ctx = session_ctx or {}
    settings = ctx.get("settings")
    if settings is None:
        return "Error: no settings context — cannot connect to Gmail."
    since_arg = (args.get("since") or "").strip()
    max_results_arg = int(args.get("max_results") or 500)
    data_dir = Path(settings.vault_db).parent
    from egovault.utils.gmail_imap import load_credentials as _lc
    imap_creds = _lc(data_dir)
    if imap_creds is None:
        return (
            "Gmail is not connected. "
            "Ask the user to run /gmail-auth to set up their App Password first."
        )
    if not since_arg:
        since_arg = store.get_setting("gmail_last_sync") or ""
    oldest_synced = store.get_setting("gmail_oldest_synced") or ""
    # Bootstrap: if we have a last-sync date but have never tracked the
    # oldest email, start the backward pass from before the last-sync date.
    if not oldest_synced and since_arg:
        oldest_synced = since_arg
    gmail_address, app_password = imap_creds
    from egovault.adapters.gmail_imap_adapter import GmailImapAdapter
    adapter = GmailImapAdapter(store=store)
    inserted = skipped = 0
    oldest_ts = None
    _pcb = (session_ctx or {}).get("_progress_cb")
    _last_pct: list[int] = [0]  # mutable cell for throttle state

    def _imap_progress(fetched: int, total: int) -> None:
        if _pcb is None or total == 0:
            return
        pct = int(fetched * 100 / total)
        # Emit at ~10 % intervals to avoid flooding Streamlit with hundreds
        # of lines.  Always emit first fetch (fetched==1) and the last one.
        if fetched == 1 or fetched == total or pct >= _last_pct[0] + 10:
            _last_pct[0] = pct
            _pcb(
                f"\u2699 Syncing emails\u2026 {fetched}/{total} "
                f"({inserted} new, {skipped} already in vault)"
            )

    try:
        for record in adapter.ingest_from_imap(
            gmail_address=gmail_address,
            app_password=app_password,
            since=since_arg,
            before=oldest_synced,
            max_results=max_results_arg,
            progress_callback=_imap_progress,
        ):
            was_new = store.upsert_record(record)
            if was_new:
                inserted += 1
            else:
                skipped += 1
            # Track oldest timestamp from ALL fetched records so the
            # backward frontier advances even on fully-up-to-date windows.
            if record.timestamp and (oldest_ts is None or record.timestamp < oldest_ts):
                oldest_ts = record.timestamp
    except Exception as exc:  # noqa: BLE001
        return f"Gmail sync failed: {exc}"
    from datetime import date as _date
    store.set_setting("gmail_last_sync", _date.today().strftime("%Y-%m-%d"))
    if oldest_ts is not None:
        oldest_str = oldest_ts.strftime("%Y-%m-%d")
        current_oldest = store.get_setting("gmail_oldest_synced") or ""
        if not current_oldest or oldest_str < current_oldest:
            store.set_setting("gmail_oldest_synced", oldest_str)
    return (
        f"GMAIL_SYNC_COMPLETE: {inserted} new email(s) added, {skipped} already in vault."
        + (f" (since {since_arg})" if since_arg else "")
        + ("\nNew emails are now searchable — call search_vault to find them." if inserted > 0 else "")
    )


def _tool_open_file(
    args: dict,
    store: "VaultStore",
    top_n: int,
    collected_chunks: list,
    session_ctx: "dict | None",
) -> str:
    path = (args.get("path") or "").strip()
    if not path:
        return "Error: path parameter is required."

    # ── URL download path ─────────────────────────────────────────────────
    if path.startswith(("http://", "https://")):
        import urllib.request as _ur
        import urllib.parse as _up
        ctx = session_ctx or {}
        settings = ctx.get("settings")
        if settings is not None:
            save_dir = Path(settings.output_dir).expanduser().resolve()
        else:
            save_dir = Path.home() / "Downloads"
        save_dir.mkdir(parents=True, exist_ok=True)

        # Determine filename: explicit arg → URL path segment → fallback
        raw_filename = (args.get("filename") or "").strip()
        if raw_filename:
            dest_name = re.sub(r'[/\\:*?"<>|]', "_", raw_filename)
        else:
            url_path = _up.urlparse(path).path
            dest_name = Path(url_path).name or "download"
            dest_name = re.sub(r'[/\\:*?"<>|]', "_", dest_name) or "download"

        dest = save_dir / dest_name
        # Avoid clobbering
        if dest.exists():
            from datetime import datetime as _dt
            ts = _dt.now().strftime("%Y%m%d_%H%M%S")
            dest = save_dir / f"{dest.stem}_{ts}{dest.suffix}"

        try:
            _ur.urlretrieve(path, str(dest))  # noqa: S310
        except Exception as exc:
            return f"Download failed: {exc}"

        size_kb = dest.stat().st_size // 1024
        if session_ctx is not None:
            session_ctx["last_file"] = str(dest)
        _open_with_default_app(str(dest))
        return f"FILE_DOWNLOADED:{dest}\nSize: {size_kb} KB"

    # ── Local file open ───────────────────────────────────────────────────
    return _open_with_default_app(path)


def _tool_take_screenshot(
    args: dict,
    store: "VaultStore",
    top_n: int,
    collected_chunks: list,
    session_ctx: "dict | None",
) -> str:
    from datetime import datetime as _dt
    import platform as _platform_ts

    ctx = session_ctx or {}
    settings = ctx.get("settings")
    # Save to the data/attachments folder so the UI can render it
    if settings is not None:
        att_dir = Path(settings.vault_db).parent / "attachments"
    else:
        att_dir = Path.home() / ".egovault" / "attachments"
    att_dir.mkdir(parents=True, exist_ok=True)

    raw_filename = (args.get("filename") or "").strip()
    ts_str = _dt.now().strftime("%Y%m%d_%H%M%S")
    stem = raw_filename if raw_filename else f"screenshot_{ts_str}"
    # Sanitise: remove path separators to prevent directory traversal
    stem = re.sub(r'[/\\:*?"<>|]', "_", stem)
    out_path = att_dir / f"{stem}.png"
    # Avoid clobbering existing file
    if out_path.exists():
        out_path = att_dir / f"{stem}_{ts_str}.png"

    region_raw = (args.get("region") or "").strip()
    bbox = None
    if region_raw:
        try:
            parts = [int(v.strip()) for v in region_raw.split(",")]
            if len(parts) == 4:
                x, y, w, h = parts
                bbox = (x, y, x + w, y + h)
        except ValueError:
            pass  # ignore malformed region — fall back to full screen

    try:
        from PIL import ImageGrab as _IG_ts
        if bbox and _platform_ts.system() == "Windows":
            # GetWindowRect coords are logical pixels; ImageGrab returns physical.
            # Scale bbox by the DPI factor before passing to ImageGrab.
            import ctypes as _ct_ts
            _u32 = _ct_ts.windll.user32  # type: ignore[attr-defined]
            _vw_l = _u32.GetSystemMetrics(78)  # SM_CXVIRTUALSCREEN (logical)
            _full_ts = _IG_ts.grab(all_screens=True)
            _scale = _full_ts.width / _vw_l if _vw_l > 0 else 1.0
            _vx_l = _u32.GetSystemMetrics(76)  # SM_XVIRTUALSCREEN (logical)
            _vy_l = _u32.GetSystemMetrics(77)  # SM_YVIRTUALSCREEN (logical)
            _cx0 = max(0, round((bbox[0] - _vx_l) * _scale))
            _cy0 = max(0, round((bbox[1] - _vy_l) * _scale))
            _cx1 = min(_full_ts.width, round((bbox[2] - _vx_l) * _scale))
            _cy1 = min(_full_ts.height, round((bbox[3] - _vy_l) * _scale))
            img = _full_ts.crop((_cx0, _cy0, _cx1, _cy1))
        else:
            img = _IG_ts.grab(all_screens=True)
        img.save(str(out_path), format="PNG")
    except Exception as exc:
        return f"Screenshot failed: {exc}"

    if session_ctx is not None:
        session_ctx.setdefault("saved_attachments", []).append(str(out_path))
        session_ctx["last_file"] = str(out_path)

    size_kb = out_path.stat().st_size // 1024
    region_desc = f" (region {region_raw})" if region_raw and bbox else " (full screen)"
    return f"SCREENSHOT_SAVED:{out_path}\nSize: {size_kb} KB{region_desc}"


def _tool_inspect_windows(
    args: dict,
    store: "VaultStore",
    top_n: int,
    collected_chunks: list,
    session_ctx: "dict | None",
) -> str:
    from datetime import datetime as _dt
    import platform as _platform

    ctx = session_ctx or {}
    settings = ctx.get("settings")
    if settings is not None:
        att_dir = Path(settings.vault_db).parent / "attachments"
    else:
        att_dir = Path.home() / ".egovault" / "attachments"
    att_dir.mkdir(parents=True, exist_ok=True)

    filter_str = (args.get("filter") or "").strip()
    want_screenshots = args.get("include_screenshots", True)
    ts_str = _dt.now().strftime("%Y%m%d_%H%M%S")

    windows = _list_windows_cross_platform(filter_str)
    if not windows:
        msg = "No visible windows found"
        if filter_str:
            msg += f" matching '{filter_str}'"
        return msg + "."

    lines: list[str] = [
        f"WINDOWS_INSPECTED:{len(windows)}",
        f"Platform: {_platform.system()}",
        "",
    ]
    saved_paths: list[str] = []

    # Take ONE full virtual-screen grab up-front, then crop per window.
    # ImageGrab.grab(all_screens=True) works from background threads and returns
    # physical pixels. On Windows with DPI scaling the logical window coordinates
    # from GetWindowRect must be scaled by (img.width / SM_CXVIRTUALSCREEN) to
    # map to physical pixel positions in the captured image.
    full_img = None
    _dpi_scale = 1.0
    virt_x = virt_y = 0
    if want_screenshots:
        try:
            from PIL import ImageGrab as _IG
            full_img = _IG.grab(all_screens=True)
            if _platform.system() == "Windows":
                import ctypes as _ct
                _u32 = _ct.windll.user32  # type: ignore[attr-defined]
                virt_x = _u32.GetSystemMetrics(76)  # SM_XVIRTUALSCREEN (logical)
                virt_y = _u32.GetSystemMetrics(77)  # SM_YVIRTUALSCREEN (logical)
                _vw_l = _u32.GetSystemMetrics(78)   # SM_CXVIRTUALSCREEN (logical)
                _dpi_scale = full_img.width / _vw_l if _vw_l > 0 else 1.0
        except Exception as exc:
            lines.append(f"Full-screen grab failed: {exc}")
            want_screenshots = False

    for i, win in enumerate(windows, 1):
        title = win["title"]
        proc = win["process"]
        left, top, w, h = win["left"], win["top"], win["width"], win["height"]
        proc_info = f" [{proc}]" if proc else ""
        lines.append(f"{i}. {title}{proc_info} — {w}×{h} at ({left},{top})")

        if want_screenshots and full_img is not None:
            try:
                # Translate logical screen coords → physical image coords:
                # 1. subtract the virtual screen origin (logical)
                # 2. multiply by DPI scale (physical_px / logical_px)
                cx0 = round((left - virt_x) * _dpi_scale)
                cy0 = round((top - virt_y) * _dpi_scale)
                cx1 = round((left - virt_x + w) * _dpi_scale)
                cy1 = round((top - virt_y + h) * _dpi_scale)
                cx0 = max(0, min(cx0, full_img.width - 1))
                cy0 = max(0, min(cy0, full_img.height - 1))
                cx1 = max(cx0 + 1, min(cx1, full_img.width))
                cy1 = max(cy0 + 1, min(cy1, full_img.height))
                img = full_img.crop((cx0, cy0, cx1, cy1))
                safe_title = re.sub(r'[/\\:*?"<>|]', "_", title[:40]).strip()
                fname = f"win_{i:02d}_{safe_title}_{ts_str}.png"
                out_path = att_dir / fname
                img.save(str(out_path), format="PNG")
                saved_paths.append(str(out_path))
                lines.append(f"   Screenshot: {out_path}")
            except Exception as exc:
                lines.append(f"   Screenshot failed: {exc}")

    if session_ctx is not None and saved_paths:
        session_ctx.setdefault("saved_attachments", []).extend(saved_paths)
        session_ctx["last_file"] = saved_paths[-1]

    return "\n".join(lines)


def _tool_inspect_window(
    args: dict,
    store: "VaultStore",
    top_n: int,
    collected_chunks: list,
    session_ctx: "dict | None",
) -> str:
    from datetime import datetime as _dt
    import platform as _platform_iw

    ctx = session_ctx or {}
    settings = ctx.get("settings")
    if settings is not None:
        att_dir = Path(settings.vault_db).parent / "attachments"
        out_dir = Path(settings.output_dir).expanduser().resolve()
    else:
        att_dir = Path.home() / ".egovault" / "attachments"
        out_dir = Path.home() / ".egovault" / "output"
    att_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    title_filter = (args.get("title_filter") or "").strip()
    if not title_filter:
        return "Error: title_filter is required for inspect_window."

    include_screenshot = args.get("include_screenshot", True)
    output_file_arg = (args.get("output_file") or "").strip()
    max_depth_arg = min(8, max(1, int(args.get("max_depth") or 5)))
    ts_str = _dt.now().strftime("%Y%m%d_%H%M%S")

    if _platform_iw.system() != "Windows":
        return "inspect_window is only supported on Windows."

    import ctypes as _ct_iw
    import ctypes.wintypes as _wt_iw
    _u32_iw = _ct_iw.windll.user32  # type: ignore[attr-defined]

    # Find the target window by title substring
    target_hwnd = 0
    target_title = ""
    _EWP = _ct_iw.WINFUNCTYPE(_ct_iw.c_bool, _wt_iw.HWND, _ct_iw.c_long)

    def _find_cb(h: int, _: int) -> bool:
        nonlocal target_hwnd, target_title
        if not _u32_iw.IsWindowVisible(h) or _u32_iw.IsIconic(h):
            return True
        buf = _ct_iw.create_unicode_buffer(256)
        _u32_iw.GetWindowTextW(h, buf, 256)
        if title_filter.lower() in buf.value.lower() and buf.value.strip():
            target_hwnd = h
            target_title = buf.value.strip()
            return False  # stop on first match
        return True

    _u32_iw.EnumWindows(_EWP(_find_cb), 0)

    if not target_hwnd:
        return f"No visible window found matching '{title_filter}'."

    # Walk the element tree
    element_lines = _enumerate_window_elements(target_hwnd, max_depth=max_depth_arg)

    # Build output text
    rect = _wt_iw.RECT()
    _u32_iw.GetWindowRect(target_hwnd, _ct_iw.byref(rect))
    win_w = rect.right - rect.left
    win_h = rect.bottom - rect.top
    header = (
        f"Window: {target_title}\n"
        f"Size: {win_w}×{win_h} at ({rect.left},{rect.top})\n"
        f"Elements ({len(element_lines)} nodes, depth={max_depth_arg}):\n"
    )
    text_output = header + "\n".join(element_lines)

    # Save text file
    if output_file_arg:
        txt_path = Path(output_file_arg)
    else:
        safe_name = re.sub(r'[/\\:*?"<>|]', "_", title_filter[:40]).strip()
        txt_path = out_dir / f"ui_elements_{safe_name}_{ts_str}.txt"
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text(text_output, encoding="utf-8")

    result_lines = [f"UI_ELEMENTS_SAVED:{txt_path}"]
    result_lines.append(f"Window: {target_title} ({win_w}×{win_h})")
    result_lines.append(f"Elements found: {len(element_lines)}")
    result_lines.append(f"Saved to: {txt_path}")

    if session_ctx is not None:
        session_ctx["last_file"] = str(txt_path)

    # Optional screenshot of this specific window
    if include_screenshot:
        try:
            from PIL import ImageGrab as _IG_iw
            import ctypes as _ct_iw2
            _u32_iw2 = _ct_iw2.windll.user32  # type: ignore[attr-defined]
            _vx = _u32_iw2.GetSystemMetrics(76)
            _vy = _u32_iw2.GetSystemMetrics(77)
            _vw_l = _u32_iw2.GetSystemMetrics(78)
            full_img = _IG_iw.grab(all_screens=True)
            _scale = full_img.width / _vw_l if _vw_l > 0 else 1.0
            cx0 = max(0, round((rect.left - _vx) * _scale))
            cy0 = max(0, round((rect.top - _vy) * _scale))
            cx1 = min(full_img.width, round((rect.right - _vx) * _scale))
            cy1 = min(full_img.height, round((rect.bottom - _vy) * _scale))
            crop = full_img.crop((cx0, cy0, cx1, cy1))
            safe_name_png = re.sub(r'[/\\:*?"<>|]', "_", title_filter[:40]).strip()
            png_path = att_dir / f"ui_{safe_name_png}_{ts_str}.png"
            crop.save(str(png_path), format="PNG")
            result_lines.append(f"Screenshot: {png_path}")
            if session_ctx is not None:
                session_ctx.setdefault("saved_attachments", []).append(str(png_path))
        except Exception as exc:
            result_lines.append(f"Screenshot failed: {exc}")

    return "\n".join(result_lines)


def _tool_system_info(
    args: dict,
    store: "VaultStore",
    top_n: int,
    collected_chunks: list,
    session_ctx: "dict | None",
) -> str:
    import platform as _plat_si
    import subprocess as _sp_si

    sections_raw = args.get("sections") or ["all"]
    if isinstance(sections_raw, str):
        sections_raw = [sections_raw]
    want_all = "all" in sections_raw or not sections_raw

    def _want(sec: str) -> bool:
        return want_all or sec in sections_raw

    lines_si: list[str] = []

    # ── OS / host ──────────────────────────────────────────────────────────
    if _want("os"):
        lines_si.append("=== System ===")
        lines_si.append(f"OS: {_plat_si.system()} {_plat_si.release()} ({_plat_si.version()})")
        lines_si.append(f"Architecture: {_plat_si.machine()}")
        lines_si.append(f"Processor: {_plat_si.processor()}")
        lines_si.append(f"Hostname: {_plat_si.node()}")
        lines_si.append("")

    # ── CPU ────────────────────────────────────────────────────────────────
    if _want("cpu"):
        lines_si.append("=== CPU ===")
        try:
            import psutil as _psu_si  # noqa: PLC0415
            cpu_p = _psu_si.cpu_count(logical=False) or "?"
            cpu_l = _psu_si.cpu_count(logical=True) or "?"
            cpu_pct = _psu_si.cpu_percent(interval=0.3)
            lines_si.append(f"Cores: {cpu_p} physical / {cpu_l} logical")
            lines_si.append(f"Overall usage: {cpu_pct:.1f}%")
            freq = _psu_si.cpu_freq()
            if freq:
                lines_si.append(
                    f"Frequency: {freq.current:.0f} MHz  (max {freq.max:.0f} MHz)"
                )
            per_core = _psu_si.cpu_percent(percpu=True)
            if per_core:
                lines_si.append(
                    "Per-core: " + "  ".join(f"C{i}:{p:.0f}%" for i, p in enumerate(per_core))
                )
        except ImportError:
            # Windows fallback: WMIC
            if _plat_si.system() == "Windows":
                try:
                    _r = _sp_si.run(
                        ["wmic", "cpu", "get",
                         "name,numberofcores,numberoflogicalprocessors,loadpercentage",
                         "/format:list"],
                        capture_output=True, text=True, timeout=10,
                    )
                    for _l in _r.stdout.splitlines():
                        _l = _l.strip()
                        if _l and "=" in _l:
                            lines_si.append(_l)
                except Exception as _exc:
                    lines_si.append(f"CPU info unavailable: {_exc}")
            else:
                lines_si.append("Install psutil for live CPU stats.")
        lines_si.append("")

    # ── Memory / RAM ───────────────────────────────────────────────────────
    if _want("memory"):
        lines_si.append("=== Memory (RAM) ===")
        try:
            import psutil as _psu_si  # noqa: PLC0415
            vm = _psu_si.virtual_memory()
            lines_si.append(
                f"Total:     {vm.total / 2**30:.2f} GB"
            )
            lines_si.append(
                f"Used:      {vm.used / 2**30:.2f} GB  ({vm.percent:.1f}%)"
            )
            lines_si.append(
                f"Available: {vm.available / 2**30:.2f} GB"
            )
            swap = _psu_si.swap_memory()
            if swap.total > 0:
                lines_si.append(
                    f"Swap:      {swap.used / 2**30:.2f} / {swap.total / 2**30:.2f} GB"
                    f"  ({swap.percent:.1f}%)"
                )
        except ImportError:
            if _plat_si.system() == "Windows":
                try:
                    _r = _sp_si.run(
                        ["wmic", "OS", "get",
                         "totalvisiblememorysize,freephysicalmemory", "/format:list"],
                        capture_output=True, text=True, timeout=10,
                    )
                    _mem: dict[str, int] = {}
                    for _l in _r.stdout.splitlines():
                        if "=" in _l:
                            k, _, v = _l.strip().partition("=")
                            try:
                                _mem[k.strip()] = int(v.strip())
                            except ValueError:
                                pass
                    total_kb = _mem.get("TotalVisibleMemorySize", 0)
                    free_kb = _mem.get("FreePhysicalMemory", 0)
                    if total_kb:
                        used_kb = total_kb - free_kb
                        pct = used_kb / total_kb * 100
                        lines_si.append(f"Total:     {total_kb / 2**20:.2f} GB")
                        lines_si.append(f"Used:      {used_kb / 2**20:.2f} GB  ({pct:.1f}%)")
                        lines_si.append(f"Available: {free_kb / 2**20:.2f} GB")
                except Exception as _exc:
                    lines_si.append(f"Memory info unavailable: {_exc}")
            else:
                lines_si.append("Install psutil for live memory stats.")
        lines_si.append("")

    # ── GPU ────────────────────────────────────────────────────────────────
    if _want("gpu"):
        lines_si.append("=== GPU ===")
        try:
            _r = _sp_si.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total,memory.used,"
                    "memory.free,utilization.gpu,utilization.memory,temperature.gpu",
                    "--format=csv,noheader",
                ],
                capture_output=True, text=True, timeout=10,
            )
            if _r.returncode == 0 and _r.stdout.strip():
                _fields = ["Name", "Driver", "VRAM Total", "VRAM Used",
                           "VRAM Free", "GPU util", "Mem util", "Temp"]
                for _row in _r.stdout.strip().splitlines():
                    _vals = [v.strip() for v in _row.split(",")]
                    for _f, _v in zip(_fields, _vals):
                        lines_si.append(f"{_f + ':':12} {_v}")
            else:
                lines_si.append("No NVIDIA GPU detected (nvidia-smi not found or no CUDA GPU).")
        except FileNotFoundError:
            lines_si.append("No NVIDIA GPU detected (nvidia-smi not in PATH).")
        except Exception as _exc:
            lines_si.append(f"GPU info unavailable: {_exc}")
        lines_si.append("")

    # ── Disk drives ────────────────────────────────────────────────────────
    if _want("disk"):
        lines_si.append("=== Disk Drives ===")
        try:
            import psutil as _psu_si  # noqa: PLC0415
            for _part in _psu_si.disk_partitions(all=False):
                try:
                    _usage = _psu_si.disk_usage(_part.mountpoint)
                    _pct = _usage.used / _usage.total * 100 if _usage.total else 0
                    lines_si.append(
                        f"{_part.device} ({_part.fstype})  "
                        f"Used: {_usage.used / 2**30:.1f} GB / "
                        f"{_usage.total / 2**30:.1f} GB  ({_pct:.1f}%)  "
                        f"Free: {_usage.free / 2**30:.1f} GB"
                    )
                except (PermissionError, OSError):
                    lines_si.append(f"{_part.device}  (not accessible)")
        except ImportError:
            if _plat_si.system() == "Windows":
                try:
                    _r = _sp_si.run(
                        ["wmic", "logicaldisk", "get",
                         "caption,size,freespace,filesystem", "/format:list"],
                        capture_output=True, text=True, timeout=10,
                    )
                    _disk: dict[str, str] = {}
                    for _l in _r.stdout.splitlines():
                        _l = _l.strip()
                        if _l and "=" in _l:
                            k, _, v = _l.partition("=")
                            _disk[k.strip()] = v.strip()
                        elif not _l and _disk.get("Caption"):
                            total_b = int(_disk.get("Size", 0) or 0)
                            free_b = int(_disk.get("FreeSpace", 0) or 0)
                            used_b = total_b - free_b
                            pct = used_b / total_b * 100 if total_b else 0
                            fs = _disk.get("FileSystem", "")
                            cap = _disk.get("Caption", "?")
                            if total_b:
                                lines_si.append(
                                    f"{cap} ({fs})  "
                                    f"Used: {used_b / 2**30:.1f} GB / {total_b / 2**30:.1f} GB"
                                    f"  ({pct:.1f}%)  Free: {free_b / 2**30:.1f} GB"
                                )
                            _disk = {}
                except Exception as _exc:
                    lines_si.append(f"Disk info unavailable: {_exc}")
            else:
                lines_si.append("Install psutil for disk info.")
        lines_si.append("")

    # ── Top processes ──────────────────────────────────────────────────────
    if _want("processes"):
        lines_si.append("=== Top Processes (by CPU%) ===")
        try:
            import psutil as _psu_si  # noqa: PLC0415
            _procs = []
            for _proc in _psu_si.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
                try:
                    _procs.append(_proc.info)
                except (_psu_si.NoSuchProcess, _psu_si.AccessDenied):
                    pass
            # Warm up cpu_percent (first call always 0)
            import time as _time_si
            _time_si.sleep(0.3)
            _procs2 = []
            for _proc in _psu_si.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
                try:
                    _procs2.append(_proc.info)
                except (_psu_si.NoSuchProcess, _psu_si.AccessDenied):
                    pass
            _procs2.sort(key=lambda p: p.get("cpu_percent") or 0, reverse=True)
            lines_si.append(f"{'PID':>6}  {'CPU%':>6}  {'MEM%':>6}  Name")
            lines_si.append("-" * 50)
            for _p in _procs2[:15]:
                lines_si.append(
                    f"{_p.get('pid', 0):>6}  "
                    f"{(_p.get('cpu_percent') or 0):>5.1f}%  "
                    f"{(_p.get('memory_percent') or 0):>5.1f}%  "
                    f"{_p.get('name', '?')}"
                )
        except ImportError:
            lines_si.append("Install psutil for process list.")
        lines_si.append("")

    # ── Network ────────────────────────────────────────────────────────────
    if _want("network"):
        lines_si.append("=== Network ===")
        try:
            import psutil as _psu_si  # noqa: PLC0415
            _io = _psu_si.net_io_counters()
            if _io:
                lines_si.append(
                    f"Bytes sent:     {_io.bytes_sent / 2**20:.1f} MB"
                )
                lines_si.append(
                    f"Bytes received: {_io.bytes_recv / 2**20:.1f} MB"
                )
            _conns = _psu_si.net_connections(kind="inet")
            established = sum(1 for c in _conns if c.status == "ESTABLISHED")
            lines_si.append(f"Active connections: {established}")
            # Per-interface
            _ifaces = _psu_si.net_io_counters(pernic=True)
            if _ifaces:
                lines_si.append("Interfaces:")
                for _iface, _stats in sorted(_ifaces.items()):
                    if _stats.bytes_sent + _stats.bytes_recv > 0:
                        lines_si.append(
                            f"  {_iface}: "
                            f"↑ {_stats.bytes_sent / 2**20:.1f} MB  "
                            f"↓ {_stats.bytes_recv / 2**20:.1f} MB"
                        )
        except ImportError:
            lines_si.append("Install psutil for network stats.")
        except Exception as _exc:
            lines_si.append(f"Network info unavailable: {_exc}")
        lines_si.append("")

    return "\n".join(lines_si).rstrip()


def _tool_web_search(
    args: dict,
    store: "VaultStore",
    top_n: int,
    collected_chunks: list,
    session_ctx: "dict | None",
) -> str:
    ctx = session_ctx or {}
    settings = ctx.get("settings")
    provider = (
        (settings.web_search.provider if settings is not None else "duckduckgo")
    ).strip().lower()

    if not provider:
        return (
            "Web search is disabled. "
            "Set [web_search] provider = \"duckduckgo\" in egovault.toml to enable it."
        )

    query = (args.get("query") or "").strip()
    if not query:
        return "Error: query parameter is required for web_search."

    max_r = min(20, int(args.get("max_results") or (settings.web_search.max_results if settings else 5)))

    if provider == "duckduckgo":
        try:
            try:
                from ddgs import DDGS as _DDGS  # noqa: PLC0415
            except ImportError:
                from duckduckgo_search import DDGS as _DDGS  # noqa: PLC0415
            results = list(_DDGS().text(query, max_results=max_r))
            if not results:
                return f"No web results found for: {query}"
            lines = [f"Web search results for: {query}\n"]
            for i, r in enumerate(results, 1):
                title = r.get("title", "").strip()
                link = r.get("href", "").strip()
                snippet = r.get("body", "").strip()
                lines.append(f"{i}. **{title}**\n   {link}\n   {snippet}")
            return "\n\n".join(lines)
        except ImportError:
            return (
                "Web search unavailable: ddgs is not installed. "
                "Run: pip install ddgs"
            )
        except Exception as exc:  # noqa: BLE001
            return f"Web search failed: {exc}"

    elif provider == "searxng":
        import urllib.parse as _up
        import urllib.error as _ue

        primary = (
            (settings.web_search.searxng_url if settings is not None else "").strip().rstrip("/")
        )
        if not primary:
            return (
                "Web search is not configured. "
                "Set [web_search] searxng_url in egovault.toml when using provider = \"searxng\"."
            )

        categories = settings.web_search.categories if settings else "general"
        fallbacks = [f.strip().rstrip("/") for f in (
            (settings.web_search.fallback_urls if settings is not None else []) or []
        ) if f.strip()]
        candidates = [primary] + [u for u in fallbacks if u != primary]

        _ws_headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) "
                "Gecko/20100101 Firefox/125.0"
            ),
            "Accept": "application/json, text/html, */*",
            "Accept-Language": "en-US,en;q=0.9",
        }
        params = _up.urlencode({"q": query, "format": "json", "categories": categories})

        last_error = ""
        for base_url in candidates:
            req = urllib.request.Request(
                f"{base_url}/search?{params}",
                headers=_ws_headers,
                method="GET",
            )
            try:
                with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
                    data = json.loads(resp.read().decode("utf-8"))
                results = data.get("results", [])[:max_r]
                if not results:
                    return f"No web results found for: {query}"
                lines = [f"Web search results for: {query}\n"]
                for i, r in enumerate(results, 1):
                    title = r.get("title", "").strip()
                    link = r.get("url", "").strip()
                    snippet = r.get("content", "").strip()
                    lines.append(f"{i}. **{title}**\n   {link}\n   {snippet}")
                return "\n\n".join(lines)
            except _ue.HTTPError as exc:  # noqa: BLE001
                last_error = f"HTTP {exc.code} from {base_url}"
                if exc.code in (429, 403):
                    continue
                return f"Web search failed: {last_error}"
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                continue

        return (
            f"Web search unavailable: all configured SearXNG instances are busy "
            f"({last_error}). Try again in a moment."
        )

    else:
        return (
            f"Web search: unknown provider '{provider}'. "
            "Set provider = \"duckduckgo\" or \"searxng\" in [web_search] egovault.toml."
        )


def _tool_schedule_task(
    args: dict,
    store: "VaultStore",
    top_n: int,
    collected_chunks: list,
    session_ctx: "dict | None",
) -> str:
    prompt_arg = (args.get("prompt") or "").strip()
    when_arg = (args.get("when") or "").strip()
    if not prompt_arg:
        return "Error: prompt parameter is required."
    if not when_arg:
        return "Error: when parameter is required (e.g. 'in 5min', 'every hour')."

    ctx = session_ctx or {}
    settings = ctx.get("settings")
    if settings is None:
        return "Error: no settings context — cannot access scheduler."

    from egovault.utils.scheduler import (
        parse_schedule_expression,
        format_next_run,
        format_interval,
        Scheduler,
        make_executor,
    )

    parsed = parse_schedule_expression(when_arg)
    if parsed is None:
        return (
            f"Could not parse time expression: '{when_arg}'. "
            "Valid examples: 'in 5min', 'in 1 hour', 'every 30min', "
            "'every day at 19:05', 'every morning'."
        )
    next_run, interval_seconds = parsed

    # Determine the scheduler command: /gmail-sync and /scan are native
    # commands; everything else is wrapped as a chat: prompt.
    lower_prompt = prompt_arg.lower().strip()
    if lower_prompt.startswith("/gmail-sync"):
        cmd = "/gmail-sync"
    elif lower_prompt.startswith("/scan"):
        cmd = prompt_arg.strip()
    else:
        cmd = f"chat: {prompt_arg}"

    data_dir = Path(settings.vault_db).parent

    # Prefer the already-running Scheduler from session_ctx so the task is
    # immediately live in its tick loop and shows up in /schedule --list.
    # Fall back to creating a new one (task is persisted to disk; the
    # running instance will pick it up on next /schedule --list refresh).
    scheduler = ctx.get("scheduler")
    if scheduler is None:
        scheduler = Scheduler(data_dir)
        notice_q = ctx.get("notice_queue") or __import__("queue").Queue()
        scheduler.start(
            executor=make_executor(settings.vault_db, settings),
            notice_queue=notice_q,
        )
    task = scheduler.add_task(
        name=f"{cmd} ({when_arg.strip()})",
        command=cmd,
        next_run=next_run,
        interval_seconds=interval_seconds,
    )
    nr = format_next_run(next_run)
    kind = format_interval(interval_seconds)
    return (
        f"TASK_SCHEDULED: {task.id}\n"
        f"Prompt: {prompt_arg}\n"
        f"Next run: {nr}  ({kind})\n"
        f"Cancel with: /schedule --cancel {task.id}"
    )


def _tool_send_email(
    args: dict,
    store: "VaultStore",
    top_n: int,
    collected_chunks: list,
    session_ctx: "dict | None",
) -> str:
    to_arg = (args.get("to") or "").strip()
    subject_arg = (args.get("subject") or "").strip()
    body_arg = (args.get("body") or "").strip()
    cc_arg = (args.get("cc") or "").strip()
    bcc_arg = (args.get("bcc") or "").strip()
    if not to_arg:
        return "Error: 'to' parameter is required."
    if not subject_arg:
        return "Error: 'subject' parameter is required."
    if not body_arg:
        return "Error: 'body' parameter is required."

    ctx = session_ctx or {}
    settings = ctx.get("settings")
    if settings is None:
        return "Error: no settings context — cannot connect to Gmail."
    data_dir = Path(settings.vault_db).parent
    from egovault.utils.gmail_imap import load_credentials as _lc
    imap_creds = _lc(data_dir)
    if imap_creds is None:
        return (
            "Gmail is not connected. "
            "Ask the user to run /gmail-auth to set up their App Password first."
        )
    gmail_address, app_password = imap_creds

    import smtplib
    from email.message import EmailMessage

    msg = EmailMessage()
    msg["From"] = gmail_address
    msg["To"] = to_arg
    msg["Subject"] = subject_arg
    if cc_arg:
        msg["Cc"] = cc_arg
    if bcc_arg:
        msg["Bcc"] = bcc_arg
    msg.set_content(body_arg)

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.ehlo()
            smtp.login(gmail_address, app_password)
            smtp.send_message(msg)
    except smtplib.SMTPAuthenticationError:
        return (
            "Authentication failed. The App Password may have been revoked. "
            "Run /gmail-auth to re-enter credentials."
        )
    except Exception as exc:  # noqa: BLE001
        return f"Failed to send email: {exc}"

    recipients = to_arg
    if cc_arg:
        recipients += f", CC: {cc_arg}"
    if bcc_arg:
        recipients += f", BCC: {bcc_arg}"
    return f"EMAIL_SENT: Message '{subject_arg}' sent to {recipients}."


def _tool_launch_frontend(
    args: dict,
    store: "VaultStore",
    top_n: int,
    collected_chunks: list,
    session_ctx: "dict | None",
) -> str:
    import subprocess as _subprocess

    frontend = (args.get("frontend") or "").strip().lower()
    valid = {"web", "telegram", "mcp", "chat"}
    if not frontend:
        return "Error: 'frontend' parameter is required."
    if frontend not in valid:
        return f"Error: unknown frontend '{frontend}'. Valid values: {', '.join(sorted(valid))}."

    # Resolve the ego entry-point: prefer the Scripts/ego(.cmd) sibling of the
    # current Python executable; fall back to  python -m egovault.
    import sys as _sys
    ego_name = "ego.cmd" if _sys.platform == "win32" else "ego"
    ego_exe = Path(_sys.executable).with_name(ego_name)
    if ego_exe.exists():
        cmd_base: list[str] = [str(ego_exe), frontend]
    else:
        cmd_base = [_sys.executable, "-m", "egovault", frontend]

    extra: list[str] = []
    if frontend == "web":
        port = args.get("port")
        if port:
            extra += ["--port", str(int(port))]

        # Guard against launching a second web instance when one is already
        # running (e.g. started by the CLI before the TUI was opened).
        import socket as _socket
        _web_port = int(args.get("port", 8501))
        with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as _s:
            _already_up = _s.connect_ex(("127.0.0.1", _web_port)) == 0
        if _already_up:
            _wan_url = os.environ.get("EGOVAULT_WAN_URL", "")
            _url = f"http://localhost:{_web_port}"
            _label = _url + (f" (WAN: {_wan_url})" if _wan_url else "")
            return f"FRONTEND_LAUNCHED:web — web interface is already running at {_label}"

    try:
        if _sys.platform == "win32":
            proc = _subprocess.Popen(
                cmd_base + extra,
                creationflags=_subprocess.CREATE_NEW_CONSOLE,
                close_fds=True,
            )
        else:
            proc = _subprocess.Popen(
                cmd_base + extra,
                start_new_session=True,
                close_fds=True,
            )
    except Exception as exc:  # noqa: BLE001
        return f"Error launching {frontend}: {exc}"

    if session_ctx is not None:
        session_ctx.setdefault("launched_frontends", {})[frontend] = proc.pid

    _wan_url = os.environ.get("EGOVAULT_WAN_URL", "")
    _web_url = f"http://localhost:{args.get('port', 8501)}"
    _web_label = f"Streamlit web UI at {_web_url}" + (f" (WAN: {_wan_url})" if _wan_url else "")
    _labels = {
        "web": _web_label,
        "telegram": "Telegram bot",
        "mcp": "MCP stdio server (new window)",
        "chat": "TUI (new terminal window)",
    }
    return f"FRONTEND_LAUNCHED:{frontend} (pid={proc.pid}) — {_labels.get(frontend, frontend)}"


def _answer_needs_retry(answer: str) -> bool:
    """Return True when the answer looks incomplete or evasive despite vault data being present.

    Used to trigger one extra reasoning pass so the model can correct itself.
    """
    if not answer or len(answer.strip()) < 40:
        return True
    lower = answer.lower()
    _EVASIVE = (
        "i don't have", "i do not have", "i cannot find", "i can't find",
        "no information", "not available", "unable to find", "could not find",
        "couldn't find", "can't find", "don't have", "doesn't have",
        "i'm unable", "i am unable", "no record", "no data",
        "i couldn't", "i don't know", "i have no",
    )
    return any(p in lower for p in _EVASIVE)


def _call_llm_agent(
    initial_messages: list[dict],
    store: "VaultStore",
    top_n: int,
    llm_kwargs: dict,
    console: "Console",
    session_ctx: dict | None = None,
    max_iterations: int = 12,
    progress_cb: "Callable[[str], None] | None" = None,
) -> tuple[str, dict, list]:
    """Run the agentic tool-calling loop.

    Sends the initial messages with tool definitions, executes any tool calls
    the model requests, and loops until the model produces a final text answer.

    Returns (answer_text, last_raw_llm_response, collected_chunks).
    Raises on HTTP/network errors so the caller can fall back.
    """
    messages = list(initial_messages)
    collected_chunks: list = []
    last_data: dict = {}
    base_url = llm_kwargs["base_url"].rstrip("/")
    model = llm_kwargs["model"]
    timeout = llm_kwargs["timeout"]
    api_key = llm_kwargs.get("api_key", "")

    url = f"{base_url}/v1/chat/completions"
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Resolve context-limit setting once (0 = disabled).
    _max_ctx_tokens: int = 0
    if session_ctx:
        _settings_obj = session_ctx.get("settings")
        if _settings_obj is not None:
            _max_ctx_tokens = getattr(getattr(_settings_obj, "llm", None), "max_ctx_tokens", 0) or 0

    from egovault.utils.memory_processors import ToolCallFilter, TokenLimiter  # noqa: PLC0415

    for iteration in range(max_iterations):
        # Trim old tool results before each LLM call to prevent context overflow.
        messages = ToolCallFilter(keep_recent=3).process(messages)
        messages = TokenLimiter(max_tokens=_max_ctx_tokens, keep_recent=5).process(messages)

        body: dict = {
            "model": model,
            "messages": messages,
            "stream": False,
            "tools": _VAULT_TOOLS,
        }
        payload = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            last_data = json.loads(resp.read().decode("utf-8"))

        msg = last_data.get("choices", [{}])[0].get("message", {})
        tool_calls = msg.get("tool_calls")

        if not tool_calls:
            # Auto-search guardrail: if the model skipped all tools on the first
            # iteration and nothing has been retrieved yet, extract keywords from
            # the user's query and run search_vault automatically.  This prevents
            # generic "I don't have a tool for that" responses when the vault has
            # relevant data.
            #
            # Skip the guardrail for queries that clearly map to a non-vault tool
            # (screenshot, window inspection, web search, etc.).  Running an
            # auto-search against those queries injects unrelated vault records
            # and makes the LLM narrate the wrong content.
            _NON_VAULT_PHRASES = (
                "screenshot", "screen shot", "screen capture", "capture screen",
                "take screen", "take a screen",
                "inspect window", "inspect screen", "inspect the screen",
                "inspect my screen", "inspect my pc", "inspect pc",
                "inspect my computer", "inspect the computer",
                "inspect my display", "inspect display",
                "open window", "open apps", "open programs",
                "what windows", "what apps", "what programs",
                "running apps", "running programs", "running windows",
                "what's open", "whats open", "what is open",
                "what's running", "whats running", "what is running",
                "show windows", "list windows", "list apps", "show apps",
                "inspect element", "ui element", "ui tree", "window element",
                "web search", "search the web", "look up online", "search online",
                # Email sending — must call send_email, not search vault
                "send an email", "send email", "send a email",
                "compose an email", "compose email", "write an email",
                "email to ", "mail to ", "write to ",
                "draft an email", "draft email",
                # Frontend launching — must call launch_frontend, not search vault
                "start the web", "start web", "open the web", "open web",
                "launch the web", "launch web", "start browser", "open browser",
                "start the telegram", "start telegram", "launch telegram",
                "open telegram", "start the bot", "launch the bot",
                "start mcp", "launch mcp", "open mcp",
                "start the interface", "launch the interface", "open the interface",
                "launch frontend", "start frontend",
            )
            user_q = (initial_messages[-1].get("content", "") if initial_messages else "")
            _user_q_lower = user_q.lower()
            _is_non_vault = any(p in _user_q_lower for p in _NON_VAULT_PHRASES)
            # Component-word detection: "inspect" + any system-context word
            if not _is_non_vault and "inspect" in _user_q_lower:
                _SYS_OBJECTS = ("screen", "window", "windows", "desktop", "display",
                                "monitor", "pc", "computer", "running", "apps", "programs")
                _is_non_vault = any(w in _user_q_lower for w in _SYS_OBJECTS)

            _current_answer = msg.get("content") or ""

            if not _is_non_vault and iteration == 0 and not collected_chunks:
                # Only inject vault context when the LLM answer is actually evasive.
                # If the LLM already gave a substantive (even if wrong) answer, don't
                # pile unrelated vault records on top of it.
                if _answer_needs_retry(_current_answer):
                    from egovault.chat.rag import _sanitize_query as _sq  # noqa: PLC0415
                    auto_terms = _sq(user_q)
                    if auto_terms:
                        comma_q = ", ".join(t for t in auto_terms.split(" OR ") if t.strip())
                        if progress_cb:
                            progress_cb(f"⚙ search_vault({comma_q})")
                        else:
                            console.print(f"  [dim]⚙ search_vault({comma_q})[/dim]")
                        auto_result = _execute_tool(
                            "search_vault", {"query": comma_q},
                            store, top_n, collected_chunks, session_ctx,
                        )
                        if not auto_result.startswith("No results") and not auto_result.startswith("Error"):
                            # Keep the assistant's (likely generic) turn, then re-inject
                            # context as a new user message and continue the loop.
                            messages.append({"role": "assistant", "content": _current_answer})
                            messages.append({
                                "role": "user",
                                "content": (
                                    f"[VAULT CONTEXT — found the following relevant records. "
                                    f"Use them to answer the original question directly.]\n\n{auto_result}"
                                ),
                            })
                            continue
            elif _is_non_vault and iteration == 0:
                # LLM skipped the tool for a non-vault request on the first pass.
                _send_intent = any(p in _user_q_lower for p in (
                    "send an email", "send email", "send a email",
                    "compose an email", "compose email", "write an email",
                    "email to ", "mail to ", "draft an email", "draft email",
                ))
                _launch_intent = any(p in _user_q_lower for p in (
                    "start the web", "start web", "open the web", "open web",
                    "launch the web", "launch web", "start browser", "open browser",
                    "start the telegram", "start telegram", "launch telegram",
                    "open telegram", "start the bot", "launch the bot",
                    "start mcp", "launch mcp", "open mcp",
                    "start the interface", "launch the interface", "open the interface",
                    "launch frontend", "start frontend",
                ))
                if _send_intent:
                    # Try to extract to/subject/body from the user query so we
                    # can auto-dispatch send_email without another LLM round-trip
                    # (Gemma often hallucinates "I sent it" then returns empty on nudge).
                    _to_m = re.search(
                        r'(?:to|To:)\s+([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})',
                        user_q,
                    )
                    _subj_m = re.search(
                        r'subject\s*[:\'""]?\s*["\']?([^"\']+?)["\']?\s+(?:and|with|body)',
                        user_q, re.I,
                    )
                    _body_m = re.search(
                        r'body\s*[:\'""]?\s*["\'](.+?)["\']',
                        user_q, re.I | re.S,
                    )
                    if _to_m and _subj_m and _body_m:
                        _email_args = {
                            "to": _to_m.group(1).strip(),
                            "subject": _subj_m.group(1).strip(),
                            "body": _body_m.group(1).strip(),
                        }
                        label = f"⚙ send_email(to={_email_args['to']}, subject={_email_args['subject'][:40]!r})"
                        if progress_cb is not None:
                            progress_cb(label)
                        else:
                            console.print(f"  [dim]{label}[/dim]")
                        auto_result = _execute_tool(
                            "send_email", _email_args,
                            store, top_n, collected_chunks,
                            {**(session_ctx or {}), "_progress_cb": progress_cb},
                        )
                        messages.append({"role": "assistant", "content": _current_answer})
                        messages.append({
                            "role": "tool",
                            "content": auto_result,
                            "tool_call_id": "auto_send_email",
                        })
                        continue
                    else:
                        # Can't auto-parse — nudge model to call the tool explicitly.
                        messages.append({"role": "assistant", "content": _current_answer})
                        messages.append({
                            "role": "user",
                            "content": (
                                "You MUST call the send_email tool to send this email. "
                                "Call the tool now with the to, subject, and body parameters."
                            ),
                        })
                        continue
                elif _launch_intent:
                    # Auto-dispatch launch_frontend when the LLM skips the tool.
                    # Parse which frontend was requested and optional port.
                    _lf_frontend = "web"
                    if any(p in _user_q_lower for p in ("telegram", "bot")):
                        _lf_frontend = "telegram"
                    elif any(p in _user_q_lower for p in ("mcp",)):
                        _lf_frontend = "mcp"
                    elif any(p in _user_q_lower for p in ("chat", "terminal", "tui", "repl")):
                        _lf_frontend = "chat"
                    _lf_port: dict = {}
                    _port_m = re.search(r'\bport\s+(\d{2,5})\b', user_q, re.I)
                    if _port_m:
                        _lf_port["port"] = int(_port_m.group(1))
                    _lf_args = {"frontend": _lf_frontend, **_lf_port}
                    _lf_label = (
                        f"⚙ launch_frontend(frontend={_lf_frontend!r}"
                        + (f", port={_lf_port['port']}" if _lf_port else "")
                        + ")"
                    )
                    if progress_cb is not None:
                        progress_cb(_lf_label)
                    else:
                        console.print(f"  [dim]{_lf_label}[/dim]")
                    auto_result = _execute_tool(
                        "launch_frontend", _lf_args,
                        store, top_n, collected_chunks,
                        {**(session_ctx or {}), "_progress_cb": progress_cb},
                    )
                    messages.append({"role": "assistant", "content": _current_answer})
                    messages.append({
                        "role": "tool",
                        "content": auto_result,
                        "tool_call_id": "auto_launch_frontend",
                    })
                    continue
                else:
                    messages.append({"role": "assistant", "content": _current_answer})
                    messages.append({
                        "role": "user",
                        "content": (
                            "You have the right tools available to fulfil this request directly. "
                            "Please call the appropriate tool now instead of explaining what you would do."
                        ),
                    })
                    continue
            # Retry once when the answer is evasive but vault data is available.
            answer = msg.get("content") or ""
            if collected_chunks and _answer_needs_retry(answer) and iteration < max_iterations - 1:
                messages.append({"role": "assistant", "content": answer})
                messages.append({
                    "role": "user",
                    "content": (
                        "Your previous answer seems incomplete or inaccurate. "
                        "Re-read the vault records above and give a specific, accurate "
                        "answer — cite exact dates, names, and details as they appear "
                        "in the records."
                    ),
                })
                continue
            # Vault fallback: if no vault chunks and answer is evasive, force a
            # vault search regardless of what tools were called before.
            # Catches: web_search called first → no results → evasive LLM answer.
            if not collected_chunks and not _is_non_vault and _answer_needs_retry(answer) and iteration < max_iterations - 1:
                from egovault.chat.rag import _sanitize_query as _sq  # noqa: PLC0415
                auto_terms = _sq(user_q)
                if auto_terms:
                    comma_q = ", ".join(t for t in auto_terms.split(" OR ") if t.strip())
                    if progress_cb is not None:
                        progress_cb(f"⚙ search_vault({comma_q})")
                    else:
                        console.print(f"  [dim]⚙ search_vault({comma_q})[/dim]")
                    auto_result = _execute_tool(
                        "search_vault", {"query": comma_q},
                        store, top_n, collected_chunks, session_ctx,
                    )
                    if not auto_result.startswith("No results") and not auto_result.startswith("Error"):
                        messages.append({"role": "assistant", "content": answer})
                        messages.append({
                            "role": "user",
                            "content": (
                                f"[VAULT CONTEXT — found the following relevant records. "
                                f"Use them to answer the original question directly.]\n\n{auto_result}"
                            ),
                        })
                        continue
            return answer, last_data, collected_chunks

        # Append the assistant turn that requested the tool calls.
        messages.append({
            "role": "assistant",
            "content": msg.get("content") or "",
            "tool_calls": tool_calls,
        })

        # Execute each tool call and feed results back.
        _vault_data_in_iteration = False
        _force_outer_continue = False
        for tc in tool_calls:
            fn = tc.get("function", {})
            tool_name = fn.get("name", "")
            raw_args = fn.get("arguments", {})
            if isinstance(raw_args, str):
                try:
                    raw_args = json.loads(raw_args)
                except (json.JSONDecodeError, ValueError):
                    raw_args = {}

            # Show tool invocation to user for transparency
            _tool_labels = {
                "search_vault": lambda a: (
                    f"⚙ search_vault({a.get('query','')}"
                    + (f", platform={a['platform']}" if a.get("platform") else "")
                    + (f", since={a['since']}" if a.get("since") else "")
                    + (f", until={a['until']}" if a.get("until") else "")
                    + ")"
                ),
                "count_records": lambda a: (
                    "⚙ count_records("
                    + (f"platform={a['platform']}, " if a.get("platform") else "")
                    + (f"since={a['since']}, " if a.get("since") else "")
                    + (f"until={a['until']}" if a.get("until") else "")
                    + ")"
                ),
                "get_vault_stats": lambda a: "⚙ get_vault_stats()",
                "read_file": lambda a: f"⚙ read_file({a.get('path','')})",
                "list_directory": lambda a: f"⚙ list_directory({a.get('path','')})",
                "scan_folder": lambda a: f"⚙ scan_folder({a.get('path','')})",
                "get_sources": lambda a: "⚙ get_sources()",
                "get_status": lambda a: "⚙ get_status()",
                "get_profile": lambda a: "⚙ get_profile()",
                "write_file": lambda a: f"⚙ write_file({a.get('path','')})",
                "fetch_attachment": lambda a: f"⚙ fetch_attachment({a.get('attachment_name','')})",
                "get_record": lambda a: f"⚙ get_record({a.get('record_id','')})",
                "gmail_sync": lambda a: (
                    "⚙ gmail_sync(" + (f"since={a['since']}" if a.get("since") else "fetching emails…") + ")"
                ),
                "open_file": lambda a: (
                    f"⚙ download({a.get('path','')})"
                    if str(a.get("path", "")).startswith(("http://", "https://"))
                    else f"⚙ open_file({a.get('path','')})"
                ),
                "take_screenshot": lambda a: (
                    "⚙ take_screenshot("
                    + (f"region={a['region']}" if a.get("region") else "full screen")
                    + ")"
                ),
                "inspect_windows": lambda a: (
                    "⚙ inspect_windows("
                    + (f"filter={a['filter']}" if a.get("filter") else "all windows")
                    + ")"
                ),
                "inspect_window": lambda a: (
                    f"⚙ inspect_window(title={a.get('title_filter', '')})"
                ),
                "system_info": lambda a: (
                    "⚙ system_info("
                    + (", ".join(a["sections"]) if a.get("sections") else "all")
                    + ")"
                ),
                "web_search": lambda a: f"⚙ web_search({a.get('query','')})",
                "schedule_task": lambda a: (
                    f"⚙ schedule_task({a.get('prompt','')[:50]!r}, when={a.get('when','')})"
                ),
                "send_email": lambda a: (
                    f"⚙ send_email(to={a.get('to','')}, subject={a.get('subject','')[:40]!r})"
                ),
                "launch_frontend": lambda a: (
                    f"⚙ launch_frontend(frontend={a.get('frontend','')!r}"
                    + (f", port={a['port']}" if a.get("port") else "")
                    + ")"
                ),
            }
            if tool_name in _tool_labels:
                label = _tool_labels[tool_name](raw_args)
                if progress_cb is not None:
                    progress_cb(label)
                else:
                    console.print(f"  [dim]{label}[/dim]")

            # Destructive-action confirmation gate.
            # On first encounter, ask the user to confirm; on the second call
            # (same tool_name within this turn) we allow it through.
            _ctx_for_confirm = session_ctx or {}
            if tool_name in _REQUIRES_CONFIRMATION:
                _confirmed = _ctx_for_confirm.get("_confirmed_tool") == tool_name
                if not _confirmed:
                    # Record that we've asked; the next LLM reply will re-invoke.
                    if isinstance(session_ctx, dict):
                        session_ctx["_confirmed_tool"] = tool_name
                    _confirm_msg = (
                        f"CONFIRMATION_REQUIRED: You are about to execute '{tool_name}'. "
                        f"Please confirm with the user before proceeding."
                    )
                    if progress_cb is not None:
                        progress_cb(f"⚠ confirmation_needed:{tool_name}")
                    messages.append({"role": "tool", "content": _confirm_msg})
                    messages.append({
                        "role": "user",
                        "content": "Please confirm: should I proceed?",
                    })
                    # Short-circuit: let the LLM ask the user for confirmation.
                    answer = (
                        f"I'm about to {_tool_labels.get(tool_name, lambda a: tool_name)(raw_args).lstrip('⚙ ')}. "
                        f"Shall I proceed? (Reply 'yes' or 'no'.)"
                    )
                    return answer, last_data, collected_chunks
                else:
                    # Clear the confirmation flag after use.
                    if isinstance(session_ctx, dict):
                        session_ctx.pop("_confirmed_tool", None)

            result = _execute_tool(
                tool_name, raw_args, store, top_n, collected_chunks,
                # Inject progress_cb so long-running tools (gmail_sync) can
                # emit live progress without polling the outer loop.
                {**(session_ctx or {}), "_progress_cb": progress_cb},
            )
            messages.append({"role": "tool", "content": result})
            if tool_name == "search_vault" and result.startswith("VAULT DATA"):
                _vault_data_in_iteration = True

            # Short-circuit: send_email — return answer directly so the LLM
            # doesn't produce an empty string after seeing EMAIL_SENT.
            if tool_name == "send_email":
                if result.startswith("EMAIL_SENT:"):
                    payload = result[len("EMAIL_SENT:"):].strip()
                    answer = f"✉ {payload}"
                else:
                    answer = result
                return answer, last_data, collected_chunks

            # Short-circuit: scan_folder — build and return answer directly from Python.            # Small LLMs consistently ignore the result or produce raw JSON when left to
            # narrate it themselves, so we bypass them entirely for this tool.
            if tool_name == "scan_folder" and result.startswith("SCAN COMPLETE"):
                new_m = re.search(r"New records added: (\d+)", result)
                skip_m = re.search(r"Already known.*?: (\d+)", result)
                total_m = re.search(r"Total files in folder: (\d+)", result)
                path_m = re.search(r"SCAN COMPLETE — (.+?)\n", result)
                added = int(new_m.group(1)) if new_m else 0
                already = int(skip_m.group(1)) if skip_m else 0
                total = int(total_m.group(1)) if total_m else (added + already)
                scanned_path = path_m.group(1).strip().rstrip(":") if path_m else "inbox"
                if added > 0:
                    answer = (
                        f"Scanned **{scanned_path}** — {added} new file(s) added to the vault "
                        f"({already} of {total} already indexed)."
                    )
                else:
                    answer = (
                        f"Inbox is already up to date — all {total} file(s) are already indexed, "
                        f"no new files found."
                    )
                return answer, last_data, collected_chunks

            # Short-circuit: gmail_sync — build and return the answer directly
            # from the raw GMAIL_SYNC_COMPLETE result, bypassing LLM narration.
            # Small LLMs consistently drop the numeric counts or add confusing
            # "I searched your vault" text when left to narrate the result.
            if tool_name == "gmail_sync":
                if result.startswith("GMAIL_SYNC_COMPLETE"):
                    new_m = re.search(r"(\d+) new email", result)
                    skip_m = re.search(r"(\d+) already in vault", result)
                    added = int(new_m.group(1)) if new_m else 0
                    skipped = int(skip_m.group(1)) if skip_m else 0
                    if added > 0:
                        answer = (
                            f"Gmail sync complete — **{added} new** email(s) added to the vault "
                            f"({skipped} already in vault)."
                        )
                    elif skipped > 0:
                        answer = (
                            f"Gmail sync complete — **0 new** emails, "
                            f"{skipped} already in vault."
                        )
                    else:
                        answer = "Gmail sync complete — no emails found in the current sync window."
                elif result.startswith("Gmail sync failed:"):
                    # Surface the raw error directly — don't let the LLM paraphrase it.
                    answer = result
                else:
                    answer = result
                return answer, last_data, collected_chunks

            # Short-circuit: inspect_windows — build human-readable summary; UI renders images.
            if tool_name == "inspect_windows" and result.startswith("WINDOWS_INSPECTED:"):
                count_m = re.search(r"WINDOWS_INSPECTED:(\d+)", result)
                count = int(count_m.group(1)) if count_m else 0
                # Extract window titles for the summary (lines starting with a digit)
                title_lines = [
                    line for line in result.splitlines()
                    if re.match(r"^\d+\.", line.strip())
                ]
                summary_lines = title_lines[:10]
                more = count - len(summary_lines)
                answer = f"Found **{count}** open window(s):"
                if summary_lines:
                    answer += "\n" + "\n".join(f"- {l.strip()}" for l in summary_lines)
                if more > 0:
                    answer += f"\n*…and {more} more.*"
                attached = (session_ctx or {}).get("saved_attachments", [])
                if attached:
                    answer += f"\n\n_{len(attached)} screenshot(s) attached._"
                return answer, last_data, collected_chunks

            # Short-circuit: take_screenshot — confirm the save; the UI renders the image.
            if tool_name == "take_screenshot" and result.startswith("SCREENSHOT_SAVED:"):
                path_m = re.search(r"SCREENSHOT_SAVED:(.+?)(?:\n|$)", result)
                saved_path = path_m.group(1).strip() if path_m else ""
                size_m = re.search(r"Size: (.+)", result)
                size_info = size_m.group(1) if size_m else ""
                answer = f"Screenshot saved ({size_info}) — displayed below."
                if saved_path and session_ctx is not None:
                    session_ctx["last_file"] = saved_path
                return answer, last_data, collected_chunks

            # Short-circuit: inspect_window — confirm element dump + screenshot.
            if tool_name == "inspect_window" and result.startswith("UI_ELEMENTS_SAVED:"):
                path_m = re.search(r"UI_ELEMENTS_SAVED:(.+?)(?:\n|$)", result)
                txt_path = path_m.group(1).strip() if path_m else ""
                win_m = re.search(r"Window: (.+?)(?:\n|$)", result)
                elem_m = re.search(r"Elements found: (\d+)", result)
                win_info = win_m.group(1).strip() if win_m else ""
                n_elems = elem_m.group(1) if elem_m else "?"
                attached = (session_ctx or {}).get("saved_attachments", [])
                answer = f"Inspected **{win_info}** — {n_elems} UI elements saved to `{txt_path}`."
                if attached:
                    answer += "\n\n_1 screenshot attached._"
                return answer, last_data, collected_chunks

            # Short-circuit: open_file URL download — confirm the saved path cleanly.
            if tool_name == "open_file" and result.startswith("FILE_DOWNLOADED:"):
                path_m = re.search(r"FILE_DOWNLOADED:(.+?)(?:\n|$)", result)
                saved_path = path_m.group(1).strip() if path_m else ""
                size_m = re.search(r"Size: (.+)", result)
                size_info = size_m.group(1) if size_m else ""
                answer = f"Downloaded to **{saved_path}**" + (f" ({size_info})" if size_info else "") + "."
                if saved_path and session_ctx is not None:
                    session_ctx["last_file"] = saved_path
                return answer, last_data, collected_chunks

            # Short-circuit: schedule_task — confirm what was scheduled and when.
            if tool_name == "schedule_task" and result.startswith("TASK_SCHEDULED:"):
                prompt_m = re.search(r"Prompt: (.+?)\n", result)
                nr_m = re.search(r"Next run: (.+?)\n", result)
                cancel_m = re.search(r"Cancel with: (.+)", result)
                sched_prompt = prompt_m.group(1).strip() if prompt_m else raw_args.get("prompt", "")
                nr_str = nr_m.group(1).strip() if nr_m else ""
                cancel_str = cancel_m.group(1).strip() if cancel_m else ""
                answer = f"Scheduled: **{sched_prompt}** — runs {nr_str}."
                if cancel_str:
                    answer += f"  (cancel: `{cancel_str}`)"
                return answer, last_data, collected_chunks

            # Short-circuit: write_file — confirm the save and stream a clean message.
            # Only short-circuit when the file has real content (more than a header stub)
            # AND at least one search has already run, so we don't skip the data-gathering step.
            if tool_name == "write_file" and result.startswith("FILE WRITTEN:"):
                content_written = str(raw_args.get("content", ""))
                has_real_data = len(content_written.splitlines()) > 2 or len(content_written) > 200
                if has_real_data:
                    path_m = re.search(r"FILE WRITTEN: (.+?)\n", result)
                    saved_path = path_m.group(1).strip() if path_m else raw_args.get("path", "")
                    if session_ctx is not None:
                        session_ctx["last_file"] = saved_path
                    answer = f"Saved to **{saved_path}**"
                    return answer, last_data, collected_chunks
                # else: fall through — let the LLM see the result and (hopefully) search first

            # Short-circuit: if search_vault already formatted the answer, return it directly
            # rather than waiting for the LLM to paraphrase (and potentially garble) it.
            # We only short-circuit when write_file has NOT also been requested — if the model
            # is expected to continue and create a file, we let it proceed.
            # Heuristic: if write_file appears in the tool list AND the model hasn't yet
            # called it, check if the original query looks like a save/export request.
            _save_words = ("create", "save", "export", "write", "generate",
                           "csv", "tsv", "json", "txt", "md", "html", "xlsx")
            _orig_query = (initial_messages[-1].get("content", "") if initial_messages else "").lower()
            _wants_output_file = any(w in _orig_query for w in _save_words)

            if tool_name == "search_vault" and result.startswith("SEARCH RESULTS") and not _wants_output_file:
                # Strip the LLM-directive header, keep just the lines
                lines = result.split("\n")
                # Find where the numbered list starts (line starting with "  1.")
                list_start = next((i for i, l in enumerate(lines) if l.strip().startswith("1.")), 2)
                # Stop before the FILE PATHS metadata section (not user-facing content)
                content_lines = []
                for l in lines[list_start:]:
                    if l.strip().startswith("FILE PATHS"):
                        break
                    content_lines.append(l)
                raw_list = [l.strip() for l in content_lines if l.strip()]

                # Parse out the text of each numbered item
                bare_lines: list[str] = []
                for l in raw_list:
                    m2 = re.match(r"^\d+\.\s+(.+)$", l)
                    bare_lines.append(m2.group(1) if m2 else l)

                formatted = "\n".join(f"{i + 1}. {ln}" for i, ln in enumerate(bare_lines))
                m = re.search(r"Found (\d+) matching", result)
                summary = f"\n\nFound {m.group(1)} mention(s)." if m else ""
                return formatted + summary, last_data, collected_chunks

            # When a file output is wanted: after search results land, inject a crisp
            # follow-up user turn that tells the model exactly what to do next.
            # A fresh user message is far more reliable than rewriting the tool result.
            if tool_name == "search_vault" and result.startswith("SEARCH RESULTS") and _wants_output_file:
                # Detect requested file type from original query
                _ft_m = re.search(r"\b(csv|tsv|json|txt|md|markdown)\b", _orig_query)
                _ft = _ft_m.group(1).lower() if _ft_m else "csv"
                if _ft == "markdown":
                    _ft = "md"
                messages.append({
                    "role": "user",
                    "content": (
                        f"Good. Now call write_file to save those results as a {_ft.upper()} file. "
                        f"Use the paths listed in the 'FILE PATHS' section of the search result as the path column. "
                        f"Format: header row (path,source) then one row per path. "
                        f"Save to the [OUTPUT DIRECTORY] shown in your system prompt."
                    ),
                })

        if _force_outer_continue:
            continue

        # Nudge the model to reason carefully before answering when vault records
        # were just returned.  Small models often latch onto the first date/name
        # they see; this extra turn tells them to re-read and extract accurately.
        if _vault_data_in_iteration:
            messages.append({
                "role": "user",
                "content": (
                    "[THINK] Read the vault records above carefully before answering. "
                    "Extract specific facts (dates, names, amounts, locations) exactly as "
                    "they appear in the records — do not confuse metadata fields such as "
                    "issue dates, file creation dates, or ticket purchase dates with the "
                    "actual content (e.g. departure dates, arrival dates). "
                    "Now answer the original question."
                ),
            })

    # Reached max iterations — return whatever the last answer was.
    return msg.get("content") or "", last_data, collected_chunks


def _handle_schedule(
    user_input: str,
    store: VaultStore,
    settings: Settings,
    scheduler: "Scheduler",
    bg_threads: "list[threading.Thread] | None" = None,
    bg_progress: "dict[str, BgProgress] | None" = None,
) -> None:
    """Execute /schedule commands from the chat REPL.

    Usage::

        /schedule --list                        — show all tasks
        /schedule --cancel <id>                 — remove a task
        /schedule /gmail-sync in 5min           — sync Gmail once in 5 minutes
        /schedule /gmail-sync every hour        — sync Gmail every hour
        /schedule /gmail-sync every day at 19:05
        /schedule /scan inbox every 30min
    """
    from egovault.utils.scheduler import (
        format_interval,
        format_next_run,
        parse_schedule_expression,
    )

    parts = user_input.split(None, 1)
    arg = parts[1].strip() if len(parts) > 1 else ""

    # ── /schedule --list ────────────────────────────────────────────────────
    if not arg or arg.lower() in ("--list", "-l", "list"):
        tasks = scheduler.list_tasks()
        # Single coordinator thread — check liveness once; display per-task
        # progress from bg_progress (populated for all pending tasks upfront).
        pipeline_alive = any(t.is_alive() for t in (bg_threads or []))
        _prg = bg_progress or {}

        if not tasks and not _prg:
            console.print(
                "[dim]No scheduled tasks.\n"
                "Examples:\n"
                "  /schedule /gmail-sync in 5min\n"
                "  /schedule /scan inbox every 30min[/dim]"
            )
            return

        if _prg:
            # Priority display order matches execution order.
            display_order = ["enrich", "context", "embed"]
            extra = [k for k in _prg if k not in display_order]
            console.print("[bold]Background pipeline:[/bold]")
            console.print(
                f"  [dim]order: enrich → context → embed  "
                f"({'running' if pipeline_alive else 'done'})[/dim]"
            )
            for label in display_order + extra:
                p = _prg.get(label)
                if p is None:
                    continue
                finished = p.done >= p.total if p.total > 0 else not pipeline_alive
                if not finished and pipeline_alive and p.done > 0:
                    elapsed = time.time() - p.started_at
                    rate = p.done / elapsed
                    remaining = p.total - p.done
                    eta_str = _format_eta(remaining / rate) if rate > 0 else "?"
                    pct = int(p.done / p.total * 100)
                    fail_str = f"  [red]{p.failed} failed[/red]" if p.failed else ""
                    console.print(
                        f"  [green]●[/green]  {label}  [bold]{pct}%[/bold]"
                        f"  [dim]({p.done}/{p.total})[/dim]"
                        f"  eta {eta_str}"
                        f"{fail_str}"
                    )
                elif not finished and pipeline_alive:
                    console.print(
                        f"  [yellow]○[/yellow]  {label}  [dim](0/{p.total})  queued[/dim]"
                    )
                else:
                    fail_str = f", {p.failed} failed" if p.failed else ""
                    console.print(f"  [dim]○  {label}  {p.done}/{p.total} done{fail_str}[/dim]")

        if tasks:
            if _prg:
                console.print()
            console.print("[bold]Scheduled tasks:[/bold]")
            for t in tasks:
                nr = format_next_run(t.next_run)
                interval_str = format_interval(t.interval_seconds)
                console.print(
                    f"  [cyan][{t.id}][/cyan]  {t.name}  "
                    f"[dim]next: {nr}  ({interval_str})[/dim]"
                )
        elif _prg:
            console.print()
            console.print(
                "[dim]No scheduled tasks. Examples:\n"
                "  /schedule /gmail-sync in 5min\n"
                "  /schedule /scan inbox every 30min[/dim]"
            )
        return

    # ── /schedule --cancel <id> ─────────────────────────────────────────────
    if arg.lower().startswith(("--cancel", "-c")):
        tokens = arg.split(None, 1)
        tid = tokens[1].strip() if len(tokens) > 1 else ""
        if not tid:
            console.print("[red]Usage: /schedule --cancel <task_id>[/red]")
            return
        if scheduler.cancel_task(tid):
            console.print(f"[green]✓[/green] Task [cyan]{tid}[/cyan] cancelled.")
        else:
            console.print(f"[yellow]Task [cyan]{tid}[/cyan] not found.[/yellow]")
        return

    # ── /schedule <command> <time_expression> ───────────────────────────────
    cmd = ""
    time_expr = ""

    if arg.lower().startswith("/gmail-sync"):
        cmd = "/gmail-sync"
        time_expr = arg[len("/gmail-sync"):].strip()
    elif arg.lower().startswith("/scan"):
        # /scan inbox every 30min  →  cmd="/scan inbox", time_expr="every 30min"
        scan_tokens = arg.split(None, 2)   # ["/scan", "<folder>", "<time>"]
        folder = scan_tokens[1] if len(scan_tokens) >= 2 else "inbox"
        cmd = f"/scan {folder}"
        time_expr = scan_tokens[2] if len(scan_tokens) >= 3 else ""
    elif arg.lower().startswith("chat:"):
        # Schedule an arbitrary chat prompt: "chat: <prompt containing time_expr>"
        # The time expression is embedded in the full text; find and extract it.
        full_text = arg[5:].strip()
        time_m_inner = _SCHEDULE_TIME_RE.search(full_text)
        if not time_m_inner:
            console.print(
                "[red]Missing time expression.[/red] Examples:\n"
                "  [dim]/schedule chat: search web for Barcelona news in 5min[/dim]\n"
                "  [dim]/schedule chat: save web search results to desktop every hour[/dim]"
            )
            return
        time_expr = time_m_inner.group(0)
        clean_prompt = _SCHEDULE_TIME_RE.sub("", full_text).strip().strip(",").strip()
        cmd = f"chat: {clean_prompt}"
    else:
        console.print(
            "[red]Unknown command to schedule.[/red] Supported: "
            "[cyan]/gmail-sync[/cyan], [cyan]/scan <folder>[/cyan], [cyan]chat: <prompt>[/cyan]"
        )
        return

    if not time_expr:
        console.print(
            "[red]Missing time expression.[/red] Examples:\n"
            "  [dim]/schedule /gmail-sync in 5min[/dim]\n"
            "  [dim]/schedule /gmail-sync every day at 19:05[/dim]\n"
            "  [dim]/schedule /scan inbox every 30min[/dim]"
        )
        return

    parsed = parse_schedule_expression(time_expr)
    if parsed is None:
        console.print(
            f"[red]Could not parse schedule:[/red] [italic]{time_expr}[/italic]\n"
            "[dim]Valid examples: 'in 5min', 'every hour', 'every day at 19:05'[/dim]"
        )
        return

    next_run, interval_seconds = parsed
    task = scheduler.add_task(
        name=f"{cmd} ({time_expr.strip()})",
        command=cmd,
        next_run=next_run,
        interval_seconds=interval_seconds,
    )
    nr = format_next_run(next_run)
    kind = format_interval(interval_seconds)
    console.print(
        f"[green]✓[/green] Scheduled [cyan]{cmd}[/cyan]  "
        f"next: [bold]{nr}[/bold]  ({kind})  "
        f"[dim](id: {task.id}  — cancel with /schedule --cancel {task.id})[/dim]"
    )


def _register_auto_schedules(scheduler: "Scheduler", settings: Settings) -> None:
    """Register adapter auto-refresh schedules from config if not already present.

    Called once at startup.  If the config specifies e.g.
    ``auto_refresh_gmail_minutes = 60`` and there is no existing ``/gmail-sync``
    task, a recurring task is added automatically.  Users can always override
    or cancel via ``/schedule``.
    """
    import time as _time

    sched_cfg = settings.scheduler
    existing_commands = {t.command for t in scheduler.list_tasks()}

    if sched_cfg.auto_refresh_inbox_minutes > 0 and "/scan inbox" not in existing_commands:
        interval = sched_cfg.auto_refresh_inbox_minutes * 60
        scheduler.add_task(
            name=f"/scan inbox (auto, every {sched_cfg.auto_refresh_inbox_minutes}min)",
            command="/scan inbox",
            next_run=_time.time() + interval,
            interval_seconds=interval,
        )

    if sched_cfg.auto_refresh_gmail_minutes > 0 and "/gmail-sync" not in existing_commands:
        interval = sched_cfg.auto_refresh_gmail_minutes * 60
        scheduler.add_task(
            name=f"/gmail-sync (auto, every {sched_cfg.auto_refresh_gmail_minutes}min)",
            command="/gmail-sync",
            next_run=_time.time() + interval,
            interval_seconds=interval,
        )

    tg_minutes = getattr(sched_cfg, "auto_refresh_telegram_minutes", 0)
    if tg_minutes > 0 and "/telegram-sync" not in existing_commands:
        interval = tg_minutes * 60
        scheduler.add_task(
            name=f"/telegram-sync (auto, every {tg_minutes}min)",
            command="/telegram-sync",
            next_run=_time.time() + interval,
            interval_seconds=interval,
        )


class BgProgress:
    """Lightweight progress tracker shared between a background worker and the REPL.

    All attributes are written from a single worker thread and read from the
    main thread.  CPython's GIL makes plain int reads/writes safe here.
    """

    __slots__ = ("label", "done", "total", "failed", "started_at")

    def __init__(self, label: str, total: int = 0) -> None:
        self.label = label
        self.done = 0
        self.total = total
        self.failed = 0
        self.started_at: float = time.time()


def _format_eta(seconds: float) -> str:
    """Format *seconds* as a short human-readable ETA string."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3_600:
        return f"{s // 60}min"
    h, m = divmod(s // 60, 60)
    return f"{h}h {m}min" if m else f"{h}h"


def _start_background_tasks(
    store: VaultStore,
    settings: Settings,
    notice_queue: "queue.Queue[str]",
    bg_progress: "dict[str, BgProgress]",
) -> list[threading.Thread]:
    """Spawn a single background pipeline thread that runs tasks sequentially.

    Tasks run one at a time in a fixed priority order to avoid LLM contention:

        1. enrich  — basic metadata; no dependencies; improves context quality
        2. context — contextual_body prefix (LLM); results used by embed
        3. embed   — dense vectors; uses contextual_body from step 2

    After all three finish the cycle repeats until no pending work remains.
    Progress for each task is tracked individually in *bg_progress* so the
    UI can display per-task counts even with a single coordinator thread.

    Posts human-readable status strings to *notice_queue* on completion so
    the REPL loop can print them between prompts without corrupting output.

    Returns a list containing the single coordinator Thread (empty if nothing
    to do).
    """
    llm = settings.llm
    embed_cfg = settings.embeddings

    # ── Determine which tasks have pending work ────────────────────────────
    pending_enrich = False
    try:
        pending_enrich = bool(store.get_unenriched_records(limit=1))
    except Exception:
        pass

    pending_ctx = False
    if getattr(embed_cfg, "contextual_enabled", False):
        try:
            pending_ctx = bool(store.get_uncontextualized_record_ids(limit=1))
        except Exception:
            pass

    pending_embed = False
    if embed_cfg.enabled:
        try:
            pending_embed = bool(store.get_unembedded_record_ids(embed_cfg.model, limit=1))
        except Exception:
            pass

    if not (pending_enrich or pending_ctx or pending_embed):
        return []

    # Pre-populate bg_progress totals for all pending tasks so the UI can
    # display them immediately (before the coordinator starts each one).
    if pending_enrich:
        bg_progress["enrich"] = BgProgress("enrich")
        try:
            bg_progress["enrich"].total = store.count_unenriched_records()
        except Exception:
            pass
    if pending_ctx:
        bg_progress["context"] = BgProgress("context")
        try:
            bg_progress["context"].total = store.count_uncontextualized_records()
        except Exception:
            pass
    if pending_embed:
        bg_progress["embed"] = BgProgress("embed")
        try:
            ids = store.get_unembedded_record_ids(embed_cfg.model, limit=_EMBED_BATCH_LIMIT)
            bg_progress["embed"].total = len(ids)
        except Exception:
            pass

    def _pipeline_coordinator() -> None:
        """Run enrich → context → embed sequentially, cycling until done."""
        from egovault.core.enrichment import EnrichmentPipeline
        from egovault.core.store import VaultStore as _VSBG

        bg_store = _VSBG(settings.vault_db)
        bg_store.init_db()
        base_url = embed_cfg.base_url.strip() or llm.base_url
        model_name = embed_cfg.model

        # Each task runs for up to this many seconds before yielding to the next.
        _SLOT_SECONDS = 3_600  # 1 hour per task per cycle

        try:
            while True:
                did_work = False

                # ── 1. enrich ─────────────────────────────────────────────
                slot_deadline = time.time() + _SLOT_SECONDS
                while time.time() < slot_deadline:
                    batch = []
                    try:
                        batch = bg_store.get_unenriched_records(limit=100)
                    except Exception:
                        pass
                    if not batch:
                        break
                    did_work = True
                    _prog = bg_progress.setdefault("enrich", BgProgress("enrich"))
                    if not _prog.total:
                        try:
                            _prog.total = bg_store.count_unenriched_records()
                        except Exception:
                            pass
                    pipeline = EnrichmentPipeline(bg_store, settings)
                    ok = _prog.done - _prog.failed
                    fail = _prog.failed
                    for rec in batch:
                        if pipeline.enrich_record(rec):
                            ok += 1
                        else:
                            fail += 1
                        _prog.done = ok + fail
                        _prog.failed = fail
                        if time.time() >= slot_deadline:
                            break  # yield mid-batch if slot expired

                # ── 2. context ────────────────────────────────────────────
                if getattr(embed_cfg, "contextual_enabled", False):
                    slot_deadline = time.time() + _SLOT_SECONDS
                    while time.time() < slot_deadline:
                        ctx_ids = []
                        try:
                            ctx_ids = bg_store.get_uncontextualized_record_ids(limit=100)
                        except Exception:
                            pass
                        if not ctx_ids:
                            break
                        did_work = True
                        _prog = bg_progress.setdefault("context", BgProgress("context"))
                        if not _prog.total:
                            try:
                                _prog.total = bg_store.count_uncontextualized_records()
                            except Exception:
                                pass
                        pipeline = EnrichmentPipeline(bg_store, settings)
                        id_set = set(ctx_ids)
                        ctx_batch = [r for r in bg_store.get_records() if r.id in id_set]
                        ok = _prog.done - _prog.failed
                        fail = _prog.failed
                        for rec in ctx_batch:
                            if pipeline.contextualize_record(rec):
                                ok += 1
                            else:
                                fail += 1
                            _prog.done = ok + fail
                            _prog.failed = fail
                            if time.time() >= slot_deadline:
                                break

                # ── 3. embed ──────────────────────────────────────────────
                if embed_cfg.enabled:
                    slot_deadline = time.time() + _SLOT_SECONDS
                    while time.time() < slot_deadline:
                        embed_ids = []
                        try:
                            embed_ids = bg_store.get_unembedded_record_ids(model_name, limit=100)
                        except Exception:
                            pass
                        if not embed_ids:
                            break
                        did_work = True
                        from egovault.chat.rag import embed_text
                        _prog = bg_progress.setdefault("embed", BgProgress("embed"))
                        if not _prog.total:
                            try:
                                all_ids = bg_store.get_unembedded_record_ids(model_name, limit=_EMBED_BATCH_LIMIT)
                                _prog.total = len(all_ids)
                            except Exception:
                                pass
                        ok = _prog.done - _prog.failed
                        fail = _prog.failed
                        for record_id in embed_ids:
                            text = bg_store.get_record_text_by_id(record_id)
                            try:
                                if text.strip():
                                    vec = embed_text(text, base_url, model_name, timeout=60)
                                    bg_store.upsert_embedding(record_id, model_name, vec)
                                else:
                                    bg_store.upsert_embedding(record_id, model_name, [0.0])
                                ok += 1
                            except Exception:
                                fail += 1
                            _prog.done = ok + fail
                            _prog.failed = fail
                            if time.time() >= slot_deadline:
                                break

                if not did_work:
                    break  # all tasks finished; exit coordinator

            # ── summary notices ───────────────────────────────────────────
            for label, _prog in bg_progress.items():
                if _prog.total == 0:
                    continue
                fail_str = f", {_prog.failed} failed" if _prog.failed else ""
                notice_queue.put(
                    f"[dim]Background {label} finished: {_prog.done - _prog.failed} done{fail_str}.[/dim]"
                )

        except Exception as exc:
            notice_queue.put(f"[dim]Background pipeline stopped: {exc}[/dim]")
        finally:
            bg_store.close()

    t = threading.Thread(target=_pipeline_coordinator, name="bg-pipeline", daemon=True)
    t.start()
    return [t]


def run_chat_session(store: VaultStore, settings: Settings) -> None:
    """Start the Level-0 interactive chat REPL."""
    llm = settings.llm
    top_n = 10
    last_sources: list[str] = []
    last_file_path: str = ""
    conversation_history: list[dict] = []
    prompt_session = _make_prompt_session()

    # Build LLM kwargs once — reused for planner, profile extractor, and main chat.
    _llm_kwargs = dict(
        base_url=llm.base_url,
        model=llm.model,
        timeout=llm.timeout_seconds,
        provider=llm.provider,
        api_key=llm.api_key,
    )

    # Load or extract owner profile (cached in vault settings table).
    owner_profile = store.get_owner_profile()
    if not owner_profile:
        with console.status("[dim]Building owner profile…[/dim]", spinner="dots"):
            owner_profile = extract_owner_profile(store, _call_llm, _llm_kwargs)

    console.print(Rule())
    console.print(f"{_BANNER}  [dim]v{_get_version()}[/dim]")
    console.print(Rule())
    console.print()

    welcome = load_agent_prompts().get("welcome", "")
    if welcome:
        console.print(Markdown(welcome))
        console.print()

    # Start background pipeline (enrich → context → embed, sequential) if work is pending.
    _bg_notice_queue: queue.Queue[str] = queue.Queue()
    _bg_progress: dict[str, BgProgress] = {}
    _bg_threads = _start_background_tasks(store, settings, _bg_notice_queue, _bg_progress)
    if _bg_threads:
        pending_tasks = list(_bg_progress.keys())
        task_names = " → ".join(pending_tasks)
        console.print(
            f"[dim]Background pipeline: {task_names} (sequential, one at a time) "
            f"— chat is fully usable now.[/dim]\n"
        )

    # ── Scheduler ────────────────────────────────────────────────────────────
    from egovault.utils.scheduler import Scheduler, make_executor
    from pathlib import Path as _Path

    _data_dir = _Path(settings.vault_db).parent
    _scheduler = Scheduler(_data_dir)
    _scheduler.start(
        executor=make_executor(settings.vault_db, settings),
        notice_queue=_bg_notice_queue,
    )
    _register_auto_schedules(_scheduler, settings)
    if _scheduler.list_tasks():
        task_count = len(_scheduler.list_tasks())
        console.print(
            f"[dim]Scheduler: {task_count} task(s) active. "
            f"Type [bold]/schedule --list[/bold] to view.[/dim]\n"
        )

    while True:
        try:
            if prompt_session is not None:
                raw = prompt_session.prompt("EgoVault> ")
            else:
                raw = input("EgoVault> ")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Session ended.[/dim]")
            break

        user_input = raw.strip()
        if not user_input:
            # Drain background notices even on empty input so they surface promptly.
            while True:
                try:
                    console.print(_bg_notice_queue.get_nowait())
                except queue.Empty:
                    break
            continue

        # Drain any background task completion notices before processing the turn.
        while True:
            try:
                console.print(_bg_notice_queue.get_nowait())
            except queue.Empty:
                break

        # ---- Catch accidental shell commands typed at the chat prompt ----
        # e.g. user types "egovault chat" or "python ..." out of habit.
        _shell_cmd_re = re.compile(
            r'^(egovault|python|pip|npm|node|git|cd|ls|dir|cls|clear|powershell|cmd)\b',
            re.IGNORECASE,
        )
        if _shell_cmd_re.match(user_input):
            console.print(
                "[dim]Tip: you're already inside EgoVault chat. "
                "Type [cyan]/help[/cyan] to see available commands.[/dim]"
            )
            continue

        # ---- Natural-language → command intent resolution ----
        if not user_input.startswith("/"):
            resolved = _resolve_intent(user_input)
            if resolved:
                console.print(f"[dim]→ {resolved}[/dim]")
                user_input = resolved

        # ---- Commands ----
        lower = user_input.lower()

        # -- Common commands via central dispatcher --
        from egovault.agent.commands import handle_command as _hc
        _cmd_ctx = {
            "settings": settings,
            "sources": last_sources,
            "owner_profile": owner_profile,
            "top_n": top_n,
        }
        _result = _hc(user_input, _cmd_ctx)
        if _result is not None:
            if _result.action == "exit":
                console.print("[dim]Goodbye.[/dim]")
                break
            if _result.action == "clear":
                conversation_history.clear()
                last_sources = []
                console.clear()
                console.print(_BANNER)
                continue
            if _result.action == "restart":
                conversation_history.clear()
                last_sources = []
                console.clear()
                console.print(_BANNER)
                console.print("[dim]Conversation history cleared.[/dim]")
                continue
            if _result.action == "refresh_profile":
                with console.status("[dim]Extracting profile\u2026[/dim]", spinner="dots"):
                    owner_profile = extract_owner_profile(store, _call_llm, _llm_kwargs)
                if owner_profile:
                    console.print(Panel(owner_profile, title="[bold]Owner Profile[/bold]", border_style="cyan"))
                else:
                    console.print("[yellow]No profile data found. Try /scan <home-folder> first.[/yellow]")
                continue
            if _result.action == "top_n":
                top_n = _result.value
            if _result.text:
                from rich.markdown import Markdown as _MD
                console.print(_MD(_result.text))
            continue

        if lower.startswith("/scan"):
            _handle_scan(user_input, store, settings)
            # Re-extract profile after scan in case new personal files were added
            store.set_setting("owner_profile", "")
            owner_profile = ""
            continue

        if lower.startswith("/gmail-auth"):
            _handle_gmail_auth(store, settings)
            continue

        if lower.startswith("/gmail-sync"):
            _handle_gmail_sync(user_input, store, settings)
            # Re-extract profile after a sync — new emails may reveal new personal context.
            store.set_setting("owner_profile", "")
            owner_profile = ""
            continue

        if lower.startswith("/telegram-auth"):
            _handle_telegram_auth(store, settings)
            continue

        if lower.startswith("/telegram-sync"):
            _handle_telegram_sync(user_input, store, settings)
            continue

        if lower.startswith("/schedule"):
            _handle_schedule(user_input, store, settings, _scheduler, _bg_threads, _bg_progress)
            continue

        if lower in ("/open",) or lower.startswith("/open "):
            if not last_file_path:
                console.print("[yellow]No recent file to open — ask me to search or save a file first.[/yellow]")
            else:
                msg = _open_with_default_app(last_file_path)
                console.print(f"[dim]{msg}[/dim]")
            continue

        # ---- Agent pipeline (agentic tool-calling) ----
        # Resolve the configured output directory so the LLM knows where to save files.
        _output_dir = str(Path(settings.output_dir).expanduser().resolve())
        from datetime import date as _date
        _today = _date.today().isoformat()
        initial_messages = build_prompt(
            user_input, "", history=conversation_history,
            owner_profile=owner_profile, output_dir=_output_dir,
            today=_today,
        )
        collected_chunks: list = []
        llm_data: dict = {}

        # Agentic loop: LLM decides what to search, can call tools multiple times.
        _session_ctx: dict = {
            "settings": settings,
            "last_sources": last_sources,
            "owner_profile": owner_profile,
            "owner_profile_ref": {},  # set dirty=True when scan happens
            "call_llm_fn": _call_llm,
            "hyde_llm_kwargs": {**_llm_kwargs, "timeout": min(_HYDE_TIMEOUT_CAP, llm.timeout_seconds)},
            "scheduler": _scheduler,
            "notice_queue": _bg_notice_queue,
        }
        with console.status("[dim]Thinking…[/dim]", spinner="dots"):
            try:
                answer, llm_data, collected_chunks = _call_llm_agent(
                    initial_messages, store, top_n, _llm_kwargs, console,
                    session_ctx=_session_ctx,
                )
            except Exception as exc:
                from egovault.utils.llm_errors import classify_llm_error  # noqa: PLC0415
                _code, _msg = classify_llm_error(exc)
                console.print(f"[red]LLM error ({_code}):[/red] {_msg}")
                continue
        # If a scan happened inside the agent loop, refresh the owner profile.
        if _session_ctx["owner_profile_ref"].get("dirty"):
            store.set_setting("owner_profile", "")
            owner_profile = ""

        last_sources = source_attribution(collected_chunks) if collected_chunks else []
        # Track the last file path for /open command.
        if _session_ctx.get("last_file"):
            last_file_path = _session_ctx["last_file"]
        elif collected_chunks:
            _fp = list(dict.fromkeys(c.record.file_path for c in collected_chunks if c.record.file_path))
            if len(_fp) == 1:
                last_file_path = _fp[0]

        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": answer})

        console.print()
        console.print(Markdown(answer))
        console.print()
        meta_parts: list[str] = []
        if last_sources:
            meta_parts.append(
                f"Sources ({len(last_sources)}): "
                + " · ".join(last_sources[:3])
                + (" …" if len(last_sources) > 3 else "")
            )
        if meta_parts:
            console.print("[dim]" + "   ".join(meta_parts) + "[/dim]")
            console.print()

    _scheduler.stop()
