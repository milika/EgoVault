"""Terminal REPL frontend for EgoVault."""
from __future__ import annotations

import queue
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from egovault.config import Settings
    from egovault.core.store import VaultStore

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule

from egovault.agent.intent import (
    _BANNER,
)
from egovault.agent.commands import (
    _handle_gmail_auth, _handle_gmail_sync, _handle_scan,
    _handle_telegram_auth, _handle_telegram_sync,
)
from egovault.agent.session import (
    AgentSession, _call_llm, _handle_schedule,
    _register_auto_schedules, BgProgress, _start_background_tasks,
)
from egovault.processing.rag import (
    extract_owner_profile,
)
from egovault.config import Settings, load_agent_prompts
from egovault.core.store import VaultStore

console = Console(highlight=False)

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
    console.print(_BANNER)
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

    # ── Agent session + persistent turn context ────────────────────────────
    _agent_session = AgentSession(store, settings)
    _session_ctx: dict = {
        "owner_profile": owner_profile,
        "owner_profile_ref": {},
        "call_llm_fn": _call_llm,
        "hyde_llm_kwargs": {**_llm_kwargs, "timeout": min(15, llm.timeout_seconds)},
        "scheduler": _scheduler,
        "notice_queue": _bg_notice_queue,
        "bg_threads": _bg_threads,
        "bg_progress": _bg_progress,
        "last_sources": last_sources,
        "top_n": top_n,
    }

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

        # ---- Unified dispatch via AgentSession.process_turn() ----
        # Keep session_ctx in sync with mutable per-turn state.
        _session_ctx["last_sources"] = last_sources
        _session_ctx["owner_profile"] = owner_profile
        _session_ctx["owner_profile_ref"] = {}
        _session_ctx["top_n"] = top_n

        def _emit(label: str) -> None:
            console.print(f"[dim]↳ {label}[/dim]")

        with console.status("[dim]Working…[/dim]", spinner="dots"):
            turn = _agent_session.process_turn(
                user_input, conversation_history, emit=_emit, session_ctx=_session_ctx,
            )

        # ---- Heavy commands: process_turn returns action="_delegate" ----
        if turn.action == "_delegate":
            delegated = (turn.value or "").strip()
            lower = delegated.lower()
            if lower.startswith("/scan"):
                _handle_scan(delegated, store, settings)
                store.set_setting("owner_profile", "")
                owner_profile = ""
            elif lower.startswith("/gmail-auth"):
                _handle_gmail_auth(store, settings)
            elif lower.startswith("/gmail-sync"):
                _handle_gmail_sync(delegated, store, settings)
                store.set_setting("owner_profile", "")
                owner_profile = ""
            elif lower.startswith("/telegram-auth"):
                _handle_telegram_auth(store, settings)
            elif lower.startswith("/telegram-sync"):
                _handle_telegram_sync(delegated, store, settings)
            elif lower.startswith("/schedule"):
                _handle_schedule(delegated, store, settings, _scheduler, _bg_threads, _bg_progress)
            elif lower in ("/open",) or lower.startswith("/open "):
                if not last_file_path:
                    console.print("[yellow]No recent file to open — ask me to search or save a file first.[/yellow]")
                else:
                    from egovault.chat.session import _open_with_default_app
                    console.print(f"[dim]{_open_with_default_app(last_file_path)}[/dim]")
            continue

        # ---- Simple command side-effects ----
        if turn.action == "exit":
            console.print("[dim]Goodbye.[/dim]")
            break

        if turn.action in ("clear", "restart"):
            conversation_history.clear()
            last_sources = []
            console.clear()
            console.print(_BANNER)
            if turn.action == "restart":
                console.print("[dim]Conversation history cleared.[/dim]")
            continue

        if turn.action == "refresh_profile":
            with console.status("[dim]Extracting profile…[/dim]", spinner="dots"):
                owner_profile = extract_owner_profile(store, _call_llm, _llm_kwargs) or ""
            if owner_profile:
                console.print(Panel(owner_profile, title="[bold]Owner Profile[/bold]", border_style="cyan"))
            else:
                console.print("[yellow]No profile data found. Try /scan <home-folder> first.[/yellow]")
            continue

        if turn.action == "top_n":
            top_n = turn.value

        # ---- Render response (commands + agent answers) ----
        if turn.text:
            console.print()
            console.print(Markdown(turn.text))
            console.print()

        # ---- Update session state (agent turns only) ----
        if not turn.is_command:
            conversation_history = list(turn.updated_history)
            last_sources = turn.sources
            if _session_ctx.get("last_file"):
                last_file_path = _session_ctx["last_file"]
            elif turn.attachments:
                last_file_path = turn.attachments[0]
            # Profile refresh when a scan happened inside the agent loop.
            if _session_ctx["owner_profile_ref"].get("dirty"):
                store.set_setting("owner_profile", "")
                owner_profile = ""
            if last_sources:
                console.print(
                    "[dim]Sources ("
                    + str(len(last_sources))
                    + "): "
                    + " · ".join(last_sources[:3])
                    + (" …" if len(last_sources) > 3 else "")
                    + "[/dim]"
                )
                console.print()

    _scheduler.stop()


