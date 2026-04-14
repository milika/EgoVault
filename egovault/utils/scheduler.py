"""Task scheduler for EgoVault — runs periodic and one-shot tasks in the background.

Supported time expressions
--------------------------
One-shot (run once after a delay):
    in 5min  /  in 30 seconds  /  in 2 hours  /  in 1 day

Recurring (run repeatedly):
    every 30min  /  every hour  /  every day
    every day at 19:05  /  every morning  /  every evening
    daily at 08:30

Scheduled commands
------------------
    /gmail-sync        — sync Gmail (IMAP or OAuth, whichever is configured)
    /scan inbox        — scan the configured inbox folder
    /scan <folder>     — scan any known folder alias

Persistence
-----------
Tasks are stored in ``<data_dir>/schedule.json`` and survive process restarts.
"""
from __future__ import annotations

import json
import logging
import queue
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from egovault.config import Settings
    from egovault.core.store import VaultStore

logger = logging.getLogger(__name__)

# How often the background thread wakes to check for due tasks.
_TICK_SECONDS = 30


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ScheduledTask:
    """A single scheduled task entry."""
    id: str                       # short UUID prefix — used when cancelling
    name: str                     # human-readable label shown in /schedule --list
    command: str                  # e.g. "/gmail-sync" or "/scan inbox"
    next_run: float               # Unix timestamp of the next (or only) run
    interval_seconds: int | None  # None → one-shot; >0 → repeat every N seconds
    enabled: bool = True
    created_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Time expression parser
# ---------------------------------------------------------------------------

def _parse_duration_seconds(n: int, unit: str) -> float | None:
    """Convert (n, unit) to seconds. Returns None for unknown units."""
    u = unit.lower().rstrip("s")  # normalise: "minutes" → "minute"
    if u in ("s", "sec", "second"):
        return float(n)
    if u in ("m", "min", "minute"):
        return float(n * 60)
    if u in ("h", "hr", "hour"):
        return float(n * 3_600)
    if u in ("d", "day"):
        return float(n * 86_400)
    return None


def _next_clock_time(hour: int, minute: int) -> float:
    """Return the Unix timestamp for the next occurrence of HH:MM (today or tomorrow)."""
    now = datetime.now()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return target.timestamp()


# Pre-compiled patterns for parse_schedule_expression.
_RE_IN_DELAY = re.compile(
    r"\bin\s+(\d+)\s*(s|sec(?:ond)?s?|m|min(?:ute)?s?|h|hr?s?|hour?s?|d|days?)\b",
    re.IGNORECASE,
)
_RE_EVERY_N = re.compile(
    r"\bevery\s+(\d+)\s*(s|sec(?:ond)?s?|m|min(?:ute)?s?|h|hr?s?|hour?s?|d|days?)\b",
    re.IGNORECASE,
)
_RE_EVERY_WORD = re.compile(
    r"\bevery\s+(hour|minute|day|morning|evening|night)\b",
    re.IGNORECASE,
)
_RE_DAILY_AT = re.compile(
    r"\b(?:daily|every\s+day)\b.*?\bat\s+(\d{1,2}):(\d{2})\b",
    re.IGNORECASE,
)
_RE_AT_TIME = re.compile(r"\bat\s+(\d{1,2}):(\d{2})\b", re.IGNORECASE)

_WORD_PRESET_SECONDS = {
    "hour":    3_600,
    "minute":  60,
    "day":     86_400,
    "morning": 86_400,
    "evening": 86_400,
    "night":   86_400,
}
_WORD_DEFAULT_TIME = {
    "morning": (9, 0),
    "evening": (18, 0),
    "night":   (22, 0),
}


def parse_schedule_expression(text: str) -> tuple[float, int | None] | None:
    """Parse a natural-language schedule expression.

    Returns ``(next_run_timestamp, interval_seconds_or_None)`` on success,
    or ``None`` if the expression cannot be understood.

    Examples::

        parse_schedule_expression("in 5min")
            → (now + 300, None)
        parse_schedule_expression("every 30min")
            → (now + 1800, 1800)
        parse_schedule_expression("every day at 19:05")
            → (next 19:05 timestamp, 86400)
        parse_schedule_expression("every morning")
            → (next 09:00 timestamp, 86400)
    """
    now = time.time()
    lower = text.lower().strip()

    # 1. "in N unit" → one-shot
    m = _RE_IN_DELAY.search(lower)
    if m:
        delta = _parse_duration_seconds(int(m.group(1)), m.group(2))
        if delta:
            return now + delta, None

    # 2. "every N unit" → recurring
    m = _RE_EVERY_N.search(lower)
    if m:
        delta = _parse_duration_seconds(int(m.group(1)), m.group(2))
        if delta:
            interval = int(delta)
            return now + delta, interval

    # 3. "every hour / every day / every morning / every evening …"
    m = _RE_EVERY_WORD.search(lower)
    if m:
        word = m.group(1).lower()
        interval = _WORD_PRESET_SECONDS[word]

        # Check for "at HH:MM" override
        t = _RE_AT_TIME.search(lower)
        if t:
            hour, minute = int(t.group(1)), int(t.group(2))
            next_run = _next_clock_time(hour, minute)
        elif word in _WORD_DEFAULT_TIME:
            h, mn = _WORD_DEFAULT_TIME[word]
            next_run = _next_clock_time(h, mn)
        else:
            next_run = now + interval

        return next_run, interval

    # 4. "daily at HH:MM" / "every day at HH:MM"
    m = _RE_DAILY_AT.search(lower)
    if m:
        hour, minute = int(m.group(1)), int(m.group(2))
        return _next_clock_time(hour, minute), 86_400

    return None


def format_next_run(ts: float) -> str:
    """Return a human-readable relative time string for a future timestamp."""
    now = time.time()
    delta = int(ts - now)
    if delta <= 0:
        return "now"
    if delta < 60:
        return f"in {delta}s"
    if delta < 3_600:
        return f"in {delta // 60}min"
    if delta < 86_400:
        h, m = divmod(delta // 60, 60)
        return f"in {h}h {m}min" if m else f"in {h}h"
    d = delta // 86_400
    return f"in {d}d"


def format_interval(seconds: int | None) -> str:
    """Return a human-readable interval string."""
    if seconds is None:
        return "once"
    if seconds < 60:
        return f"every {seconds}s"
    if seconds < 3_600:
        return f"every {seconds // 60}min"
    if seconds < 86_400:
        h, m = divmod(seconds // 60, 60)
        return f"every {h}h {m}min" if m else f"every {h}h"
    d = seconds // 86_400
    return f"every {d}d"


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class Scheduler:
    """Background scheduler — persists tasks to disk and runs them on time.

    Lifecycle::

        scheduler = Scheduler(data_dir)
        executor = _make_executor(store, settings)
        scheduler.start(executor, notice_queue)
        # ... later ...
        scheduler.stop()

    The *executor* callable receives ``(command: str, notice_queue: queue.Queue)``
    and is responsible for dispatching the command and posting results to the queue.
    """

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._path = data_dir / "schedule.json"
        self._tasks: list[ScheduledTask] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._executor: Callable[[str, "queue.Queue[str]"], None] | None = None
        self._queue: "queue.Queue[str] | None" = None
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            self._tasks = [ScheduledTask(**t) for t in raw]
            logger.debug("Loaded %d scheduled task(s).", len(self._tasks))
        except Exception as exc:
            logger.warning("Could not load schedule.json: %s", exc)

    def _save(self) -> None:
        self._data_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._path.write_text(
                json.dumps([asdict(t) for t in self._tasks], indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Could not save schedule.json: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_task(
        self,
        name: str,
        command: str,
        next_run: float,
        interval_seconds: int | None,
    ) -> ScheduledTask:
        """Register a new task and persist it. Returns the new task."""
        task = ScheduledTask(
            id=str(uuid.uuid4())[:8],
            name=name,
            command=command,
            next_run=next_run,
            interval_seconds=interval_seconds,
        )
        with self._lock:
            self._tasks.append(task)
            self._save()
        logger.info("Scheduled task added: %s (id=%s)", task.name, task.id)
        return task

    def cancel_task(self, task_id: str) -> bool:
        """Remove a task by its id. Returns True if it was found and removed."""
        with self._lock:
            before = len(self._tasks)
            self._tasks = [t for t in self._tasks if t.id != task_id]
            removed = len(self._tasks) < before
            if removed:
                self._save()
        return removed

    def list_tasks(self) -> list[ScheduledTask]:
        """Return a snapshot of all registered tasks."""
        with self._lock:
            return list(self._tasks)

    # ------------------------------------------------------------------
    # Startup / shutdown
    # ------------------------------------------------------------------

    def start(
        self,
        executor: Callable[[str, "queue.Queue[str]"], None],
        notice_queue: "queue.Queue[str]",
    ) -> None:
        """Start the background tick thread."""
        self._executor = executor
        self._queue = notice_queue
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._tick_loop,
            name="bg-scheduler",
            daemon=True,
        )
        self._thread.start()
        logger.debug("Scheduler started (%d task(s)).", len(self._tasks))

    def stop(self) -> None:
        """Signal the tick thread to exit."""
        self._stop.set()

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    def _tick_loop(self) -> None:
        while not self._stop.wait(_TICK_SECONDS):
            now = time.time()
            with self._lock:
                due = [t for t in self._tasks if t.enabled and t.next_run <= now]

            for task in due:
                self._run_task(task)
                with self._lock:
                    if task in self._tasks:
                        if task.interval_seconds:
                            task.next_run = now + task.interval_seconds
                        else:
                            self._tasks.remove(task)
                        self._save()

    def _run_task(self, task: ScheduledTask) -> None:
        """Execute a single task in a new daemon thread so we don't block the ticker."""
        if self._queue:
            self._queue.put(
                f"[dim]⏰ Running scheduled task: [bold]{task.name}[/bold]…[/dim]"
            )

        def _worker() -> None:
            try:
                if self._executor and self._queue:
                    self._executor(task.command, self._queue)
            except Exception as exc:
                logger.error("Scheduled task %r failed: %s", task.name, exc)
                if self._queue:
                    self._queue.put(
                        f"[yellow]⚠ Scheduled task [bold]{task.name}[/bold] failed: {exc}[/yellow]"
                    )

        t = threading.Thread(target=_worker, name=f"task-{task.id}", daemon=True)
        t.start()


# ---------------------------------------------------------------------------
# Default executor (dispatches /gmail-sync and /scan commands)
# ---------------------------------------------------------------------------

def make_executor(
    vault_db: str,
    settings: "Settings",
) -> Callable[[str, "queue.Queue[str]"], None]:
    """Build a command executor that runs scheduled tasks against a fresh DB connection.

    Each invocation opens its own ``VaultStore`` connection so it is safe to
    call from a background thread while the main REPL is using the primary
    connection.
    """
    # Pre-import all adapter modules in the calling (main) thread.
    # adapters/__init__.py mass-imports every submodule so its @register
    # decorators fire; if two task threads trigger this for the first time
    # simultaneously they deadlock on Python's _ModuleLock.  Importing here
    # guarantees every lock is settled before any daemon thread runs.
    import egovault.adapters  # noqa: F401

    def _execute(command: str, notice_queue: "queue.Queue[str]") -> None:
        from egovault.core.store import VaultStore

        lower = command.lower().strip()
        bg_store = VaultStore(vault_db)
        bg_store.init_db()

        try:
            if lower.startswith("/gmail-sync"):
                _exec_gmail_sync(bg_store, settings, notice_queue)
            elif lower.startswith("/telegram-sync"):
                _exec_telegram_sync(bg_store, settings, notice_queue)
            elif lower.startswith("/scan"):
                folder_arg = command.split(None, 1)[1].strip() if " " in command else "inbox"
                _exec_scan(folder_arg, bg_store, settings, notice_queue)
            elif lower.startswith("chat:"):
                prompt = command[5:].strip()
                _exec_chat_prompt(prompt, bg_store, settings, notice_queue)
            else:
                notice_queue.put(f"[yellow]⚠ Unknown scheduled command: {command}[/yellow]")
        finally:
            bg_store.close()

    return _execute


def _exec_gmail_sync(
    store: "VaultStore",
    settings: "Settings",
    notice_queue: "queue.Queue[str]",
) -> None:
    """Background Gmail sync — IMAP or OAuth, whichever is configured."""
    from pathlib import Path as _Path

    data_dir = _Path(settings.vault_db).parent

    # Prefer IMAP
    try:
        from egovault.utils.gmail_imap import load_credentials as _load_imap
        imap_creds = _load_imap(data_dir)
    except Exception:
        imap_creds = None

    if imap_creds is not None:
        from egovault.adapters.gmail_imap_adapter import GmailImapAdapter
        gmail_addr, app_pwd = imap_creds
        since = store.get_setting("gmail_last_sync") or ""
        inserted = skipped = 0
        try:
            adapter = GmailImapAdapter(store=store)
            for record in adapter.ingest_from_imap(
                gmail_address=gmail_addr,
                app_password=app_pwd,
                since=since,
                max_results=100_000,  # no cap in background — fetch everything
            ):
                was_new = store.upsert_record(record)
                if was_new:
                    inserted += 1
                else:
                    skipped += 1
        except Exception as exc:
            notice_queue.put(f"[yellow]⚠ Scheduled Gmail sync (IMAP) failed: {exc}[/yellow]")
            return
    else:
        # Try OAuth
        try:
            from egovault.utils.gmail_auth import get_token_path, load_credentials as _load_oauth
            from egovault.adapters.gmail_api import GmailApiAdapter
            token_path = get_token_path(data_dir)
            if _load_oauth(token_path) is None:
                notice_queue.put(
                    "[yellow]⚠ Scheduled Gmail sync skipped — not authenticated. "
                    "Run [bold]/gmail-auth[/bold] first.[/yellow]"
                )
                return
            since = store.get_setting("gmail_last_sync") or ""
            effective_query = (f"after:{since.replace('-','/')}" if since
                               else "-in:spam -in:trash")
            adapter = GmailApiAdapter(store=store)
            inserted = skipped = 0
            for record in adapter.ingest_from_api(
                token_path=token_path,
                query=effective_query,
                max_results=100_000,  # no cap in background — fetch everything
            ):
                was_new = store.upsert_record(record)
                if was_new:
                    inserted += 1
                else:
                    skipped += 1
        except ImportError:
            notice_queue.put(
                "[yellow]⚠ Scheduled Gmail sync skipped — no auth configured.[/yellow]"
            )
            return
        except Exception as exc:
            notice_queue.put(f"[yellow]⚠ Scheduled Gmail sync (OAuth) failed: {exc}[/yellow]")
            return

    from datetime import date
    store.set_setting("gmail_last_sync", date.today().strftime("%Y-%m-%d"))
    notice_queue.put(
        f"[green]✓[/green] Scheduled Gmail sync — "
        f"[bold]{inserted}[/bold] new  [dim]({skipped} already in vault)[/dim]"
    )


def _exec_scan(
    folder_arg: str,
    store: "VaultStore",
    settings: "Settings",
    notice_queue: "queue.Queue[str]",
) -> None:
    """Background inbox / folder scan."""
    from pathlib import Path as _Path
    from egovault.adapters.local_inbox import LocalInboxAdapter
    from egovault.utils.folders import resolve_folder

    # Resolve "inbox" alias to the configured inbox_dir.
    import re as _re
    if _re.search(r'^[/\\]?inbox[/\\]?$', folder_arg.strip(), _re.IGNORECASE):
        src = _Path(settings.inbox_dir).expanduser().resolve()
    else:
        resolved = resolve_folder(folder_arg.strip())
        src = resolved if resolved else _Path(folder_arg).expanduser().resolve()

    if not src.exists():
        notice_queue.put(f"[yellow]⚠ Scheduled scan: folder not found: {src}[/yellow]")
        return

    adapter = LocalInboxAdapter(store=store)
    inserted = skipped = 0
    try:
        for record in adapter.ingest(src):
            was_new = store.upsert_record(record)
            if was_new:
                inserted += 1
                if record.file_path:
                    store.upsert_ingested_file(
                        file_id=str(record.raw.get("file_id", "")),
                        path=record.file_path,
                        mtime=float(record.raw.get("mtime", 0)),
                        size_bytes=int(record.raw.get("size_bytes", 0)),
                        platform="local_inbox",
                        content_hash=record.raw.get("content_hash"),
                    )
            else:
                skipped += 1
    except Exception as exc:
        notice_queue.put(f"[yellow]⚠ Scheduled scan failed: {exc}[/yellow]")
        return

    notice_queue.put(
        f"[green]✓[/green] Scheduled scan [cyan]{src.name}[/cyan] — "
        f"[bold]{inserted}[/bold] new  [dim]({skipped} already in vault)[/dim]"
    )


def _exec_telegram_sync(
    store: "VaultStore",
    settings: "Settings",
    notice_queue: "queue.Queue[str]",
) -> None:
    """Background Telegram sync via Telethon MTProto."""
    from pathlib import Path as _Path

    data_dir = _Path(settings.vault_db).parent

    try:
        from egovault.utils.telegram_api import get_session_path, load_credentials as _load_tg
    except ImportError:
        notice_queue.put(
            "[yellow]\u26a0 Scheduled Telegram sync skipped \u2014 Telethon not installed.[/yellow]"
        )
        return

    creds = _load_tg(data_dir)
    if creds is None:
        notice_queue.put(
            "[yellow]\u26a0 Scheduled Telegram sync skipped \u2014 not authenticated. "
            "Run [bold]egovault telegram-auth[/bold] first.[/yellow]"
        )
        return

    session_path = get_session_path(data_dir)
    if not session_path.with_suffix(".session").exists():
        notice_queue.put(
            "[yellow]\u26a0 Scheduled Telegram sync skipped \u2014 session file missing. "
            "Run [bold]egovault telegram-auth[/bold] to re-authenticate.[/yellow]"
        )
        return

    since = store.get_setting("telegram_last_sync") or ""
    inserted = skipped = 0
    try:
        from egovault.adapters.telegram_history import TelegramHistoryAdapter
        adapter = TelegramHistoryAdapter(store=store)
        for record in adapter.ingest_from_api(
            api_id=creds["api_id"],
            api_hash=creds["api_hash"],
            phone=creds["phone"],
            session_path=session_path,
            since=since,
            max_messages=100_000,
        ):
            was_new = store.upsert_record(record)
            if was_new:
                inserted += 1
            else:
                skipped += 1
    except Exception as exc:
        notice_queue.put(f"[yellow]\u26a0 Scheduled Telegram sync failed: {exc}[/yellow]")
        return

    from datetime import date
    store.set_setting("telegram_last_sync", date.today().strftime("%Y-%m-%d"))
    notice_queue.put(
        f"[green]\u2713[/green] Scheduled Telegram sync \u2014 "
        f"[bold]{inserted}[/bold] new  [dim]({skipped} already in vault)[/dim]"
    )


def _exec_chat_prompt(
    prompt: str,
    store: "VaultStore",
    settings: "Settings",
    notice_queue: "queue.Queue[str]",
) -> None:
    """Execute an arbitrary chat prompt as a scheduled background task.

    Runs the full agentic tool-calling loop (web search, write_file, etc.)
    exactly as if the user had typed the prompt in the REPL.  Progress lines
    are posted to *notice_queue* and the final answer is displayed there too.
    """
    from pathlib import Path as _Path
    from datetime import date as _date

    if not prompt:
        notice_queue.put("[yellow]⚠ Scheduled chat: empty prompt, skipping.[/yellow]")
        return

    notice_queue.put(f"[dim]⏰ Running scheduled prompt: {prompt[:80]!r}…[/dim]")

    try:
        from egovault.chat.rag import build_prompt
        from egovault.chat.session import _call_llm, _call_llm_agent
        from egovault.utils.llm import auto_top_n
    except Exception as exc:
        notice_queue.put(f"[yellow]⚠ Scheduled chat import error: {exc}[/yellow]")
        return

    llm_cfg = settings.llm
    llm_kwargs: dict = dict(
        base_url=llm_cfg.base_url,
        model=llm_cfg.model,
        timeout=llm_cfg.timeout_seconds,
        provider=llm_cfg.provider,
        api_key=llm_cfg.api_key,
        num_gpu=getattr(llm_cfg, "num_gpu", -1),
        num_thread=getattr(llm_cfg, "num_thread", 0),
        num_ctx=getattr(llm_cfg, "num_ctx", 0),
        auto_ctx=getattr(llm_cfg, "auto_ctx", False),
    )

    top_n = auto_top_n()
    owner_profile = store.get_owner_profile() or ""
    output_dir = str(_Path(settings.output_dir).expanduser().resolve())
    today = _date.today().isoformat()

    session_ctx: dict = {
        "settings": settings,
        "last_sources": [],
        "owner_profile": owner_profile,
        "owner_profile_ref": {},
        "call_llm_fn": _call_llm,
        "hyde_llm_kwargs": {**llm_kwargs, "timeout": min(15, llm_cfg.timeout_seconds)},
    }

    initial_messages = build_prompt(
        prompt, "",
        history=[],
        owner_profile=owner_profile,
        output_dir=output_dir,
        today=today,
    )

    try:
        answer, _, _ = _call_llm_agent(
            initial_messages, store, top_n, llm_kwargs, None,
            session_ctx=session_ctx,
            progress_cb=lambda lbl: notice_queue.put(f"[dim]  {lbl}[/dim]"),
        )
        short_answer = answer[:600] + "…" if len(answer) > 600 else answer
        notice_queue.put(
            f"[bold]Scheduled task result:[/bold]\n{short_answer}"
        )
    except Exception as exc:
        notice_queue.put(f"[yellow]⚠ Scheduled chat prompt failed: {exc}[/yellow]")
