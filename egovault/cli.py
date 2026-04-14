"""EgoVault CLI — entry point for all commands."""
from __future__ import annotations

import sys

# On Windows the default console codepage (cp1252) can't encode the Unicode
# spinner/box-drawing glyphs Rich uses. Switch the console to UTF-8 and
# reconfigure stdout/stderr before any Rich output is produced.
if sys.platform == "win32":
    import ctypes
    ctypes.windll.kernel32.SetConsoleOutputCP(65001)  # type: ignore[attr-defined]
    ctypes.windll.kernel32.SetConsoleCP(65001)        # type: ignore[attr-defined]
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path
import os
import subprocess
from typing import TYPE_CHECKING

import click
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from egovault.config import configure_logging, get_settings, load_settings

if TYPE_CHECKING:
    from egovault.core.store import VaultStore
    from egovault.core.adapter import BasePlatformAdapter

console = Console()


# ---------------------------------------------------------------------------
# Shared CLI helpers (DRY: ingest loop + progress bar)
# ---------------------------------------------------------------------------

def _make_ingest_progress() -> Progress:
    """Create a consistent Rich Progress bar for ingest commands."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed} records"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )


def _run_ingest_loop(
    adapter: "BasePlatformAdapter",
    source_path: Path,
    store: "VaultStore",
    progress: Progress,
    task_id: object,
    platform: str,
) -> tuple[int, int]:
    """Ingest all records from *adapter*. Returns (inserted, skipped).

    Handles dedup, body back-fill, and ingested_files tracking for every
    record — logic shared by ``ingest``, ``scan``, and ``scan-folder``.
    """
    inserted = skipped = 0
    for record in adapter.ingest(source_path):
        was_new = store.upsert_record(record)
        if was_new:
            inserted += 1
        else:
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
                platform=platform,
                content_hash=record.raw.get("content_hash"),
            )
        progress.advance(task_id)
    return inserted, skipped


@click.group(invoke_without_command=True)
@click.version_option(package_name="egovault")
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=False, dir_okay=False, path_type=str),
    help="Path to egovault.toml (default: auto-detect).",
)
@click.pass_context
def main(ctx: click.Context, config_path: str | None) -> None:
    """EgoVault — local-first personal data vault with LLM enrichment."""
    path = Path(config_path) if config_path else None
    settings = load_settings(path)
    configure_logging(settings)

    # Auto-update check — runs only in interactive terminals, silently skipped
    # when piped or when the update check fails for any reason.
    from egovault.utils.updater import prompt_and_maybe_update, restart
    if prompt_and_maybe_update(console):
        restart(console)
        return  # only reached on Windows (restart calls sys.exit internally)

    if ctx.invoked_subcommand is None:
        ctx.invoke(chat)


@main.command()
@click.argument("source_path", type=click.Path(exists=True))
def ingest(source_path: str) -> None:
    """Ingest a data export from a supported platform."""
    from egovault.core.registry import AdapterRegistry
    from egovault.core.store import VaultStore

    settings = get_settings()
    store = VaultStore(settings.vault_db)
    store.init_db()

    src = Path(source_path).resolve()

    try:
        adapter = AdapterRegistry.get_adapter(src, store=store)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise SystemExit(1) from exc

    platform = adapter.platform_id
    inserted = skipped = 0

    with _make_ingest_progress() as progress:
        task = progress.add_task(f"Ingesting [cyan]{platform}[/cyan]…", total=None)
        inserted, skipped = _run_ingest_loop(adapter, src, store, progress, task, platform)

    store.close()
    console.print(
        f"[green]✓[/green] Ingested [bold]{inserted}[/bold] records from "
        f"[cyan]{platform}[/cyan] ([dim]{skipped} skipped[/dim])"
    )


@main.command("scan")
def scan() -> None:
    """Ingest everything in the configured inbox_dir (default: ./inbox)."""
    from egovault.core.registry import AdapterRegistry
    from egovault.core.store import VaultStore

    settings = get_settings()
    inbox = Path(settings.inbox_dir).resolve()

    if not inbox.exists():
        inbox.mkdir(parents=True, exist_ok=True)
        console.print(f"[yellow]Created inbox folder:[/yellow] {inbox}")

    if not any(inbox.iterdir()):
        console.print(
            f"[yellow]Inbox is empty.[/yellow] Drop files into [cyan]{inbox}[/cyan] then re-run."
        )
        return

    store = VaultStore(settings.vault_db)
    store.init_db()

    try:
        adapter = AdapterRegistry.get_adapter(inbox, store=store)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise SystemExit(1) from exc

    platform = adapter.platform_id
    inserted = skipped = 0

    with _make_ingest_progress() as progress:
        task = progress.add_task(f"Scanning [cyan]{inbox.name}/[/cyan]…", total=None)
        inserted, skipped = _run_ingest_loop(adapter, inbox, store, progress, task, platform)

    store.close()
    console.print(
        f"[green]✓[/green] Scanned [bold]{inserted}[/bold] new records "
        f"([dim]{skipped} skipped[/dim]) from [cyan]{inbox}[/cyan]"
    )


@main.command("scan-folder")
@click.argument("folder", default="", required=False)
@click.option(
    "--list-known",
    is_flag=True,
    default=False,
    help="Print well-known folder aliases for this system and exit.",
)
def scan_folder(folder: str, list_known: bool) -> None:
    """Scan any folder (or well-known alias) once — results are saved but the
    folder is NOT added to the automatic scan configuration.

    \b
    FOLDER may be:
      - A well-known alias:  desktop, documents, downloads, pictures,
                             music, videos, movies, home
      - A tilde path:        ~/notes
      - Any absolute path:   /data/exports  or  C:\\Users\\me\\exports

    \b
    Examples:
      egovault scan-folder desktop
      egovault scan-folder downloads
      egovault scan-folder ~/research/papers
      egovault scan-folder --list-known
    """
    from egovault.adapters.local_inbox import LocalInboxAdapter
    from egovault.core.store import VaultStore
    from egovault.utils.folders import list_known_folders, resolve_folder

    if list_known:
        console.print("[bold]Well-known folder aliases on this system:[/bold]")
        for alias, path in list_known_folders():
            if path is not None:
                console.print(f"  [cyan]{alias:<12}[/cyan] {path}")
            else:
                console.print(f"  [cyan]{alias:<12}[/cyan] [dim]not found[/dim]")
        return

    if not folder:
        console.print("[red]Error:[/red] FOLDER argument is required. Use --list-known to see available aliases.")
        raise SystemExit(1)

    try:
        src = resolve_folder(folder)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise SystemExit(1) from exc

    if not src.exists():
        console.print(f"[red]Error:[/red] Folder does not exist: {src}")
        raise SystemExit(1)
    if not src.is_dir():
        console.print(f"[red]Error:[/red] Path is not a directory: {src}")
        raise SystemExit(1)

    settings = get_settings()
    store = VaultStore(settings.vault_db)
    store.init_db()

    adapter = LocalInboxAdapter(store=store)

    if not adapter.can_handle(src):
        console.print(
            f"[yellow]No supported files found in[/yellow] [cyan]{src}[/cyan]\n"
            f"[dim]Supported: .md .txt .pdf .docx .html .epub .xlsx .pptx[/dim]"
        )
        store.close()
        return

    inserted = skipped = 0

    with _make_ingest_progress() as progress:
        task = progress.add_task(f"Scanning [cyan]{src.name}/[/cyan]…", total=None)
        inserted, skipped = _run_ingest_loop(adapter, src, store, progress, task, "local")

    store.close()
    console.print(
        f"[green]✓[/green] Scanned [bold]{inserted}[/bold] new records "
        f"([dim]{skipped} skipped[/dim]) from [cyan]{src}[/cyan]"
    )
    if inserted == 0 and skipped == 0:
        console.print("[dim]No supported files found.[/dim]")


@main.command()
@click.option("--limit", default=500, show_default=True, help="Max records to enrich per run.")
@click.option("--export", "do_export", is_flag=True, default=False, help="Export Markdown after enrichment.")
def enrich(limit: int, do_export: bool) -> None:
    """Run LLM enrichment over un-enriched records."""
    from egovault.bootstrap import ensure_llama_server
    from egovault.core.enrichment import EnrichmentPipeline
    from egovault.core.store import VaultStore
    from egovault.output.markdown import MarkdownGenerator

    settings = get_settings()
    if not ensure_llama_server(settings, console):
        raise SystemExit(1)

    store = VaultStore(settings.vault_db)
    store.init_db()

    pending = store.get_unenriched_records(limit=limit)
    if not pending:
        console.print("[yellow]No pending records to enrich.[/yellow]")
        store.close()
        return

    pipeline = EnrichmentPipeline(store, settings)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Enriching records…", total=len(pending))
        ok = fail = 0
        for rec in pending:
            if pipeline.enrich_record(rec):
                ok += 1
            else:
                fail += 1
            progress.advance(task)

    console.print(
        f"[green]✓[/green] Enriched [bold]{ok}[/bold] records "
        f"([red]{fail} failed[/red])"
    )

    if do_export:
        gen = MarkdownGenerator(store, settings)
        paths = gen.generate_all()
        console.print(f"[green]✓[/green] Exported [bold]{len(paths)}[/bold] Markdown files to [cyan]{settings.output_dir}[/cyan]")

    store.close()


@main.command("export")
def export_cmd() -> None:
    """Export enriched records to Markdown files."""
    from egovault.core.store import VaultStore
    from egovault.output.markdown import MarkdownGenerator

    settings = get_settings()
    store = VaultStore(settings.vault_db)
    store.init_db()

    gen = MarkdownGenerator(store, settings)
    paths = gen.generate_all()

    store.close()
    console.print(
        f"[green]✓[/green] Exported [bold]{len(paths)}[/bold] Markdown files to "
        f"[cyan]{settings.output_dir}[/cyan]"
    )


@main.command()
def chat() -> None:
    """Start an interactive RAG chat session over the vault."""
    from egovault.bootstrap import ensure_llama_server
    from egovault.chat.session import run_chat_session
    from egovault.core.store import VaultStore

    settings = get_settings()
    if not ensure_llama_server(settings, console):
        raise SystemExit(1)

    store = VaultStore(settings.vault_db)
    store.init_db()
    try:
        run_chat_session(store, settings)
    finally:
        store.close()


@main.command("embed")
@click.option("--model", default="", help="Embedding model override (default: from config).")
@click.option("--limit", default=5000, show_default=True, help="Max records to embed per run.")
def embed_cmd(model: str, limit: int) -> None:
    """Pre-compute dense embeddings for all vault records.

    \b
    Setup (one-time):
      1. Start llama-server with an embedding model and --embedding flag
      2. egovault embed
      3. Set embeddings.enabled = true in egovault.toml

    Re-run after each ingest to keep the index fresh.
    Already-embedded records are skipped automatically.
    """
    from egovault.chat.rag import embed_text
    from egovault.core.store import VaultStore

    settings = get_settings()
    embed_cfg = settings.embeddings
    model_name = model.strip() or embed_cfg.model
    base_url = embed_cfg.base_url.strip() or settings.llm.base_url
    embed_provider = embed_cfg.provider or settings.llm.provider

    store = VaultStore(settings.vault_db)
    store.init_db()

    pending_ids = store.get_unembedded_record_ids(model_name, limit=limit)
    if not pending_ids:
        console.print("[yellow]All records already embedded — nothing to do.[/yellow]")
        store.close()
        return

    console.print(
        f"[dim]Embedding [bold]{len(pending_ids)}[/bold] records with "
        f"[cyan]{model_name}[/cyan] via {base_url} …[/dim]"
    )

    ok = fail = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Embedding with [cyan]{model_name}[/cyan]…", total=len(pending_ids))
        for record_id in pending_ids:
            text = store.get_record_text_by_id(record_id)
            if not text.strip():
                store.upsert_embedding(record_id, model_name, [0.0])  # sentinel for empty records
                ok += 1
                progress.advance(task)
                continue
            try:
                vec = embed_text(text, base_url, model_name)
                store.upsert_embedding(record_id, model_name, vec)
                ok += 1
            except Exception as exc:
                fail += 1
                import logging as _log
                _log.getLogger(__name__).debug("Embedding failed for %s: %s", record_id, exc)
            progress.advance(task)

    store.close()
    console.print(
        f"[green]✓[/green] Embedded [bold]{ok}[/bold] records "
        f"([red]{fail} failed[/red]) using [cyan]{model_name}[/cyan]"
    )
    if fail == 0:
        console.print(
            "[dim]Set [bold]embeddings.enabled = true[/bold] in egovault.toml "
            "to activate semantic search.[/dim]"
        )

    # ── HyPE: generate question texts + embed them ───────────────────────────
    if not getattr(embed_cfg, "hype_enabled", False):
        return

    from egovault.core.enrichment import _generate_hype_questions
    from egovault.core.store import VaultStore as _VS

    hype_store = _VS(settings.vault_db)
    hype_store.init_db()

    pending_hype = hype_store.get_records_without_hype_questions(model_name, limit=limit)
    if not pending_hype:
        console.print("[yellow]All records already have HyPE question embeddings.[/yellow]")
        hype_store.close()
        return

    # Fetch full records so we have body text for question generation
    id_set = set(pending_hype)
    all_recs = hype_store.get_records()
    pending_recs = [r for r in all_recs if r.id in id_set]

    console.print(
        f"[dim]Generating HyPE question embeddings for [bold]{len(pending_recs)}[/bold] records…[/dim]"
    )

    hok = hfail = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        htask = progress.add_task("HyPE question embeddings…", total=len(pending_recs))
        llm = settings.llm
        for rec in pending_recs:
            questions = _generate_hype_questions(record=rec, llm=llm)
            if not questions:
                hfail += 1
                progress.advance(htask)
                continue
            record_ok = True
            for q in questions:
                try:
                    vec = embed_text(q, base_url, model_name)
                    hype_store.upsert_question_embedding(rec.id, model_name, q, vec)
                except Exception as exc:
                    import logging as _log2
                    _log2.getLogger(__name__).debug("HyPE embed failed for %s: %s", rec.id, exc)
                    record_ok = False
            if record_ok:
                hok += 1
            else:
                hfail += 1
            progress.advance(htask)

    hype_store.close()
    console.print(
        f"[green]✓[/green] HyPE question embeddings: [bold]{hok}[/bold] records "
        f"([red]{hfail} failed[/red])"
    )


@main.command("chunk")
@click.option("--model", default="", help="Embedding model override (default: from config).")
@click.option("--limit", default=5000, show_default=True, help="Max records to chunk+embed per run.")
def chunk_cmd(model: str, limit: int) -> None:
    """Build sentence-window chunk embeddings (Sentence Window Retrieval).

    \b
    Splits each record's body into overlapping sentence windows and embeds
    every window separately.  These fine-grained embeddings enable sub-record
    semantic matching — the query is matched against window embeddings and the
    winning window's surrounding context is returned to the LLM.

    \b
    Setup:
      1. Set sentence_window.enabled = true in egovault.toml
      2. egovault chunk          (builds chunk embeddings; safe to re-run)
      3. egovault embed          (ensures full-record embeddings also exist)

    Already-chunked records are skipped automatically.
    """
    from egovault.chat.rag import embed_text
    from egovault.core.store import VaultStore
    from egovault.utils.chunking import make_sentence_windows

    settings = get_settings()
    sw_cfg = settings.sentence_window
    if not sw_cfg.enabled:
        console.print(
            "[yellow]sentence_window.enabled = false[/yellow] in egovault.toml — nothing to do.\n"
            "[dim]Set [bold]sentence_window.enabled = true[/bold] to activate.[/dim]"
        )
        return

    embed_cfg = settings.embeddings
    model_name = model.strip() or embed_cfg.model
    base_url = embed_cfg.base_url.strip() or settings.llm.base_url
    embed_provider = embed_cfg.provider or settings.llm.provider

    store = VaultStore(settings.vault_db)
    store.init_db()

    pending_ids = store.get_records_without_chunks(model_name, limit=limit)
    if not pending_ids:
        console.print("[yellow]All records already have sentence-window chunks — nothing to do.[/yellow]")
        store.close()
        return

    id_set = set(pending_ids)
    all_recs = store.get_records()
    pending_recs = [r for r in all_recs if r.id in id_set]

    console.print(
        f"[dim]Building sentence-window embeddings for [bold]{len(pending_recs)}[/bold] records "
        f"(window={sw_cfg.window_size}, overlap={sw_cfg.overlap})…[/dim]"
    )

    ok = fail = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Chunking records…", total=len(pending_recs))
        for rec in pending_recs:
            body = rec.body or ""
            windows = make_sentence_windows(
                body,
                window_size=sw_cfg.window_size,
                overlap=sw_cfg.overlap,
            )
            if not windows:
                # Store a single sentinel chunk so we don't re-process this record
                try:
                    vec = embed_text(body[:200] or "empty", base_url, model_name)
                    store.upsert_chunk_embedding(rec.id, model_name, 0, body[:200] or "", vec)
                    ok += 1
                except Exception as exc:
                    import logging as _log
                    _log.getLogger(__name__).debug("Chunk embed failed for %s: %s", rec.id, exc)
                    fail += 1
                progress.advance(task)
                continue

            record_ok = True
            for cidx, chunk_text in windows:
                try:
                    vec = embed_text(chunk_text, base_url, model_name)
                    store.upsert_chunk_embedding(rec.id, model_name, cidx, chunk_text, vec)
                except Exception as exc:
                    import logging as _log2
                    _log2.getLogger(__name__).debug(
                        "Chunk embed failed for %s[%d]: %s", rec.id, cidx, exc
                    )
                    record_ok = False
            if record_ok:
                ok += 1
            else:
                fail += 1
            progress.advance(task)

    store.close()
    console.print(
        f"[green]✓[/green] Chunked [bold]{ok}[/bold] records "
        f"([red]{fail} failed[/red]) using [cyan]{model_name}[/cyan]"
    )
    console.print(
        "[dim]Run [bold]egovault embed[/bold] to also build full-record embeddings, "
        "then set [bold]embeddings.enabled = true[/bold] to activate retrieval.[/dim]"
    )


@main.command("context")
@click.option("--limit", default=5000, show_default=True, help="Max records to process per run.")
def context_cmd(limit: int) -> None:
    """Generate contextual prefixes for vault records (Contextual Retrieval).

    \b
    For each record that does not yet have a context blurb, calls the LLM to
    generate a 2-3 sentence situating description and stores it as
    contextual_body.  The embed command then uses contextual_body instead of
    body when building dense vectors.

    \b
    Setup:
      1. Set embeddings.contextual_enabled = true in egovault.toml
      2. egovault context          (generates prefixes; safe to re-run)
      3. egovault embed            (re-embeds using contextual_body)

    Already-contextualized records are skipped automatically.
    """
    from egovault.bootstrap import ensure_llama_server
    from egovault.core.enrichment import EnrichmentPipeline
    from egovault.core.store import VaultStore

    settings = get_settings()
    if not getattr(settings.embeddings, "contextual_enabled", False):
        console.print(
            "[yellow]contextual_enabled = false[/yellow] in egovault.toml — nothing to do.\n"
            "[dim]Set [bold]embeddings.contextual_enabled = true[/bold] to enable.[/dim]"
        )
        return

    if not ensure_llama_server(settings, console):
        raise SystemExit(1)

    store = VaultStore(settings.vault_db)
    store.init_db()

    pending_ids = store.get_uncontextualized_record_ids(limit=limit)
    if not pending_ids:
        console.print("[yellow]All records already have context prefixes — nothing to do.[/yellow]")
        store.close()
        return

    console.print(
        f"[dim]Generating context prefixes for [bold]{len(pending_ids)}[/bold] records…[/dim]"
    )

    pipeline = EnrichmentPipeline(store, settings)

    # Fetch full records for the pending IDs
    id_set = set(pending_ids)
    all_records = store.get_records()
    pending = [r for r in all_records if r.id in id_set]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Contextualizing records…", total=len(pending))
        ok = fail = 0
        for rec in pending:
            if pipeline.contextualize_record(rec):
                ok += 1
            else:
                fail += 1
            progress.advance(task)

    store.close()
    console.print(
        f"[green]✓[/green] Contextualized [bold]{ok}[/bold] records "
        f"([red]{fail} skipped/failed[/red])"
    )
    console.print(
        "[dim]Run [bold]egovault embed[/bold] to rebuild embeddings using the new context prefixes.[/dim]"
    )


# ---------------------------------------------------------------------------
# Gmail OAuth commands
# ---------------------------------------------------------------------------

_GMAIL_SETUP_GUIDE = """\
[bold cyan]Gmail OAuth Setup[/bold cyan]
─────────────────────────────────────────────
EgoVault needs a Google OAuth2 credentials file to access your Gmail.
This is a one-time setup — EgoVault will open your browser automatically.

[bold]Steps to get credentials:[/bold]

  1. Open  [link=https://console.cloud.google.com/]https://console.cloud.google.com/[/link]
  2. Create a new project  (or select an existing one)
  3. Enable the Gmail API:
       APIs & Services → Library → search "Gmail API" → Enable
  4. Create credentials:
       APIs & Services → Credentials → Create Credentials
       → OAuth client ID → Application type: [bold]Desktop app[/bold]
       → Name: EgoVault (or anything) → Create
  5. Click [bold]Download JSON[/bold] and save it anywhere (e.g. ~/Downloads/)

Then run:
  [bold]egovault gmail-auth --credentials ~/Downloads/client_secret_*.json[/bold]
"""


@main.command("gmail-auth")
@click.option(
    "--credentials",
    "creds_path",
    default=None,
    type=click.Path(dir_okay=False),
    help="Path to client_secret_*.json from Google Cloud Console.",
    show_default=False,
)
def gmail_auth_cmd(creds_path: str | None) -> None:
    """Authorise EgoVault to read your Gmail (one-time OAuth2 setup).

    \b
    Opens your browser for Google sign-in, then saves a local token so
    that `egovault gmail-sync` works without logging in again.

    \b
    Steps:
      1. Download OAuth credentials from Google Cloud Console
         (Desktop app type — see output below for guide if needed)
      2. egovault gmail-auth --credentials ~/Downloads/client_secret_*.json
      3. egovault gmail-sync
    """
    try:
        from egovault.utils.gmail_auth import _require_deps  # noqa: F401 — fast check
        _require_deps()
    except ImportError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise SystemExit(1) from exc

    from egovault.utils.gmail_auth import get_token_path, run_oauth_flow

    settings = get_settings()
    data_dir = Path(settings.vault_db).parent
    token_path = get_token_path(data_dir)

    # If no credentials path supplied, print guide and prompt.
    if not creds_path:
        console.print(_GMAIL_SETUP_GUIDE)
        creds_path = click.prompt(
            "Path to credentials JSON",
            default="",
            show_default=False,
        ).strip()
        if not creds_path:
            console.print("[red]No credentials path provided — aborting.[/red]")
            raise SystemExit(1)

    creds_file = Path(creds_path).expanduser().resolve()
    if not creds_file.exists():
        console.print(f"[red]Error:[/red] File not found: {creds_file}")
        raise SystemExit(1)

    console.print(
        "[dim]Opening your browser for Google sign-in…  "
        "(grant the [bold]Gmail read-only[/bold] permission)[/dim]"
    )

    try:
        run_oauth_flow(creds_file, token_path)
    except Exception as exc:
        console.print(f"[red]OAuth error:[/red] {exc}")
        raise SystemExit(1) from exc

    console.print(f"[green]✓[/green] Authenticated — token saved to [cyan]{token_path}[/cyan]")
    console.print("[dim]Run [bold]egovault gmail-sync[/bold] to import your emails.[/dim]")


@main.command("gmail-sync")
@click.option(
    "--query",
    default="",
    help=(
        "Gmail search query (default: skip spam and trash).  "
        "Examples:  'in:inbox'   'from:boss@acme.com'   'is:important'"
    ),
)
@click.option(
    "--since",
    default="",
    help=(
        "Only fetch emails after this date (YYYY-MM-DD).  "
        "Appended to --query as 'after:YYYY/MM/DD'.  "
        "If omitted, the date of the last successful sync is used automatically."
    ),
)
@click.option(
    "--max-results",
    default=500,
    show_default=True,
    help="Maximum number of emails to fetch per run.",
)
def gmail_sync_cmd(query: str, since: str, max_results: int) -> None:
    """Import emails from Gmail via OAuth2.

    \b
    Requires a one-time setup:  egovault gmail-auth

    \b
    Examples:
      egovault gmail-sync                          # up to 500 recent non-spam emails
      egovault gmail-sync --max-results 2000       # fetch more
      egovault gmail-sync --since 2025-06-01       # only emails after a date
      egovault gmail-sync --query "in:inbox"       # inbox only
      egovault gmail-sync --query "from:alice@example.com"

    Re-running is always safe — duplicate emails are silently skipped.
    """
    try:
        from egovault.utils.gmail_auth import _require_deps  # noqa: F401
        _require_deps()
    except ImportError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise SystemExit(1) from exc

    from egovault.adapters.gmail_api import GmailApiAdapter
    from egovault.core.store import VaultStore
    from egovault.utils.gmail_auth import get_token_path, load_credentials

    settings = get_settings()
    data_dir = Path(settings.vault_db).parent
    token_path = get_token_path(data_dir)

    # Check auth before touching the store.
    if load_credentials(token_path) is None:
        console.print(
            "[red]Error:[/red] Not authenticated with Gmail.\n"
            "Run:  [bold]egovault gmail-auth[/bold]"
        )
        raise SystemExit(1)

    store = VaultStore(settings.vault_db)
    store.init_db()

    # Build effective query, incorporating --since or the last sync date.
    effective_query = query.strip()
    since_date = since.strip()
    if not since_date:
        # Fall back to the stored last-sync date.
        since_date = store.get_setting("gmail_last_sync") or ""
    if since_date:
        # Normalise YYYY-MM-DD → YYYY/MM/DD for Gmail API after: operator.
        gmail_date = since_date.replace("-", "/")
        after_clause = f"after:{gmail_date}"
        effective_query = (
            f"{effective_query} {after_clause}".strip()
            if effective_query
            else after_clause
        )
    if not effective_query:
        effective_query = "-in:spam -in:trash"

    console.print(
        f"[dim]Fetching up to [bold]{max_results}[/bold] emails "
        f"(query: [italic]{effective_query}[/italic])…[/dim]"
    )

    adapter = GmailApiAdapter(store=store)
    inserted = skipped = 0

    with _make_ingest_progress() as progress:
        task = progress.add_task("Syncing Gmail…", total=None)

        def _on_progress(count: int) -> None:
            progress.update(task, completed=count)

        try:
            for record in adapter.ingest_from_api(
                token_path=token_path,
                query=effective_query,
                max_results=max_results,
                progress_callback=_on_progress,
            ):
                was_new = store.upsert_record(record)
                if was_new:
                    inserted += 1
                else:
                    skipped += 1
        except RuntimeError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            store.close()
            raise SystemExit(1) from exc

    # Record today as the last-sync date for incremental future runs.
    from datetime import date
    store.set_setting("gmail_last_sync", date.today().strftime("%Y-%m-%d"))
    store.close()

    console.print(
        f"[green]✓[/green] Gmail sync complete — "
        f"[bold]{inserted}[/bold] new records "
        f"([dim]{skipped} already in vault[/dim])"
    )
    if inserted > 0:
        console.print(
            "[dim]Run [bold]egovault enrich[/bold] to summarise the new emails "
            "with your local LLM.[/dim]"
        )


def _ensure_wan_key(data_dir: Path) -> "Path | None":
    """Return path to the localhost.run SSH identity key, generating it if absent.

    The key is stored at <data_dir>/wan_id_ed25519.  On first
    generation the public key and the upload URL are printed so the user can
    register it for a persistent tunnel subdomain.
    """
    import subprocess

    key_path = data_dir / "wan_id_ed25519"
    pub_path = data_dir / "wan_id_ed25519.pub"

    if not key_path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                [
                    "ssh-keygen", "-t", "ed25519",
                    "-f", str(key_path),
                    "-N", "",
                    "-C", "egovault-localhost-run",
                ],
                check=True,
                capture_output=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            return None

        pub_key = pub_path.read_text().strip()
        console.print(
            "[bold green]SSH key generated[/bold green] "
            f"at [cyan]{key_path}[/cyan]\n"
            "Upload your public key at [link=https://admin.localhost.run]https://admin.localhost.run[/link] "
            "to get a [bold]persistent tunnel URL[/bold] that never changes.\n"
            f"[dim]{pub_key}[/dim]"
        )
    return key_path


def _start_wan_tunnel(
    port: int,
    key_path: "Path | None" = None,
) -> "tuple[str | None, subprocess.Popen | None]":
    """Spawn a localhost.run SSH tunnel and return (url, process).

    Reads stdout of the ssh process until the lhr.life URL appears (max 15 s).
    Returns (None, proc) if SSH is unavailable or the URL never appears.
    When *key_path* is provided the key is passed via ``-i`` and the plain
    ``localhost.run`` host is used (no ``nokey@``) so the subdomain persists
    across sessions once the key is registered at https://admin.localhost.run.
    """
    import re
    import subprocess
    import threading

    if key_path and key_path.exists():
        host = "localhost.run"
        identity_args = ["-i", str(key_path)]
    else:
        host = "nokey@localhost.run"
        identity_args = []

    try:
        proc = subprocess.Popen(
            [
                "ssh", "-o", "StrictHostKeyChecking=no",
                *identity_args,
                "-R", f"80:localhost:{port}",
                host,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except FileNotFoundError:
        return None, None

    found: list[str] = []

    def _reader() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            m = re.search(r"https://\S+\.lhr\.life", line)
            if m:
                found.append(m.group(0))
                break

    t = threading.Thread(target=_reader, daemon=True)
    t.start()
    t.join(timeout=15)

    return (found[0] if found else None), proc


@main.command("web-password")
def web_password_cmd() -> None:
    """Set or update the WAN access password for the Streamlit web UI.

    \b
    Prompts for a password, hashes it with SHA-256, and saves the hash to
    data/wan.password.  No plaintext is stored.
    Run this once before using egovault web (--wan requires a password).
    """
    import getpass
    import hashlib

    from egovault.config import get_settings

    pwd1 = getpass.getpass("New WAN password: ")
    if not pwd1:
        console.print("[red]Password cannot be empty.[/red]")
        raise SystemExit(1)
    pwd2 = getpass.getpass("Confirm password: ")
    if pwd1 != pwd2:
        console.print("[red]Passwords do not match.[/red]")
        raise SystemExit(1)

    pw_hash = "sha256:" + hashlib.sha256(pwd1.encode()).hexdigest()

    settings = get_settings()
    data_dir = Path(settings.vault_db).parent
    data_dir.mkdir(parents=True, exist_ok=True)
    pw_file = data_dir / "wan.password"
    pw_file.write_text(pw_hash + "\n", encoding="utf-8")

    console.print(f"[bold green]WAN password saved[/bold green] to [cyan]{pw_file}[/cyan]")
    console.print("Run [bold]egovault web[/bold] to start the web UI with the tunnel.")


@main.command("web")
@click.option("--host", default="localhost", show_default=True, help="Address to bind the Streamlit server.")
@click.option("--port", default=8501, show_default=True, help="Port to bind the Streamlit server.")
@click.option("--wan/--no-wan", default=True, show_default=True, help="Auto-start localhost.run SSH tunnel for WAN access.")
@click.pass_context
def web_cmd(ctx: click.Context, host: str, port: int, wan: bool) -> None:
    """Launch the EgoVault Streamlit browser UI.

    \b
    Opens a chat interface in your browser for querying the local vault.
    Included in: pip install egovault

    \b
    Examples:
      egovault web
      egovault web --port 8888
      egovault web --host 0.0.0.0 --port 8501
      egovault web --no-wan
    """
    import os
    import subprocess
    import sys

    from egovault.bootstrap import ensure_llama_server

    try:
        import streamlit  # noqa: F401
    except ImportError:
        console.print("[red]Streamlit is not installed.[/red]")
        console.print("Run:  pip install egovault")
        console.print("If you removed it manually, run:  pip install streamlit")
        raise SystemExit(1)

    settings = get_settings()
    if not ensure_llama_server(settings, console):
        raise SystemExit(1)

    # ── WAN tunnel ────────────────────────────────────────────────────────────
    wan_proc = None
    if wan:
        if not settings.wan_password_hash:
            console.print(
                "[bold red]WAN tunnel refused:[/bold red] no WAN password set.\n"
                "Run [bold]egovault web-password[/bold] to set one, then try again."
            )
            raise SystemExit(1)
        data_dir = Path(settings.vault_db).parent
        wan_key = _ensure_wan_key(data_dir)
        console.print("[dim]Starting WAN tunnel via localhost.run…[/dim]")
        wan_url, wan_proc = _start_wan_tunnel(port, key_path=wan_key)
        if wan_url:
            os.environ["EGOVAULT_WAN_URL"] = wan_url
            console.print(f"[bold green]WAN URL:[/bold green] {wan_url}")
        else:
            console.print("[yellow]WAN tunnel unavailable (ssh not found or localhost.run unreachable).[/yellow]")

    app_path = Path(__file__).parent / "frontends" / "web.py"
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.address", host,
        "--server.port", str(port),
        "--server.headless", "false",
    ]

    # Spawn the web server detached so we can run the TUI concurrently.
    if sys.platform == "win32":
        web_proc = subprocess.Popen(
            cmd,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            close_fds=True,
        )
    else:
        web_proc = subprocess.Popen(cmd, start_new_session=True, close_fds=True)

    console.print(f"[bold cyan]EgoVault Web UI[/bold cyan] — starting at http://{host}:{port} (background)")
    console.print("[dim]TUI starting — Ctrl+C stops both[/dim]")

    try:
        ctx.invoke(chat)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        if web_proc.poll() is None:
            web_proc.terminate()
        if wan_proc is not None and wan_proc.poll() is None:
            wan_proc.terminate()


@main.command("mcp")
def mcp_cmd() -> None:
    """Start the EgoVault MCP server (stdio transport).

    \b
    Exposes the vault as MCP tools for AnythingLLM and other MCP clients:
      search_vault, chat, vault_stats, list_platforms, get_gems, record_feedback

    \b
    Connect from AnythingLLM:
      Settings → AI Providers → MCP → add a Custom Agent tool:
        Command: egovault mcp
        (or full path: .venv/Scripts/egovault mcp on Windows)

    \b
    Included in: pip install egovault
    """
    try:
        from egovault.chat.web import launch  # noqa: F401 — checks import only
    except ImportError:
        console.print("[red]MCP SDK is not installed.[/red]", err=True)
        console.print("Run:  pip install egovault", err=True)
        console.print("If you removed it manually, run:  pip install mcp", err=True)
        raise SystemExit(1)

    from egovault.bootstrap import ensure_llama_server
    from egovault.core.store import VaultStore

    settings = get_settings()

    # stdout is owned by the MCP protocol — ALL console output must go to stderr.
    err_console = Console(stderr=True)

    if not ensure_llama_server(settings, err_console):
        raise SystemExit(1)

    store = VaultStore(settings.vault_db)
    store.init_db()

    err_console.print(
        "[bold cyan]EgoVault MCP Server[/bold cyan] — listening on stdio  (Ctrl+C to stop)"
    )
    err_console.print(
        "[dim]Connect from AnythingLLM: Settings → AI Providers → MCP → Custom Agent[/dim]"
    )

    from egovault.chat.web import launch
    try:
        launch(store, settings)
    finally:
        store.close()


@main.command("telegram")
@click.pass_context
def telegram_cmd(ctx: click.Context) -> None:
    """Start the EgoVault Telegram bot.

    \b
    Chat with your vault from any device via Telegram.
    Uses long-polling — no public server or port forwarding needed.
    On first run the setup wizard guides you through token + chat ID in ~60 s.

    \b
    The bot supports:
      /start   — welcome message
      /help    — command list
      /clear   — reset conversation history
      /sources — sources used in last answer
      /status  — LLM server and vault stats
      Any text — hybrid RAG query over the vault
    """
    try:
        import telegram  # noqa: F401
    except ImportError:
        console.print("[red]python-telegram-bot is not installed.[/red]")
        console.print("Run:  pip install 'python-telegram-bot>=20'")
        raise SystemExit(1)

    from egovault.bootstrap import ensure_llama_server
    from egovault.frontends.telegram import launch as tg_launch, run_setup_wizard
    from egovault.config import reset_settings

    settings = get_settings()

    # ── First-run wizard: fire when token or chat IDs are missing ─────────────
    if not settings.telegram.token or not settings.telegram.allowed_chat_ids:
        result = run_setup_wizard(console=console)
        if result is None:
            raise SystemExit(0)
        # Reload settings now that egovault.toml has been updated
        reset_settings()
        settings = get_settings()

    if not ensure_llama_server(settings, console):
        raise SystemExit(1)

    # ── Single-instance guard ─────────────────────────────────────────────────
    import psutil as _psutil
    _current_pid = os.getpid()
    for _proc in _psutil.process_iter(["pid", "cmdline"]):
        try:
            if _proc.pid == _current_pid:
                continue
            _cmd = " ".join(_proc.info.get("cmdline") or [])
            if ("egovault" in _cmd or "ego" in _cmd) and "telegram" in _cmd:
                console.print(
                    f"[yellow]Stopping existing Telegram bot (PID {_proc.pid})…[/yellow]"
                )
                _proc.terminate()
                try:
                    _proc.wait(timeout=3)
                except Exception:
                    _proc.kill()
        except (_psutil.NoSuchProcess, _psutil.AccessDenied):
            pass

    console.print(
        "[bold cyan]EgoVault Telegram Bot[/bold cyan] — polling for messages  (Ctrl+C to stop)"
    )
    console.print(
        f"[dim]Allowed chat IDs: {settings.telegram.allowed_chat_ids}[/dim]"
    )

    import threading as _threading

    # Run the Telegram bot as a daemon thread so it stops when the TUI exits.
    tg_errors: list[Exception] = []

    def _tg_thread() -> None:
        try:
            tg_launch(settings)
        except ValueError as exc:
            tg_errors.append(exc)

    bot_thread = _threading.Thread(target=_tg_thread, daemon=True, name="egovault-telegram")
    bot_thread.start()

    console.print("[bold cyan]EgoVault Telegram Bot[/bold cyan] — polling in background")
    console.print("[dim]TUI starting — Ctrl+C stops both[/dim]")

    try:
        ctx.invoke(chat)
    except (KeyboardInterrupt, SystemExit):
        pass

    if tg_errors:
        console.print(f"[red]Telegram bot error:[/red] {tg_errors[0]}")


# ---------------------------------------------------------------------------
# Telegram history sync commands  (MTProto via Telethon)
# ---------------------------------------------------------------------------

_TELEGRAM_AUTH_GUIDE = """\
[bold cyan]Telegram API Setup[/bold cyan]
─────────────────────────────────────────────
You need a free Telegram API app (one-time):

[bold]Step 1[/bold] — open this URL on your phone or desktop:
"""

_TELEGRAM_AUTH_GUIDE_STEPS = """\

  In the my.telegram.org page:
    1. Log in with your Telegram phone number
    2. Click [bold]API development tools[/bold]
    3. Create an application (any name and short name)
    4. Copy [bold]App api_id[/bold] (a number) and [bold]App api_hash[/bold] (hex string)
"""


@main.command("telegram-auth")
def telegram_auth_cmd() -> None:
    """Authenticate EgoVault with your Telegram account (one-time setup).

    \b
    Guides you through getting api_id + api_hash from my.telegram.org/apps
    and authenticates with your phone number + Telegram verification code.
    Saves a local session file so telegram-sync works without logging in again.

    \b
    After this, run:  egovault telegram-sync
    """
    try:
        import telethon  # noqa: F401
    except ImportError:
        console.print("[red]Telethon is not installed.[/red]")
        console.print("Run:  pip install telethon")
        raise SystemExit(1)

    from egovault.adapters.telegram_history import run_auth
    from egovault.chat.telegram_bot import _print_qr
    from egovault.utils.telegram_api import get_session_path, save_credentials

    settings = get_settings()
    data_dir = Path(settings.vault_db).parent
    data_dir.mkdir(parents=True, exist_ok=True)

    _url = "https://my.telegram.org/apps"
    console.print(_TELEGRAM_AUTH_GUIDE)
    _print_qr(_url)
    console.print(f"  Or open: [bold cyan]{_url}[/bold cyan]")
    console.print(_TELEGRAM_AUTH_GUIDE_STEPS)

    # Collect api_id + api_hash
    while True:
        api_id_raw = click.prompt("api_id (numbers only)").strip()
        try:
            api_id = int(api_id_raw)
            break
        except ValueError:
            console.print("[red]api_id must be a number.[/red]")

    api_hash = click.prompt("api_hash").strip()
    if not api_hash:
        console.print("[red]api_hash cannot be empty.[/red]")
        raise SystemExit(1)

    phone = click.prompt("Your phone number (international format, e.g. +385991234567)").strip()
    if not phone:
        console.print("[red]Phone number cannot be empty.[/red]")
        raise SystemExit(1)

    session_path = get_session_path(data_dir)

    console.print("\n[dim]Connecting to Telegram…[/dim]")

    def _ask_code() -> str:
        return click.prompt("Verification code sent to your Telegram app").strip()

    def _ask_password() -> str:
        import getpass
        return getpass.getpass("Two-step verification password (leave blank if none): ")

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
        raise SystemExit(1)

    save_credentials(data_dir, api_id, api_hash, phone)
    console.print(
        f"[green]✓[/green] Authenticated as [bold]{display_name}[/bold]\n"
        f"[dim]Session saved to [cyan]{session_path}.session[/cyan][/dim]\n"
        f"[dim]Credentials saved to [cyan]{data_dir / 'telegram_api.json'}[/cyan][/dim]\n\n"
        "Run [bold]egovault telegram-sync[/bold] to import your messages."
    )


@main.command("telegram-sync")
@click.option(
    "--since",
    default="",
    help="Only fetch messages after this date (YYYY-MM-DD). Defaults to last sync date.",
    show_default=False,
)
@click.option(
    "--max-messages",
    default=5000,
    show_default=True,
    help="Maximum messages to fetch per dialog.",
)
@click.option(
    "--include-channels",
    is_flag=True,
    default=False,
    help="Also sync broadcast channels (disabled by default — can be very large).",
)
def telegram_sync_cmd(since: str, max_messages: int, include_channels: bool) -> None:
    """Import Telegram message history into the vault.

    \b
    Requires a one-time setup:  egovault telegram-auth

    \b
    Examples:
      egovault telegram-sync
      egovault telegram-sync --since 2025-01-01
      egovault telegram-sync --max-messages 2000
      egovault telegram-sync --include-channels
    """
    try:
        import telethon  # noqa: F401
    except ImportError:
        console.print("[red]Telethon is not installed.[/red]")
        console.print("Run:  pip install telethon")
        raise SystemExit(1)

    from egovault.adapters.telegram_history import TelegramHistoryAdapter
    from egovault.core.store import VaultStore
    from egovault.utils.telegram_api import get_session_path, load_credentials

    settings = get_settings()
    data_dir = Path(settings.vault_db).parent

    creds = load_credentials(data_dir)
    if creds is None:
        console.print(
            "[red]Telegram is not connected.[/red]\n"
            "Run [bold]egovault telegram-auth[/bold] to set up your account first."
        )
        raise SystemExit(1)

    session_path = get_session_path(data_dir)
    if not session_path.with_suffix(".session").exists():
        console.print(
            "[red]Telegram session file not found.[/red]\n"
            "Run [bold]egovault telegram-auth[/bold] to authenticate first."
        )
        raise SystemExit(1)

    store = VaultStore(settings.vault_db)
    store.init_db()

    # Use stored last-sync date if --since not given
    if not since:
        since = store.get_setting("telegram_last_sync") or ""

    if since:
        console.print(f"[dim]Fetching messages since [bold]{since}[/bold]…[/dim]")
    else:
        console.print("[dim]Fetching all message history (first run — may take a while)…[/dim]")

    adapter = TelegramHistoryAdapter(store=store)

    inserted = skipped = total_fetched = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed} messages"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("[cyan]Syncing Telegram messages…", total=None)

        def _on_progress(fetched: int, estimated: int) -> None:
            progress.update(
                task,
                completed=fetched,
                total=max(estimated, fetched),
                description=(
                    f"[cyan]Syncing Telegram…[/cyan] "
                    f"[dim]({inserted} new / {skipped} seen)[/dim]"
                ),
            )

        for record in adapter.ingest_from_api(
            api_id=creds["api_id"],
            api_hash=creds["api_hash"],
            phone=creds["phone"],
            session_path=session_path,
            since=since,
            max_messages=max_messages,
            skip_channels=not include_channels,
            progress_callback=_on_progress,
        ):
            total_fetched += 1
            was_new = store.upsert_record(record)
            if was_new:
                inserted += 1
            else:
                skipped += 1

    # Save last-sync date
    from datetime import date as _date
    store.set_setting("telegram_last_sync", _date.today().strftime("%Y-%m-%d"))
    store.close()

    console.print(
        f"[green]✓[/green] Telegram sync — "
        f"[bold]{inserted}[/bold] new  [dim]({skipped} already in vault)[/dim]"
    )
    if inserted > 0:
        console.print(
            "[dim]New messages are searchable now. "
            "Run [bold]egovault enrich[/bold] to summarise them.[/dim]"
        )
