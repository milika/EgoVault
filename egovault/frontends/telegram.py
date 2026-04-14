"""EgoVault Telegram bot — chat with your vault from any device.

Launch with:  egovault telegram

Quick setup (5 min):
  1. Message @BotFather on Telegram → /newbot → copy the token
  2. Run: egovault telegram          (wizard auto-captures your chat ID)

The bot uses long-polling (no public server or port forwarding needed).
All queries run through the same full RAG pipeline as the Streamlit web UI.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from egovault.config import Settings

logger = logging.getLogger(__name__)

from egovault.agent.commands import handle_command as _handle_command  # noqa: E402
from egovault.agent.session import AgentSession  # noqa: E402

_chat_histories: dict[int, list[dict]] = {}    # chat_id → conversation turns
_chat_sources: dict[int, list[str]] = {}        # chat_id → last answer sources
_chat_last_files: dict[int, str] = {}           # chat_id → last file path from agent

_TG_MAX_LEN = 4096   # Telegram hard limit per message


# ---------------------------------------------------------------------------
# First-run interactive setup wizard
# ---------------------------------------------------------------------------

def _tg_api(token: str, method: str, params: dict | None = None, timeout: int = 10) -> dict:
    """Call a Telegram Bot API method and return the parsed JSON response."""
    url = f"https://api.telegram.org/bot{token}/{method}"
    if params:
        data = json.dumps(params).encode()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    else:
        req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return json.loads(resp.read().decode())


def _save_telegram_config(token: str, chat_id: int) -> bool:
    """Write token and chat_id into egovault.toml in-place.

    Looks for the [telegram] section and patches the two lines.
    If the section doesn't exist it is appended at the end.
    Returns True on success.
    """
    toml_paths = [
        Path("egovault.toml"),
        Path.home() / ".config" / "egovault" / "egovault.toml",
    ]
    toml_file: Path | None = next((p for p in toml_paths if p.exists()), None)

    entry = (
        f"\n[telegram]\n"
        f'token            = "{token}"\n'
        f"allowed_chat_ids = [{chat_id}]\n"
        f"top_n            = 10\n"
    )

    if toml_file is None:
        # No config file yet — create minimal one
        toml_file = Path("egovault.toml")
        toml_file.write_text(entry, encoding="utf-8")
        return True

    text = toml_file.read_text(encoding="utf-8")

    # Replace existing [telegram] section values
    if "[telegram]" in text:
        text = re.sub(
            r'(^\s*token\s*=\s*).*$',
            f'\\1"{token}"',
            text, flags=re.MULTILINE,
        )
        text = re.sub(
            r'(^\s*allowed_chat_ids\s*=\s*).*$',
            f'\\1[{chat_id}]',
            text, flags=re.MULTILINE,
        )
    else:
        # Append section
        text = text.rstrip() + "\n" + entry

    toml_file.write_text(text, encoding="utf-8")
    return True


def _print_qr(url: str) -> None:
    """Print an ASCII QR code for *url* to the terminal.

    Uses the ``qrcode`` library if available; silently skips otherwise so the
    wizard still works without it (the URL is always printed as fallback).
    """
    try:
        import qrcode  # type: ignore[import-untyped]
        qr = qrcode.QRCode(border=2)
        qr.add_data(url)
        qr.make(fit=True)
        qr.print_ascii(invert=True)   # invert=True: dark modules → light (better on dark terminals)
    except Exception:  # noqa: BLE001
        pass  # qrcode not installed or terminal too narrow — caller shows plain URL


def run_setup_wizard(console=None) -> tuple[str, int] | None:  # type: ignore[type-arg]
    """Interactive first-run wizard.  Returns (token, chat_id) on success, None on abort.

    Steps:
      1. Show @BotFather instructions
      2. Prompt for bot token
      3. Validate token via getMe API
      4. Tell user to send /start to the bot
      5. Poll getUpdates until a message arrives (max 90 s) to capture chat_id
      6. Save both to egovault.toml
    """
    import click

    _print = console.print if console else print

    _botfather_url = "https://t.me/BotFather?start=newbot"
    _print(
        "\n[bold cyan]EgoVault Telegram Setup[/bold cyan]\n"
        "─────────────────────────────────────────────\n"
        "Step 1 — Create a free bot on Telegram (~30 s).\n\n"
        "[bold]Scan this QR with your phone[/bold] to open @BotFather:\n"
    )
    _print_qr(_botfather_url)
    _print(
        f"  Or open: [bold cyan]{_botfather_url}[/bold cyan]\n\n"
        "  In the BotFather chat:\n"
        "    · tap [bold]Start[/bold] (or send [bold]/newbot[/bold])\n"
        "    · choose a name and username for your bot\n"
        "    · [bold]copy the token[/bold] it gives you\n"
    )

    # ── Step 1: get & validate token ─────────────────────────────────────────
    while True:
        token = click.prompt("Paste your bot token (or 'q' to quit)").strip()
        if token.lower() in ("q", "quit", "exit", ""):
            _print("[yellow]Setup cancelled.[/yellow]")
            return None

        _print("[dim]Validating token…[/dim]")
        try:
            me = _tg_api(token, "getMe")
            if not me.get("ok"):
                _print("[red]Invalid token — please check and try again.[/red]")
                continue
            bot_username = me["result"]["username"]
            bot_name = me["result"]["first_name"]
            _print(f"[green]✓[/green] Connected to [bold]{bot_name}[/bold] (@{bot_username})")
            break
        except urllib.error.HTTPError as exc:
            if exc.code == 401:
                _print("[red]Token rejected by Telegram (HTTP 401). Double-check it.[/red]")
            else:
                _print(f"[red]HTTP error {exc.code} — check your internet connection.[/red]")
        except Exception as exc:  # noqa: BLE001
            _print(f"[red]Could not reach Telegram API: {exc}[/red]")

    # ── Step 2: auto-capture chat_id via QR code ────────────────────────────
    bot_url = f"https://t.me/{bot_username}"
    _print("\n[bold]Scan the QR code with your phone to open the bot:[/bold]\n")
    _print_qr(bot_url)
    _print(
        f"\n  Or open: [bold cyan]{bot_url}[/bold cyan]\n"
        f"\n[bold]Then send [cyan]/start[/cyan] — the chat ID is captured automatically.[/bold]\n"
        "[dim](waiting up to 90 seconds…)[/dim]\n"
    )

    # Delete any stale updates so we only see fresh ones
    try:
        _tg_api(token, "getUpdates", {"offset": -1, "timeout": 0})
    except Exception:
        pass

    chat_id: int | None = None
    deadline = 90
    for _ in range(deadline):
        import time
        try:
            result = _tg_api(token, "getUpdates", {"timeout": 1, "limit": 5}, timeout=5)
            updates = result.get("result", [])
            for upd in updates:
                msg = upd.get("message") or upd.get("channel_post")
                if msg and "chat" in msg:
                    chat_id = msg["chat"]["id"]
                    sender = (
                        msg["chat"].get("first_name", "")
                        + " "
                        + msg["chat"].get("last_name", "")
                    ).strip() or str(chat_id)
                    # Ack the update
                    _tg_api(token, "getUpdates", {"offset": upd["update_id"] + 1, "limit": 1, "timeout": 0})
                    break
            if chat_id:
                break
        except Exception:
            pass
        time.sleep(1)

    if chat_id is None:
        _print(
            "[yellow]No message received in 90 s.[/yellow]\n"
            "Find your chat ID by messaging @userinfobot on Telegram,\n"
            "then add it manually to egovault.toml:\n"
            "  [telegram]\n"
            "  allowed_chat_ids = [<your-id>]"
        )
        return None

    _print(f"[green]✓[/green] Chat ID captured: [bold cyan]{chat_id}[/bold cyan]  (from {sender})")

    # ── Step 3: save config ───────────────────────────────────────────────────
    if _save_telegram_config(token, chat_id):
        _print("[green]✓[/green] Saved to [cyan]egovault.toml[/cyan]")
    else:
        _print(
            f"[yellow]Could not auto-save.[/yellow]  Add manually to egovault.toml:\n"
            f'  [telegram]\n'
            f'  token = "{token}"\n'
            f"  allowed_chat_ids = [{chat_id}]"
        )

    # Send a welcome message back to the user
    try:
        _tg_api(
            token, "sendMessage",
            {
                "chat_id": chat_id,
                "text": (
                    "✅ EgoVault is connected!\n\n"
                    "Ask me anything about your emails, documents, "
                    "and files.\n\nSend /help to see all commands."
                ),
            },
        )
    except Exception:
        pass

    return token, chat_id


# ---------------------------------------------------------------------------
# Markdown → Telegram HTML converter (minimal, safe)
# ---------------------------------------------------------------------------

def _md_to_html(text: str) -> str:
    """Convert simple Markdown to Telegram HTML (safe subset).

    Handles bold, italic, inline code, code blocks, and headings.
    Escapes raw HTML characters first so injected HTML in the LLM answer
    cannot break Telegram's parser.
    """
    # 1. Escape any raw HTML characters in the text
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # 2. Fenced code blocks  ```lang\n...\n```  →  <pre>...</pre>
    text = re.sub(
        r"```[^\n]*\n(.*?)```",
        lambda m: "<pre>" + m.group(1).rstrip() + "</pre>",
        text,
        flags=re.DOTALL,
    )

    # 3. Inline code  `code`  →  <code>code</code>
    text = re.sub(r"`([^`\n]+)`", r"<code>\1</code>", text)

    # 4. Bold  **text** or __text__  →  <b>text</b>
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"__(.+?)__", r"<b>\1</b>", text)

    # 5. Italic  *text* or _text_  →  <i>text</i>
    #    Use word-boundary trick to avoid touching already-processed tags
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"(?<!_)_(?!_)(.+?)(?<!_)_(?!_)", r"<i>\1</i>", text)

    # 6. Headings  ## Title  →  <b>Title</b>
    text = re.sub(r"^#{1,6}\s+(.+)$", r"<b>\1</b>", text, flags=re.MULTILINE)

    return text


def _store_forwarded_message(message, settings: "Settings") -> tuple[bool, str]:
    """Store a forwarded Telegram message as a vault record.

    Returns ``(is_new, sender_description)`` where *is_new* is True when the
    record was freshly inserted (False = duplicate already in vault).
    """
    from datetime import timezone

    from telegram import (
        MessageOriginChannel,
        MessageOriginChat,
        MessageOriginHiddenUser,
        MessageOriginUser,
    )

    from egovault.core.schema import NormalizedRecord
    from egovault.core.store import VaultStore

    origin = message.forward_origin

    if isinstance(origin, MessageOriginUser):
        u = origin.sender_user
        sender_id = str(u.id)
        last = f" {u.last_name}" if u.last_name else ""
        sender_name = f"{u.first_name}{last}".strip()
    elif isinstance(origin, MessageOriginHiddenUser):
        sender_id = origin.sender_user_name
        sender_name = origin.sender_user_name
    elif isinstance(origin, MessageOriginChat):
        chat = origin.sender_chat
        sender_id = str(chat.id)
        sender_name = chat.title or chat.username or sender_id
    elif isinstance(origin, MessageOriginChannel):
        chat = origin.chat
        sender_id = str(chat.id)
        sender_name = chat.title or chat.username or sender_id
    else:
        sender_id = "unknown"
        sender_name = "Unknown"

    ts = origin.date
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)

    forwarded_by = str(message.from_user.id) if message.from_user else None
    record = NormalizedRecord(
        platform="telegram",
        record_type="message",
        timestamp=ts,
        sender_id=sender_id,
        sender_name=sender_name,
        thread_id=sender_id,
        thread_name=sender_name,
        body=message.text or "",
        raw={"forwarded_by": forwarded_by, "source": "bot_forward"},
    )

    store = VaultStore(settings.vault_db)
    store.init_db()
    try:
        is_new = store.upsert_record(record)
    finally:
        store.close()

    return is_new, sender_name


def _split_message(text: str, max_len: int = _TG_MAX_LEN) -> list[str]:
    """Split *text* into chunks that fit within Telegram's message length limit.

    Splits at paragraph boundaries where possible to preserve readability.
    """
    if len(text) <= max_len:
        return [text]
    parts: list[str] = []
    while text:
        if len(text) <= max_len:
            parts.append(text)
            break
        # Prefer splitting at a blank line (paragraph boundary)
        cut = text.rfind("\n\n", 0, max_len)
        if cut == -1:
            cut = text.rfind("\n", 0, max_len)
        if cut == -1:
            cut = max_len
        parts.append(text[:cut].strip())
        text = text[cut:].strip()
    return [p for p in parts if p]


# ---------------------------------------------------------------------------
# Bot handlers
# ---------------------------------------------------------------------------

def _is_allowed(chat_id: int, settings: "Settings") -> bool:
    """Return True if *chat_id* is in the whitelist."""
    allowed = settings.telegram.allowed_chat_ids
    # Empty whitelist = no access (fail-safe)
    return bool(allowed) and chat_id in allowed


async def _cmd_start(update, context) -> None:
    """Handle /start."""
    from telegram import Update
    assert isinstance(update, Update)  # noqa: S101
    settings: Settings = context.bot_data["settings"]
    chat_id = update.effective_chat.id
    if not _is_allowed(chat_id, settings):
        await update.message.reply_text("Access denied.")
        return
    await update.message.reply_text(
        "<b>EgoVault</b> — your personal data vault\n\n"
        "Ask me anything about your emails, documents, and files.\n"
        "Send /help for all commands.",
        parse_mode="HTML",
    )


async def _cmd_help(update, context) -> None:
    """Handle /help."""
    from telegram import Update
    assert isinstance(update, Update)  # noqa: S101
    settings: Settings = context.bot_data["settings"]
    if not _is_allowed(update.effective_chat.id, settings):
        return
    # Telegram doesn't render Markdown tables — use a plain list instead.
    # Telegram commands can't have hyphens — use underscores for gmail commands.
    lines = (
        "<b>EgoVault commands</b>\n\n"
        "/help — show this help\n"
        "/clear — clear conversation history\n"
        "/restart — reset conversation history\n"
        "/sources — sources from last answer\n"
        "/profile — show owner profile\n"
        "/status — LLM server + vault stats\n"
        "/top N — set retrieval depth (1–50)\n"
        "/scan &lt;folder&gt; — scan folder into vault\n"
        "/scan — list known folder aliases\n"
        "/open — open last saved file\n"
        "/gmail_auth — connect Gmail (one-time)\n"
        "/gmail_sync — import emails from Gmail\n"
        "/telegram_auth — authenticate Telegram (one-time)\n"
        "/telegram_sync — import Telegram message history\n"
        "/schedule — manage scheduled tasks\n"
        "/exit — stop the bot"
    )
    await update.message.reply_text(lines, parse_mode="HTML")


async def _cmd_clear(update, context) -> None:
    """Handle /clear — reset conversation history for this chat."""
    from telegram import Update
    assert isinstance(update, Update)  # noqa: S101
    settings: Settings = context.bot_data["settings"]
    chat_id = update.effective_chat.id
    if not _is_allowed(chat_id, settings):
        return
    _chat_histories.pop(chat_id, None)
    _chat_sources.pop(chat_id, None)
    await update.message.reply_text("Conversation history cleared.")


async def _cmd_sources(update, context) -> None:
    """Handle /sources."""
    from telegram import Update
    assert isinstance(update, Update)  # noqa: S101
    settings: Settings = context.bot_data["settings"]
    chat_id = update.effective_chat.id
    if not _is_allowed(chat_id, settings):
        return
    result = _handle_command("/sources", {"sources": _chat_sources.get(chat_id, [])})
    await update.message.reply_text(result.text, parse_mode="Markdown")


async def _cmd_status(update, context) -> None:
    """Handle /status."""
    from telegram import Update
    assert isinstance(update, Update)  # noqa: S101
    settings: Settings = context.bot_data["settings"]
    chat_id = update.effective_chat.id
    if not _is_allowed(chat_id, settings):
        return
    result = _handle_command("/status", {"settings": settings})
    await update.message.reply_text(result.text, parse_mode="Markdown")


async def _cmd_profile(update, context) -> None:
    """Handle /profile."""
    from telegram import Update
    assert isinstance(update, Update)  # noqa: S101
    settings: Settings = context.bot_data["settings"]
    chat_id = update.effective_chat.id
    if not _is_allowed(chat_id, settings):
        return
    profile = context.bot_data.get("owner_profile", "")
    result = _handle_command("/profile", {"owner_profile": profile})
    await update.message.reply_text(result.text, parse_mode="Markdown")


async def _cmd_top(update, context) -> None:
    """Handle /top <N>."""
    from telegram import Update
    assert isinstance(update, Update)  # noqa: S101
    settings: Settings = context.bot_data["settings"]
    chat_id = update.effective_chat.id
    if not _is_allowed(chat_id, settings):
        return
    args = context.args or []
    cmd = f"/top {args[0]}" if args else "/top"
    result = _handle_command(cmd, {})
    if result is None:
        await update.message.reply_text("Usage: /top <number>")
        return
    if result.action == "top_n":
        context.bot_data["top_n"] = result.value
    await update.message.reply_text(result.text, parse_mode="Markdown")


async def _cmd_restart(update, context) -> None:
    """Handle /restart — clear conversation history."""
    from telegram import Update
    assert isinstance(update, Update)  # noqa: S101
    settings: Settings = context.bot_data["settings"]
    chat_id = update.effective_chat.id
    if not _is_allowed(chat_id, settings):
        return
    _chat_histories.pop(chat_id, None)
    _chat_sources.pop(chat_id, None)
    result = _handle_command("/restart", {})
    await update.message.reply_text(result.text, parse_mode="Markdown")


async def _cmd_exit(update, context) -> None:
    """Handle /exit — send goodbye and stop the bot."""
    from telegram import Update
    assert isinstance(update, Update)  # noqa: S101
    settings: Settings = context.bot_data["settings"]
    if not _is_allowed(update.effective_chat.id, settings):
        return
    await update.message.reply_text("Goodbye. EgoVault bot is shutting down.")
    await context.application.stop()

async def _cmd_scan(update, context) -> None:
    """Handle /scan [folder]."""
    from telegram import Update
    assert isinstance(update, Update)  # noqa: S101
    settings: Settings = context.bot_data["settings"]
    chat_id = update.effective_chat.id
    if not _is_allowed(chat_id, settings):
        return
    args = context.args or []
    folder = " ".join(args).strip()
    user_input = f"/scan {folder}" if folder else "/scan"
    status_msg = await update.message.reply_text(
        f"\u23f3 Scanning `{folder}`..." if folder else "\u23f3 Listing known folders...",
        parse_mode="Markdown",
    )
    loop = asyncio.get_event_loop()
    def _run():
        from egovault.core.store import VaultStore
        from egovault.agent.commands import _run_capturing, _handle_scan
        store = VaultStore(settings.vault_db)
        store.init_db()
        try:
            return _run_capturing(_handle_scan, user_input, store, settings)
        finally:
            store.close()
    output = await loop.run_in_executor(None, _run)
    await status_msg.edit_text(output[:_TG_MAX_LEN] or "Done.")


async def _cmd_gmail_sync(update, context) -> None:
    """Handle /gmail\_sync [--since DATE] [--max N]."""
    from telegram import Update
    assert isinstance(update, Update)  # noqa: S101
    settings: Settings = context.bot_data["settings"]
    chat_id = update.effective_chat.id
    if not _is_allowed(chat_id, settings):
        return
    args = context.args or []
    user_input = "/gmail-sync " + " ".join(args) if args else "/gmail-sync"
    status_msg = await update.message.reply_text("\u23f3 Syncing Gmail\u2026")
    loop = asyncio.get_event_loop()
    def _run():
        from egovault.core.store import VaultStore
        from egovault.agent.commands import _run_capturing, _handle_gmail_sync
        store = VaultStore(settings.vault_db)
        store.init_db()
        try:
            return _run_capturing(_handle_gmail_sync, user_input, store, settings)
        finally:
            store.close()
    output = await loop.run_in_executor(None, _run)
    await status_msg.edit_text(output[:_TG_MAX_LEN] or "Sync complete.")


async def _cmd_gmail_auth(update, context) -> None:
    """Handle /gmail\_auth \u2014 start Gmail App Password setup flow."""
    from telegram import Update
    assert isinstance(update, Update)  # noqa: S101
    settings: Settings = context.bot_data["settings"]
    chat_id = update.effective_chat.id
    if not _is_allowed(chat_id, settings):
        return
    from egovault.utils.gmail_imap import load_credentials as _load_imap
    data_dir = Path(settings.vault_db).parent
    if _load_imap(data_dir) is not None:
        await update.message.reply_text(
            "\u2705 Already connected to Gmail.\nRun /gmail\_sync to import emails.",
            parse_mode="Markdown",
        )
        return
    context.user_data["_gmail_auth_step"] = "email"
    await update.message.reply_text(
        "<b>Gmail Setup \u2014 IMAP App Password</b>\n\n"
        "Step 1: Enter your Gmail address:",
        parse_mode="HTML",
    )


async def _cmd_telegram_sync(update, context) -> None:
    """Handle /telegram\_sync [--since DATE] [--max N]."""
    from telegram import Update
    assert isinstance(update, Update)  # noqa: S101
    settings: Settings = context.bot_data["settings"]
    chat_id = update.effective_chat.id
    if not _is_allowed(chat_id, settings):
        return
    args = context.args or []
    user_input = "/telegram-sync " + " ".join(args) if args else "/telegram-sync"
    status_msg = await update.message.reply_text("\u23f3 Syncing Telegram messages\u2026")
    loop = asyncio.get_event_loop()
    def _run():
        from egovault.core.store import VaultStore
        from egovault.agent.commands import _run_capturing, _handle_telegram_sync
        store = VaultStore(settings.vault_db)
        store.init_db()
        try:
            return _run_capturing(_handle_telegram_sync, user_input, store, settings)
        finally:
            store.close()
    output = await loop.run_in_executor(None, _run)
    await status_msg.edit_text(output[:_TG_MAX_LEN] or "Sync complete.")


# ---------------------------------------------------------------------------
# Telegram auth multi-step wizard helpers
# ---------------------------------------------------------------------------

def _clear_tg_auth_state(user_data: dict) -> None:
    for key in (
        "_tg_auth_step", "_tg_auth_data",
        "_tg_code_queue", "_tg_pw_queue", "_tg_result_queue",
    ):
        user_data.pop(key, None)


async def _monitor_tg_auth(chat_id: int, bot, user_data: dict) -> None:
    """Background task: relay Telethon auth events back to the user."""
    result_queue = user_data.get("_tg_result_queue")
    if result_queue is None:
        return
    while True:
        try:
            event = await asyncio.wait_for(result_queue.get(), timeout=300)
        except asyncio.TimeoutError:
            await bot.send_message(chat_id, "\u23f1 Timed out waiting for Telegram auth.")
            _clear_tg_auth_state(user_data)
            return
        kind = event[0]
        if kind == "need_pw":
            user_data["_tg_auth_step"] = "tg_2fa"
            await bot.send_message(
                chat_id,
                "\U0001f512 Your account has Two-Step Verification enabled.\n"
                "Enter your <b>2FA password</b>:",
                parse_mode="HTML",
            )
            # Keep monitoring for the final result
        elif kind == "ok":
            display_name = event[1] if len(event) > 1 else "unknown"
            _clear_tg_auth_state(user_data)
            await bot.send_message(
                chat_id,
                f"\u2705 Authenticated as <b>{display_name}</b>!\n"
                "Run /telegram\\_sync to import your messages.",
                parse_mode="HTML",
            )
            return
        elif kind == "error":
            error_msg = event[1] if len(event) > 1 else "Unknown error"
            _clear_tg_auth_state(user_data)
            await bot.send_message(
                chat_id,
                f"\u274c Authentication failed: {error_msg}\n"
                "Try /telegram\\_auth again.",
            )
            return


async def _cmd_telegram_auth(update, context) -> None:
    """Handle /telegram\_auth \u2014 start Telegram MTProto authentication wizard."""
    from telegram import Update
    assert isinstance(update, Update)  # noqa: S101
    settings: Settings = context.bot_data["settings"]
    chat_id = update.effective_chat.id
    if not _is_allowed(chat_id, settings):
        return

    from pathlib import Path
    from egovault.utils.telegram_api import load_credentials, get_session_path
    data_dir = Path(settings.vault_db).parent
    creds = load_credentials(data_dir)
    if creds is not None and get_session_path(data_dir).with_suffix(".session").exists():
        await update.message.reply_text(
            "\u2705 Already authenticated.\nUse /telegram\\_sync to import messages.",
        )
        return

    _clear_tg_auth_state(context.user_data)
    context.user_data["_tg_auth_step"] = "api_id"
    context.user_data["_tg_auth_data"] = {}
    await update.message.reply_text(
        "<b>Telegram Auth Setup</b>\n\n"
        "Get your <b>api_id</b> and <b>api_hash</b> at "
        "https://my.telegram.org/apps\n\n"
        "Step 1 of 3: Enter your <b>api_id</b> (numbers only):",
        parse_mode="HTML",
    )


async def _cmd_schedule(update, context) -> None:
    """Handle /schedule [args]."""
    from telegram import Update
    assert isinstance(update, Update)  # noqa: S101
    settings: Settings = context.bot_data["settings"]
    chat_id = update.effective_chat.id
    if not _is_allowed(chat_id, settings):
        return
    scheduler = context.bot_data.get("scheduler")
    if scheduler is None:
        await update.message.reply_text("Scheduler not available.")
        return
    args = context.args or []
    user_input = "/schedule " + " ".join(args) if args else "/schedule"
    loop = asyncio.get_event_loop()
    def _run():
        from egovault.core.store import VaultStore
        from egovault.agent.commands import _run_capturing
        from egovault.agent.session import _handle_schedule
        store = VaultStore(settings.vault_db)
        store.init_db()
        try:
            return _run_capturing(_handle_schedule, user_input, store, settings, scheduler)
        finally:
            store.close()
    output = await loop.run_in_executor(None, _run)
    await update.message.reply_text(output[:_TG_MAX_LEN] or "Done.")


async def _cmd_open(update, context) -> None:
    """Handle /open \u2014 open last saved file with the OS default app (server-side)."""
    from telegram import Update
    assert isinstance(update, Update)  # noqa: S101
    settings: Settings = context.bot_data["settings"]
    chat_id = update.effective_chat.id
    if not _is_allowed(chat_id, settings):
        return
    path = _chat_last_files.get(chat_id, "")
    if not path:
        await update.message.reply_text(
            "No recent file to open \u2014 ask me to find or save a file first."
        )
        return
    from egovault.chat.session import _open_with_default_app
    await update.message.reply_text(_open_with_default_app(path))

async def _handle_message(update, context) -> None:
    """Handle a plain text message — run the full RAG agent pipeline."""
    from telegram import Update
    assert isinstance(update, Update)  # noqa: S101
    settings: Settings = context.bot_data["settings"]
    chat_id = update.effective_chat.id

    if not _is_allowed(chat_id, settings):
        await update.message.reply_text("Access denied.")
        return

    user_text = (update.message.text or "").strip()
    if not user_text:
        return

    # ── Gmail auth conversation flow ──────────────────────────────────────────
    auth_step = context.user_data.get("_gmail_auth_step")
    if auth_step == "email":
        email = user_text.strip()
        if "@" not in email:
            email = f"{email}@gmail.com"
        context.user_data["_gmail_auth_email"] = email
        context.user_data["_gmail_auth_step"] = "password"
        await update.message.reply_text(
            f"\u2705 Got it: <code>{email}</code>\n\n"
            "Step 2: Enter your 16-character App Password\n"
            "(Create one at: https://myaccount.google.com/apppasswords)",
            parse_mode="HTML",
        )
        return
    if auth_step == "password":
        pwd = user_text.replace(" ", "").strip()
        email = context.user_data.pop("_gmail_auth_email", "")
        context.user_data.pop("_gmail_auth_step", None)
        await context.bot.send_chat_action(chat_id=chat_id, action="typing")
        loop = asyncio.get_event_loop()
        def _do_auth():
            from egovault.utils.gmail_imap import verify_connection, save_credentials
            data_dir = Path(settings.vault_db).parent
            try:
                verify_connection(email, pwd)
                save_credentials(data_dir, email, pwd)
                return "\u2705 Connected! Run /gmail\_sync to import your emails."
            except Exception as exc:  # noqa: BLE001
                return f"\u274c Authentication failed: {exc}\nCheck your App Password and try /gmail\_auth again."
        result = await loop.run_in_executor(None, _do_auth)
        await update.message.reply_text(result, parse_mode="Markdown")
        return

    # ── Telegram auth conversation flow ───────────────────────────────────────
    tg_auth_step = context.user_data.get("_tg_auth_step")
    if tg_auth_step == "api_id":
        try:
            api_id = int(user_text.strip())
        except ValueError:
            await update.message.reply_text("\u274c api_id must be numbers only. Try again:")
            return
        context.user_data["_tg_auth_data"]["api_id"] = api_id
        context.user_data["_tg_auth_step"] = "api_hash"
        await update.message.reply_text(
            "\u2705 Got api_id.\n\nStep 2 of 3: Enter your <b>api_hash</b>:",
            parse_mode="HTML",
        )
        return
    if tg_auth_step == "api_hash":
        api_hash = user_text.strip()
        if not api_hash:
            await update.message.reply_text("\u274c api_hash cannot be empty. Try again:")
            return
        context.user_data["_tg_auth_data"]["api_hash"] = api_hash
        context.user_data["_tg_auth_step"] = "phone"
        await update.message.reply_text(
            "\u2705 Got api_hash.\n\nStep 3 of 3: Enter your <b>phone number</b> "
            "(e.g. <code>+385991234567</code>):",
            parse_mode="HTML",
        )
        return
    if tg_auth_step == "phone":
        phone = user_text.strip()
        if not phone:
            await update.message.reply_text("\u274c Phone cannot be empty. Try again:")
            return
        data = context.user_data["_tg_auth_data"]
        api_id = data["api_id"]
        api_hash = data["api_hash"]
        context.user_data["_tg_auth_step"] = "code"

        # Create async queues on the running event loop
        loop = asyncio.get_event_loop()
        code_queue: "asyncio.Queue[str]" = asyncio.Queue()
        pw_queue: "asyncio.Queue[str]" = asyncio.Queue()
        result_queue: "asyncio.Queue[tuple]" = asyncio.Queue()
        context.user_data["_tg_code_queue"] = code_queue
        context.user_data["_tg_pw_queue"] = pw_queue
        context.user_data["_tg_result_queue"] = result_queue

        # Start monitoring task (listens for need_pw / ok / error from the thread)
        asyncio.create_task(_monitor_tg_auth(chat_id, context.bot, context.user_data))

        def _run_telethon():
            from pathlib import Path as _P
            from egovault.adapters.telegram_history import run_auth
            from egovault.utils.telegram_api import get_session_path, save_credentials

            def code_cb() -> str:
                fut = asyncio.run_coroutine_threadsafe(code_queue.get(), loop)
                return fut.result(timeout=120)

            def pw_cb() -> str:
                loop.call_soon_threadsafe(result_queue.put_nowait, ("need_pw",))
                fut = asyncio.run_coroutine_threadsafe(pw_queue.get(), loop)
                return fut.result(timeout=300)

            data_dir = _P(settings.vault_db).parent
            session_path = get_session_path(data_dir)
            try:
                display_name = run_auth(
                    api_id=api_id,
                    api_hash=api_hash,
                    phone=phone,
                    session_path=session_path,
                    code_callback=code_cb,
                    password_callback=pw_cb,
                )
                save_credentials(data_dir, api_id, api_hash, phone)
                loop.call_soon_threadsafe(result_queue.put_nowait, ("ok", display_name))
            except Exception as exc:  # noqa: BLE001
                loop.call_soon_threadsafe(result_queue.put_nowait, ("error", str(exc)))

        import threading
        threading.Thread(target=_run_telethon, daemon=True, name="tg-auth").start()
        await update.message.reply_text(
            "\u23f3 Connecting to Telegram\u2026\n\n"
            "A verification code will be sent to your Telegram app.\n"
            "Enter it here when you receive it:",
        )
        return
    if tg_auth_step == "code":
        code = user_text.strip()
        code_queue = context.user_data.get("_tg_code_queue")
        if code_queue is None:
            await update.message.reply_text("\u274c Auth session lost. Try /telegram\\_auth again.")
            _clear_tg_auth_state(context.user_data)
            return
        loop = asyncio.get_event_loop()
        loop.call_soon_threadsafe(code_queue.put_nowait, code)
        await update.message.reply_text("\u23f3 Verifying code\u2026")
        return
    if tg_auth_step == "tg_2fa":
        pw = user_text.strip()
        pw_queue = context.user_data.get("_tg_pw_queue")
        if pw_queue is None:
            await update.message.reply_text("\u274c Auth session lost. Try /telegram\\_auth again.")
            _clear_tg_auth_state(context.user_data)
            return
        loop = asyncio.get_event_loop()
        loop.call_soon_threadsafe(pw_queue.put_nowait, pw)
        await update.message.reply_text("\u23f3 Verifying 2FA password\u2026")
        return

    # ── Forward-to-vault ──────────────────────────────────────────────────────
    # When the user forwards any Telegram message to the bot, store it directly
    # as a vault record instead of running the LLM pipeline.
    if update.message.forward_origin is not None:
        loop = asyncio.get_event_loop()
        is_new, sender = await loop.run_in_executor(
            None, _store_forwarded_message, update.message, settings
        )
        escaped = sender.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        if is_new:
            await update.message.reply_text(
                f"✅ Stored message from <b>{escaped}</b> in your vault.",
                parse_mode="HTML",
            )
        else:
            await update.message.reply_text(
                f"ℹ️ Message from <b>{escaped}</b> is already in your vault.",
                parse_mode="HTML",
            )
        return

    history = _chat_histories.setdefault(chat_id, [])
    progress_lines: list[str] = []

    # Run blocking agent pipeline in a thread so the async event loop is free.
    loop = asyncio.get_event_loop()

    def _run() -> None:
        from egovault.core.store import VaultStore
        store = VaultStore(settings.vault_db)
        store.init_db()
        try:
            session = AgentSession(store, settings)
            session_ctx: dict = {
                "settings": settings,
                "last_sources": [],
                "owner_profile": store.get_owner_profile() or "",
                "owner_profile_ref": {},
                "top_n": settings.telegram.top_n,
            }
            _run.turn = session.process_turn(
                user_text,
                list(history),  # snapshot to avoid race conditions
                emit=lambda label: progress_lines.append(label),
                session_ctx=session_ctx,
            )
            _run.last_file = session_ctx.get("last_file", "")
        finally:
            store.close()

    _run.turn = None
    _run.last_file = ""
    await loop.run_in_executor(None, _run)

    turn = _run.turn
    if turn is None:
        await update.message.reply_text("Could not process your request.")
        return

    # Track last file for /open
    if _run.last_file:
        _chat_last_files[chat_id] = _run.last_file
    elif turn.attachments:
        _chat_last_files[chat_id] = turn.attachments[0]

    # Update conversation history
    _chat_histories[chat_id] = list(turn.updated_history)
    # Keep last 20 turns to avoid unbounded growth
    if len(_chat_histories[chat_id]) > 40:
        _chat_histories[chat_id] = _chat_histories[chat_id][-40:]

    # Store sources for /sources command
    if turn.sources:
        _chat_sources[chat_id] = turn.sources
    else:
        _chat_sources.pop(chat_id, None)

    # Send the answer (split if over 4096 chars, convert Markdown → HTML)
    html_answer = _md_to_html(turn.text)
    for part in _split_message(html_answer):
        await update.message.reply_text(part, parse_mode="HTML")

    # Send any saved images / documents
    for att_path_str in turn.attachments:
        att_path = Path(att_path_str)
        if not att_path.exists():
            continue
        suffix = att_path.suffix.lower()
        if suffix in (".jpg", ".jpeg", ".png", ".gif", ".webp"):
            try:
                with att_path.open("rb") as f:
                    await context.bot.send_photo(chat_id=chat_id, photo=f)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not send photo %s: %s", att_path, exc)
        elif suffix == ".pdf":
            try:
                with att_path.open("rb") as f:
                    await context.bot.send_document(chat_id=chat_id, document=f)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not send document %s: %s", att_path, exc)


# ---------------------------------------------------------------------------
# Bot launch
# ---------------------------------------------------------------------------

def launch(settings: "Settings") -> None:
    """Start the Telegram bot in polling mode (blocking).

    This function blocks until the process is interrupted (Ctrl+C).
    It opens its own event loop and runs the python-telegram-bot Application.
    """
    try:
        from telegram.ext import Application, CommandHandler, MessageHandler, filters
    except ImportError as exc:
        raise ImportError(
            "python-telegram-bot is not installed. "
            "Run: pip install 'python-telegram-bot>=20'"
        ) from exc

    token = settings.telegram.token
    if not token:
        raise ValueError(
            "Telegram bot token is not set. "
            "Add [telegram] token = '...' to egovault.toml "
            "or set the EGOVAULT_TELEGRAM_TOKEN environment variable."
        )
    if not settings.telegram.allowed_chat_ids:
        raise ValueError(
            "No allowed_chat_ids configured. "
            "Add your Telegram chat ID to [telegram] allowed_chat_ids = [<id>] "
            "in egovault.toml.  Find your ID by messaging @userinfobot."
        )

    app = (
        Application.builder()
        .token(token)
        .build()
    )
    app.bot_data["settings"] = settings

    # Initialize scheduler so /schedule commands work
    import queue as _queue
    from egovault.utils.scheduler import Scheduler, make_executor
    from egovault.agent.session import _register_auto_schedules
    _notice_q: _queue.Queue[str] = _queue.Queue()
    _scheduler = Scheduler(Path(settings.vault_db).parent)
    _scheduler.start(
        executor=make_executor(settings.vault_db, settings),
        notice_queue=_notice_q,
    )
    _register_auto_schedules(_scheduler, settings)
    app.bot_data["scheduler"] = _scheduler

    app.add_handler(CommandHandler("start", _cmd_start))
    app.add_handler(CommandHandler("help", _cmd_help))
    app.add_handler(CommandHandler("clear", _cmd_clear))
    app.add_handler(CommandHandler("restart", _cmd_restart))
    app.add_handler(CommandHandler("sources", _cmd_sources))
    app.add_handler(CommandHandler("profile", _cmd_profile))
    app.add_handler(CommandHandler("status", _cmd_status))
    app.add_handler(CommandHandler("top", _cmd_top))
    app.add_handler(CommandHandler("scan", _cmd_scan))
    app.add_handler(CommandHandler("gmail_sync", _cmd_gmail_sync))
    app.add_handler(CommandHandler("gmail_auth", _cmd_gmail_auth))
    app.add_handler(CommandHandler("telegram_sync", _cmd_telegram_sync))
    app.add_handler(CommandHandler("telegram_auth", _cmd_telegram_auth))
    app.add_handler(CommandHandler("schedule", _cmd_schedule))
    app.add_handler(CommandHandler("open", _cmd_open))
    app.add_handler(CommandHandler("exit", _cmd_exit))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _handle_message))

    # Suppress per-poll HTTP noise from httpx and python-telegram-bot internals
    for _noisy in ("httpx", "telegram.ext.Application", "telegram.ext.Updater"):
        logging.getLogger(_noisy).setLevel(logging.WARNING)

    logger.info(
        "EgoVault Telegram bot starting (allowed chat IDs: %s)",
        settings.telegram.allowed_chat_ids,
    )
    app.run_polling(allowed_updates=["message"])
