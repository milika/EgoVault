"""EgoVault Streamlit web UI.

Launch with:  egovault web [--port PORT] [--host HOST]
Or directly: streamlit run egovault/chat/streamlit_app.py

Requires:  pip install "egovault[web]"
"""
from __future__ import annotations

import os
import queue
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import streamlit as st


if TYPE_CHECKING:
    pass

st.set_page_config(
    page_title="EgoVault",
    page_icon="vault",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "EgoVault - local-first personal data vault"},
)

# ---------------------------------------------------------------------------
# WAN password gate — runs before any content when tunnel is active
# ---------------------------------------------------------------------------

def _wan_password_gate() -> None:
    """Block access with a password prompt when the WAN tunnel is active.

    Activated only when EGOVAULT_WAN_URL is set (i.e. egovault web --wan).
    Uses hmac.compare_digest to prevent timing attacks.
    """
    import hashlib
    import hmac

    if not os.environ.get("EGOVAULT_WAN_URL", ""):
        return  # local access — no gate

    if st.session_state.get("_wan_auth"):
        return  # already authenticated this session

    from egovault.config import load_settings as _ls
    _pwd_hash = _ls().wan_password_hash

    st.title("EgoVault — Remote Access")
    pwd = st.text_input("Password", type="password", key="_wan_pwd_input")
    if st.button("Sign in"):
        candidate = "sha256:" + hashlib.sha256(pwd.encode()).hexdigest()
        if _pwd_hash and hmac.compare_digest(candidate, _pwd_hash):
            st.session_state["_wan_auth"] = True
            st.rerun()
        else:
            st.error("Incorrect password")
    st.stop()


_wan_password_gate()

# (Help text imported from egovault.agent.commands.HELP_MD — single source of truth)

# ---------------------------------------------------------------------------
# Cached resources (one per server process, shared across browser sessions)
# ---------------------------------------------------------------------------


@st.cache_resource
def _init_resources():
    from egovault.config import load_settings
    from egovault.core.store import VaultStore

    settings = load_settings()
    store = VaultStore(settings.vault_db)
    store.init_db()
    return store, settings


@st.cache_resource
def _init_scheduler():
    """Start the background scheduler and wait-queue once per process."""
    store, settings = _init_resources()
    data_dir = Path(settings.vault_db).parent

    from egovault.utils.scheduler import Scheduler, make_executor
    from egovault.agent.session import _register_auto_schedules

    notice_q: queue.Queue[str] = queue.Queue()
    scheduler = Scheduler(data_dir)
    scheduler.start(
        executor=make_executor(settings.vault_db, settings),
        notice_queue=notice_q,
    )
    _register_auto_schedules(scheduler, settings)
    return scheduler, notice_q


@st.cache_resource
def _init_background_tasks():
    """Start background embed/enrich/context threads once per process."""
    store, settings = _init_resources()
    _, notice_q = _init_scheduler()

    from egovault.agent.session import _start_background_tasks

    bg_progress: dict = {}
    threads = _start_background_tasks(store, settings, notice_q, bg_progress)
    return threads, bg_progress


@st.cache_resource
def _init_profile_thread() -> dict:
    """Run extract_owner_profile once in a background thread at startup.

    Returns a shared dict with keys:
      - 'profile': extracted string (empty until done)
      - 'done': bool — True once extraction has finished
      - 'record_count': vault record count at extraction time (for drift detection)
    """
    _, settings = _init_resources()
    from egovault.core.store import VaultStore as _VS
    # Open a thread-local connection — SQLite objects must not be shared across threads.
    local_store = _VS(settings.vault_db)
    result: dict = {"profile": local_store.get_owner_profile() or "", "done": False, "record_count": 0}

    # If a valid cached profile already exists, skip re-extraction.
    if result["profile"]:
        result["done"] = True
        try:
            result["record_count"] = local_store.count_records()
        except Exception:
            pass
        return result

    def _run() -> None:
        from egovault.processing.rag import extract_owner_profile
        from egovault.chat.session import _call_llm
        from egovault.core.store import VaultStore as _VS2
        thread_store = _VS2(settings.vault_db)
        llm_cfg = settings.llm
        kw = dict(
            base_url=llm_cfg.base_url, model=llm_cfg.model,
            timeout=llm_cfg.timeout_seconds, provider=llm_cfg.provider,
            api_key=llm_cfg.api_key,
            num_gpu=getattr(llm_cfg, "num_gpu", -1),
            num_thread=getattr(llm_cfg, "num_thread", 0),
        )
        try:
            result["profile"] = extract_owner_profile(thread_store, _call_llm, kw) or ""
        except Exception:
            pass
        finally:
            try:
                result["record_count"] = thread_store.count_records()
            except Exception:
                pass
            result["done"] = True

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return result


store, settings = _init_resources()
scheduler, _notice_q = _init_scheduler()
_bg_threads, _bg_progress = _init_background_tasks()
_profile_state = _init_profile_thread()

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

if "top_n" not in st.session_state:
    from egovault.utils.llm import auto_top_n
    st.session_state.top_n = auto_top_n(settings.llm.chunk_target_tokens)

if "messages" not in st.session_state:
    from egovault.config import load_agent_prompts
    _welcome_text = load_agent_prompts().get("welcome", "")
    _init_messages: list[dict] = (
        [{"role": "assistant", "content": _welcome_text}] if _welcome_text else []
    )
    _wan_init = os.environ.get("EGOVAULT_WAN_URL", "")
    if _wan_init:
        _init_messages.append({
            "role": "assistant",
            "content": f"⚡ **WAN access active:** [{_wan_init}]({_wan_init})",
        })
    st.session_state.messages = _init_messages

if "conv_history" not in st.session_state:
    st.session_state.conv_history = []

if "sources" not in st.session_state:
    st.session_state.sources = []

_INPUT_HISTORY_FILE = Path(settings.vault_db).parent / "input_history.json"


def _load_input_history() -> list[str]:
    try:
        return __import__("json").loads(_INPUT_HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_input_history(history: list[str]) -> None:
    try:
        _INPUT_HISTORY_FILE.write_text(
            __import__("json").dumps(history[-500:]),
            encoding="utf-8",
        )
    except Exception:
        pass


if "input_history" not in st.session_state:
    st.session_state.input_history = _load_input_history()

if "ingested_upload_hashes" not in st.session_state:
    st.session_state.ingested_upload_hashes = set()

if "recently_uploaded" not in st.session_state:
    # list of (filename, extracted_chars) — cleared after first query that uses them
    st.session_state.recently_uploaded = []

if "owner_profile" not in st.session_state:
    # Prefer the background-extracted profile; fall back to DB cache.
    if _profile_state["done"] and _profile_state["profile"]:
        st.session_state.owner_profile = _profile_state["profile"]
    else:
        try:
            st.session_state.owner_profile = store.get_owner_profile() or ""
        except Exception:
            st.session_state.owner_profile = ""
elif _profile_state["done"] and _profile_state["profile"] and not st.session_state.owner_profile:
    # Background thread finished after the session state was initialised.
    st.session_state.owner_profile = _profile_state["profile"]

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("EgoVault")
    st.caption("Local-first personal data vault")
    st.divider()

    if st.button("Clear conversation", width='stretch'):
        st.session_state.messages = []
        st.session_state.conv_history = []
        st.session_state.sources = []
        st.rerun()

    st.divider()
    st.caption(f"**Model:** `{settings.llm.model}`")
    st.caption(f"**Provider:** `{settings.llm.provider}`")
    st.caption(f"**Vault DB:** `{Path(settings.vault_db).name}`")
    st.caption(f"**Top-N chunks:** `{st.session_state.top_n}` (auto)")

    _wan_url = os.environ.get("EGOVAULT_WAN_URL", "")
    if _wan_url:
        st.divider()
        st.caption(f"⚡ **WAN:** [{_wan_url}]({_wan_url})")

    bg_pipeline_alive = any(t.is_alive() for t in _bg_threads)
    if _bg_progress:
        st.divider()

        @st.fragment(run_every=3)
        def _bg_progress_panel():
            import time as _time
            alive = any(t.is_alive() for t in _bg_threads)
            st.caption("**Background tasks** *(enrich → context → embed)*")
            for label in ["enrich", "context", "embed"]:
                p = _bg_progress.get(label)
                if p is None:
                    continue
                if p.total > 0:
                    ratio = p.done / p.total
                    eta_str = ""
                    if alive and p.done > 0 and ratio < 1.0:
                        elapsed = _time.time() - p.started_at
                        rate = p.done / elapsed
                        remaining = p.total - p.done
                        secs = int(remaining / rate) if rate > 0 else 0
                        if secs < 60:
                            eta_str = f"  eta {secs}s"
                        elif secs < 3600:
                            eta_str = f"  eta {secs // 60}min"
                        else:
                            h, m = divmod(secs // 60, 60)
                            eta_str = f"  eta {h}h {m}min" if m else f"  eta {h}h"
                    st.progress(ratio, text=f"{label} {p.done}/{p.total}{eta_str}")
                elif alive:
                    st.caption(f"* {label} queued…")

        _bg_progress_panel()

    @st.fragment(run_every=3)
    def _scheduled_tasks_panel():
        from egovault.utils.scheduler import format_next_run
        tasks = scheduler.list_tasks()
        if tasks:
            st.divider()
            st.caption("**Scheduled tasks**")
            for t in tasks:
                col_label, col_btn = st.columns([5, 1])
                col_label.caption(f"`[{t.id}]` {t.name}  \nnext: {format_next_run(t.next_run)}")
                with col_btn:
                    st.html(f"""
                    <style>
                    div[data-testid="stButton"] button[title="Cancel task {t.id}"] {{
                        font-size: 0.28rem;
                        padding: 0px 2px;
                        min-height: unset;
                        min-width: unset;
                        height: auto;
                        width: auto;
                        max-width: fit-content;
                        display: block;
                        margin: auto;
                    }}
                    </style>
                    """)
                    if st.button("✕", key=f"cancel_{t.id}", help=f"Cancel task {t.id}"):
                        scheduler.cancel_task(t.id)
                        st.rerun(scope="fragment")

    _scheduled_tasks_panel()

    if st.session_state.owner_profile:
        with st.expander("Owner profile"):
            st.caption(st.session_state.owner_profile)

    if st.session_state.sources:
        st.divider()
        st.markdown("**Sources - last answer**")
        for src in st.session_state.sources[:8]:
            st.caption(f"- {src}")
        if len(st.session_state.sources) > 8:
            st.caption(f"... and {len(st.session_state.sources) - 8} more")

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

st.title("EgoVault Chat")
st.caption("Ask anything about your personal data vault. Type `/help` for commands.")

# Capture chat input here (before the iframe) so input_history already contains
# the just-submitted prompt when _hist_js is serialised.  The widget still
# renders at the bottom of the viewport regardless of where it is declared.
prompt = st.chat_input("Ask about your vault...")
if prompt:
    st.session_state.input_history.append(prompt)
    _save_input_history(st.session_state.input_history)

# Inject JS for arrow-key input history — re-attaches via MutationObserver after
# every Streamlit re-render so the textarea reference is always fresh.
#
# st.iframe() serves inline HTML via a srcdoc iframe (same origin as the app),
# so window.parent.document is accessible. A nonce prevents Streamlit's
# content-hash dedup from reusing a stale cached iframe across re-runs.
_hist_js = __import__("json").dumps(st.session_state.input_history)
_hist_nonce = __import__("time").time_ns()
st.iframe(f"""<!-- nonce:{_hist_nonce} --><script>
(function() {{
    var history = {_hist_js};
    var idx = history.length;
    var lastTA = null;

    function attachHistory(ta) {{
        if (lastTA === ta) return;  // already attached to this element
        // Remove the old handler if present on the DOM element itself —
        // lastTA is null on each fresh iframe eval, so we can't rely on it.
        // ta.__ego_handler persists on the DOM node across Streamlit re-renders.
        if (ta.__ego_handler) {{
            ta.removeEventListener('keydown', ta.__ego_handler);
        }}
        lastTA = ta;
        ta.__ego_handler = function(e) {{
            if (!history.length) return;
            var setter = Object.getOwnPropertyDescriptor(
                window.parent.HTMLTextAreaElement.prototype, 'value'
            ).set;
            if (e.key === 'ArrowUp') {{
                e.preventDefault();
                if (idx > 0) idx--;
                setter.call(ta, history[idx]);
                ta.dispatchEvent(new Event('input', {{bubbles: true}}));
            }} else if (e.key === 'ArrowDown') {{
                e.preventDefault();
                if (idx < history.length - 1) {{
                    idx++;
                    setter.call(ta, history[idx]);
                }} else {{
                    idx = history.length;
                    setter.call(ta, '');
                }}
                ta.dispatchEvent(new Event('input', {{bubbles: true}}));
            }}
        }};
        ta.addEventListener('keydown', ta.__ego_handler);
    }}

    function start() {{
        var doc = window.parent.document;
        var mo = new MutationObserver(function() {{
            var ta = doc.querySelector('textarea[data-testid="stChatInputTextArea"]');
            if (ta) attachHistory(ta);
        }});
        mo.observe(doc.body, {{childList: true, subtree: true}});
        var ta = doc.querySelector('textarea[data-testid="stChatInputTextArea"]');
        if (ta) attachHistory(ta);
    }}

    function init() {{
        try {{ start(); }} catch(e) {{ setTimeout(init, 300); }}
    }}
    setTimeout(init, 100);
}})();
</script>
""", height=1)

def _render_attachments(att_paths: list, key_prefix: str = "") -> None:
    """Render image thumbnails + download buttons for *att_paths* in-place."""
    _img_exts = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".heic", ".bmp"}
    seen: set[str] = set()
    for att_path in att_paths:
        if att_path in seen:
            continue
        seen.add(att_path)
        p = Path(att_path)
        if not p.exists():
            continue
        if p.suffix.lower() in _img_exts:
            st.image(str(att_path), caption=p.name, width=220)
            st.download_button(
                label=f"Download {p.name}",
                data=p.read_bytes(),
                file_name=p.name,
                mime=f"image/{p.suffix.lower().lstrip('.')}",
                key=f"{key_prefix}dl_{att_path}",
            )
        else:
            st.download_button(
                label=f"Download {p.name}",
                data=p.read_bytes(),
                file_name=p.name,
                key=f"{key_prefix}dl_{att_path}",
            )


for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("attachments"):
            _render_attachments(msg["attachments"], key_prefix=f"h{i}_")

# Drain background notices and show as toasts
while True:
    try:
        _notice_q.get_nowait()
    except queue.Empty:
        break

# ---------------------------------------------------------------------------
# File uploader — attach files to vault
# ---------------------------------------------------------------------------

def _ingest_uploaded_files(uploaded_files, vault_store) -> list[tuple[str, str]]:
    """Write *uploaded_files* to inbox_dir and ingest them into the vault.

    Returns a list of ``(filename, status)`` pairs where status is one of:
    ``"added"``, ``"already known"``, or ``"skipped: <reason>"``.
    """
    import hashlib
    from datetime import datetime, timezone as _tz
    from egovault.adapters.local_inbox import (
        SUPPORTED_SUFFIXES, _extract_text, _FILE_TYPE_INFO,
    )
    from egovault.utils.hashing import compute_file_id
    from egovault.core.schema import NormalizedRecord

    inbox = Path(settings.inbox_dir).expanduser().resolve()
    inbox.mkdir(parents=True, exist_ok=True)

    results: list[tuple[str, str]] = []
    for uf in uploaded_files:
        fname = uf.name
        dest = inbox / fname
        # Avoid clobbering existing files with a numeric suffix
        stem, suffix = Path(fname).stem, Path(fname).suffix
        counter = 1
        while dest.exists():
            dest = inbox / f"{stem}_{counter}{suffix}"
            counter += 1

        if dest.suffix.lower() not in SUPPORTED_SUFFIXES:
            results.append((fname, f"skipped: unsupported type {dest.suffix}"))
            continue

        # Write bytes
        data = uf.getvalue()
        dest.write_bytes(data)

        # Dedup check using content hash — but if the existing record has an
        # empty body (e.g. previously uploaded when pdfplumber was missing),
        # fall through to re-extract and update it.
        content_hash = hashlib.sha256(data).hexdigest()
        if vault_store.is_file_known_by_hash(content_hash):
            _path_in_vault = None
            try:
                row = vault_store._con.execute(
                    "SELECT path FROM ingested_files WHERE content_hash = ? LIMIT 1",
                    (content_hash,),
                ).fetchone()
                if row:
                    _path_in_vault = row["path"]
            except Exception:
                pass
            _needs_body = (
                _path_in_vault is not None
                and vault_store.record_needs_body_update(_path_in_vault)
            )
            if not _needs_body:
                dest.unlink(missing_ok=True)
                results.append((fname, "already known"))
                st.session_state.ingested_upload_hashes.add(content_hash)
                continue
            # Body was empty — fall through to re-extract using the already-written dest

        # Extract text and build record
        try:
            body = _extract_text(dest)
        except Exception as exc:
            results.append((fname, f"skipped: extraction failed ({exc})"))
            continue

        stat = dest.stat()
        fid = compute_file_id(str(dest), stat.st_mtime, stat.st_size)
        effective_suffix = dest.suffix.lower() or ".conf"
        rec_type, mime = _FILE_TYPE_INFO.get(effective_suffix, ("document", "text/plain"))
        ts = datetime.fromtimestamp(stat.st_mtime, tz=_tz.utc)

        path_header = f"File: {dest}\n---\n"
        record = NormalizedRecord(
            platform="local",
            record_type=rec_type,
            timestamp=ts,
            sender_id="user",
            sender_name="user",
            thread_id=str(dest.parent),
            thread_name=str(dest.name),
            body=path_header + body,
            attachments=[str(dest)],
            raw={
                "file_name": dest.name,
                "file_id": fid,
                "content_hash": content_hash,
                "size_bytes": stat.st_size,
                "mtime": stat.st_mtime,
            },
            file_path=str(dest),
            mime_type=mime,
        )
        was_new = vault_store.upsert_record(record)
        if not was_new:
            # Record existed (same ID) but body may need updating (e.g. previously
            # ingested with empty body when pdfplumber was absent).
            vault_store.update_body_by_file_path(str(dest), path_header + body)
        vault_store.upsert_ingested_file(
            file_id=str(fid),
            path=str(dest),
            mtime=float(stat.st_mtime),
            size_bytes=int(stat.st_size),
            platform="local",
            content_hash=content_hash,
        )
        st.session_state.ingested_upload_hashes.add(content_hash)
        extracted_chars = len(body)

        # Auto-embed the record so it ranks in the semantic search lane
        # immediately, without requiring a separate `egovault embed` run.
        if body.strip():
            try:
                from egovault.processing.rag import embed_text
                _embed_cfg = settings.embeddings
                if _embed_cfg.enabled:
                    _embed_base = _embed_cfg.base_url.strip() or settings.llm.base_url
                    _vec = embed_text(body[:8192], _embed_base, _embed_cfg.model)
                    vault_store.upsert_embedding(record.id, _embed_cfg.model, _vec)
            except Exception:
                pass  # embedding failure is non-fatal; FTS5 still works

        st.session_state.recently_uploaded.append((fname, extracted_chars))
        if extracted_chars == 0:
            status = "added (0 chars — image-based PDF? text extraction failed)"
        else:
            status = ("added" if was_new else "updated") + f" ({extracted_chars:,} chars extracted)"
        results.append((fname, status))

    return results


with st.expander("📎 Attach files to vault", expanded=False):
    st.caption("Uploaded files are saved to the inbox and indexed — searchable immediately.")
    _uploaded = st.file_uploader(
        "Upload files",
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="vault_file_uploader",
    )
    if _uploaded:
        # Only process files not yet ingested this session (avoids re-run double-ingest)
        _new_uploads = [
            f for f in _uploaded
            if __import__("hashlib").sha256(f.getvalue()).hexdigest()
            not in st.session_state.ingested_upload_hashes
        ]
        if _new_uploads:
            with st.spinner(f"Adding {len(_new_uploads)} file(s) to vault…"):
                from egovault.core.store import VaultStore as _VS
                _up_store = _VS(settings.vault_db)
                _up_store.init_db()
                try:
                    _results = _ingest_uploaded_files(_new_uploads, _up_store)
                finally:
                    _up_store.close()
            for _fn, _status in _results:
                if _status == "added" or _status == "updated":
                    st.success(f"✓ **{_fn}** — {_status}")
                elif _status == "already known":
                    st.info(f"**{_fn}** — already in vault")
                else:
                    st.warning(f"**{_fn}** — {_status}")

# ---------------------------------------------------------------------------
# User input handler
# ---------------------------------------------------------------------------

# prompt and input_history.append() were already handled above (before the iframe).
if prompt:
    lower = prompt.strip().lower()

    # --- Natural-language scheduling detection (must run before /schedule check) ---
    if not lower.startswith("/"):
        from egovault.agent.intent import _resolve_schedule_intent
        _sched_cmd = _resolve_schedule_intent(lower, prompt.strip())
        if _sched_cmd:
            prompt = _sched_cmd
            lower = prompt.lower()

    # --- /schedule: frontend-specific handler (complex Streamlit UI) ---
    if lower.startswith("/schedule"):
        arg = prompt.strip()[len("/schedule"):].strip()
        reply_lines: list[str] = []

        if not arg or arg.lower() in ("--list", "-l", "list"):
            tasks = scheduler.list_tasks()
            active_bg = [t for t in _bg_threads if t.is_alive()]
            if not tasks and not active_bg:
                reply_lines.append("No scheduled tasks.")
                reply_lines.append("")
                reply_lines.append("Examples:")
                reply_lines.append("- `/schedule /gmail-sync in 5min`")
                reply_lines.append("- `/schedule /gmail-sync every day at 19:05`")
                reply_lines.append("- `/schedule /scan inbox every 30min`")
            else:
                if active_bg:
                    reply_lines.append("**Background tasks:**")
                    for t in active_bg:
                        label = t.name.replace("bg-", "")
                        p = _bg_progress.get(label)
                        if p and p.total > 0:
                            reply_lines.append(f"- {label}: {p.done}/{p.total} ({int(p.done/p.total*100)}%)")
                        else:
                            reply_lines.append(f"- {label}: running")
                if tasks:
                    from egovault.utils.scheduler import format_next_run, format_interval
                    reply_lines.append("**Scheduled tasks:**")
                    for t in tasks:
                        reply_lines.append(f"- `[{t.id}]` {t.name} - next: {format_next_run(t.next_run)} ({format_interval(t.interval_seconds)})")

        elif arg.lower().startswith(("--cancel", "-c")):
            tokens = arg.split(None, 1)
            tid = tokens[1].strip() if len(tokens) > 1 else ""
            if not tid:
                reply_lines.append("Usage: `/schedule --cancel <task_id>`")
            elif scheduler.cancel_task(tid):
                reply_lines.append(f"Task `{tid}` cancelled.")
            else:
                reply_lines.append(f"Task `{tid}` not found.")

        else:
            try:
                from egovault.utils.scheduler import parse_schedule_expression, format_next_run, format_interval

                cmd = ""
                time_expr = ""
                if arg.lower().startswith("/gmail-sync"):
                    cmd = "/gmail-sync"
                    time_expr = arg[len("/gmail-sync"):].strip()
                elif arg.lower().startswith("/scan"):
                    scan_tokens = arg.split(None, 2)
                    folder = scan_tokens[1] if len(scan_tokens) >= 2 else "inbox"
                    cmd = f"/scan {folder}"
                    time_expr = scan_tokens[2] if len(scan_tokens) >= 3 else ""
                elif arg.lower().startswith("chat:"):
                    from egovault.agent.intent import _SCHEDULE_TIME_RE as _stre
                    import re as _re_sc
                    full_text = arg[5:].strip()
                    time_m_inner = _stre.search(full_text)
                    if not time_m_inner:
                        reply_lines.append("Missing time expression in `chat:` prompt. Example:")
                        reply_lines.append("- `/schedule chat: search web for X and save to desktop in 5min`")
                    else:
                        time_expr = time_m_inner.group(0)
                        clean_prompt = _re_sc.sub(r"\s{2,}", " ", _stre.sub("", full_text).strip().strip(",").strip())
                        cmd = f"chat: {clean_prompt}"
                else:
                    reply_lines.append("Unknown command to schedule. Supported: `/gmail-sync`, `/scan <folder>`, `chat: <prompt>`")
                if cmd and time_expr:
                    parsed = parse_schedule_expression(time_expr)
                    next_run, interval_seconds = parsed
                    task = scheduler.add_task(
                        name=f"{cmd} ({time_expr.strip()})",
                        command=cmd,
                        next_run=next_run,
                        interval_seconds=interval_seconds,
                    )
                    reply_lines.append(f"Scheduled `{cmd}`  next: **{format_next_run(next_run)}** ({format_interval(interval_seconds)})")
                    reply_lines.append(f"Cancel with: `/schedule --cancel {task.id}`")
                elif cmd and not time_expr:
                    reply_lines.append("Missing time expression. Examples:")
                    reply_lines.append("- `/schedule /gmail-sync in 5min`")
                    reply_lines.append("- `/schedule /gmail-sync every day at 19:05`")
            except Exception as exc:
                reply_lines.append(f"Error: {exc}")

        reply = "\n".join(reply_lines)
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            st.markdown(reply)
        st.session_state.messages += [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": reply},
        ]
        st.stop()

    # --- All other input: unified dispatch via AgentSession.process_turn() ---
    #
    # Commands (exit, clear, help, sources, …) are handled immediately.
    # Chat messages run in a worker thread so the Streamlit main thread
    # can drive the progress display with st.status().

    from egovault.agent.session import AgentSession, TurnResult as _TurnResult

    _web_ctx: dict = {
        "settings": settings,
        "last_sources": st.session_state.sources,
        "owner_profile": st.session_state.owner_profile,
        "owner_profile_ref": {},
        "top_n": st.session_state.get("top_n", 10),
        "bg_threads": _bg_threads,
        "bg_progress": _bg_progress,
        "scheduler": scheduler,
        "notice_queue": _notice_q,
    }

    # Upload hint: tell the agent about recently uploaded files.
    _upload_hint = ""
    if st.session_state.recently_uploaded:
        _names = ", ".join(n for n, _ in st.session_state.recently_uploaded)
        _upload_hint = (
            f"[RECENTLY UPLOADED TO VAULT] The user just added these files: {_names}. "
            "Search for their content using the file name as the query keywords."
        )
        st.session_state.recently_uploaded = []
    if _upload_hint:
        _web_ctx["_upload_hint"] = _upload_hint

    conv_history: list = list(st.session_state.conv_history)

    # Fast synchronous check: does this input resolve to a simple command?
    # process_turn() returns immediately for commands without touching the LLM.
    _cmd_session = AgentSession(store, settings)
    _fast_turn = _cmd_session.process_turn(
        prompt, conv_history, emit=lambda _: None, session_ctx=dict(_web_ctx),
    )

    if _fast_turn.is_command and _fast_turn.action != "_delegate":
        # --- Simple command result ---
        with st.chat_message("user"):
            st.markdown(prompt)

        if _fast_turn.action == "exit":
            with st.chat_message("assistant"):
                st.markdown("Shutting down EgoVault web server…")
            st.session_state.messages += [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "Shutting down EgoVault web server…"},
            ]
            import signal
            signal.raise_signal(signal.SIGTERM)
            st.stop()

        if _fast_turn.action in ("clear", "restart"):
            st.session_state.messages = []
            st.session_state.conv_history = []
            st.session_state.sources = []
            st.rerun()

        if _fast_turn.action == "refresh_profile":
            from egovault.processing.rag import extract_owner_profile as _exp
            from egovault.agent.session import _call_llm as _cllm
            from egovault.core.store import VaultStore as _VS
            _ref_store = _VS(settings.vault_db)
            _ref_store.init_db()
            llm_cfg = settings.llm
            _ref_kw = dict(
                base_url=llm_cfg.base_url, model=llm_cfg.model,
                timeout=llm_cfg.timeout_seconds, provider=llm_cfg.provider,
                api_key=llm_cfg.api_key,
            )
            try:
                st.session_state.owner_profile = _exp(_ref_store, _cllm, _ref_kw) or ""
            finally:
                _ref_store.close()
            _reply = st.session_state.owner_profile or "_No profile found._"
            with st.chat_message("assistant"):
                st.markdown(_reply)
            st.session_state.messages += [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": _reply},
            ]
            st.stop()

        if _fast_turn.action == "top_n":
            st.session_state.top_n = _fast_turn.value

        if _fast_turn.text:
            with st.chat_message("assistant"):
                st.markdown(_fast_turn.text)
            st.session_state.messages += [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": _fast_turn.text},
            ]
        st.stop()

    # --- Agent / LLM turn — run in worker thread to keep UI responsive ---
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    answer = ""
    _saved_attachments: list[str] = []
    sources: list = []

    with st.chat_message("assistant"):
        answer_placeholder = st.empty()

        progress_lines: list[str] = []
        result_box: dict = {}
        done_event = threading.Event()

        def _progress_cb(label: str) -> None:
            progress_lines.append(label)

        # Snapshot session_state values in the main thread — st.session_state is
        # not accessible from background threads (no ScriptRunContext).
        _owner_profile_snap = st.session_state.get("owner_profile", "")
        _top_n_snap = st.session_state.get("top_n", 10)

        def _run() -> None:
            from egovault.core.store import VaultStore as _VS
            thread_store = _VS(settings.vault_db)
            thread_store.init_db()
            try:
                thread_session = AgentSession(thread_store, settings)
                thread_ctx: dict = {
                    "settings": settings,
                    "last_sources": [],
                    "owner_profile": _owner_profile_snap,
                    "owner_profile_ref": {},
                    "top_n": _top_n_snap,
                    "scheduler": scheduler,
                    "notice_queue": _notice_q,
                }
                if _upload_hint:
                    thread_ctx["_upload_hint"] = _upload_hint
                turn = thread_session.process_turn(
                    prompt, conv_history, emit=_progress_cb, session_ctx=thread_ctx,
                )
                result_box["turn"] = turn
                result_box["profile_dirty"] = thread_ctx["owner_profile_ref"].get("dirty", False)
            finally:
                thread_store.close()
                done_event.set()

        t = threading.Thread(target=_run, daemon=True)
        t.start()

        with st.status("Thinking...", expanded=True) as status:
            emitted = 0
            _sync_placeholder = None
            while not done_event.wait(timeout=0.4):
                for line in progress_lines[emitted:]:
                    if line.startswith("⚙ Syncing emails"):
                        if _sync_placeholder is None:
                            _sync_placeholder = st.empty()
                        _sync_placeholder.write(f"- {line}")
                    else:
                        _sync_placeholder = None
                        st.write(f"- {line}")
                    emitted += 1
            for line in progress_lines[emitted:]:
                if line.startswith("⚙ Syncing emails"):
                    if _sync_placeholder is None:
                        _sync_placeholder = st.empty()
                    _sync_placeholder.write(f"- {line}")
                else:
                    _sync_placeholder = None
                    st.write(f"- {line}")

            turn_result: "_TurnResult | None" = result_box.get("turn")
            if turn_result is None or (turn_result.is_command and not turn_result.text):
                status.update(label="Error", state="error")
                answer = "**Error:** agent pipeline returned no result."
            elif "**LLM error:**" in (turn_result.text or ""):
                status.update(label="Error", state="error")
                answer = turn_result.text
            else:
                status.update(label="Done", state="complete")
                answer = turn_result.text or ""
                sources = turn_result.sources
                _saved_attachments = turn_result.attachments

        if result_box.get("profile_dirty"):
            store.set_setting("owner_profile", "")
            st.session_state.owner_profile = ""
            _profile_state["done"] = False
            _profile_state["profile"] = ""
            _profile_state["record_count"] = 0

            def _refresh_after_scan() -> None:
                from egovault.processing.rag import extract_owner_profile
                from egovault.agent.session import _call_llm
                from egovault.core.store import VaultStore as _VS
                r_store = _VS(settings.vault_db)
                r_store.init_db()
                llm_cfg = settings.llm
                kw = dict(
                    base_url=llm_cfg.base_url, model=llm_cfg.model,
                    timeout=llm_cfg.timeout_seconds, provider=llm_cfg.provider,
                    api_key=llm_cfg.api_key,
                    num_gpu=getattr(llm_cfg, "num_gpu", -1),
                    num_thread=getattr(llm_cfg, "num_thread", 0),
                )
                try:
                    _profile_state["profile"] = extract_owner_profile(r_store, _call_llm, kw) or ""
                except Exception:
                    pass
                finally:
                    try:
                        _profile_state["record_count"] = r_store.count_records()
                    except Exception:
                        pass
                    _profile_state["done"] = True
                    r_store.close()

            threading.Thread(target=_refresh_after_scan, daemon=True).start()

        st.session_state.sources = sources

        if sources:
            footer = "\n\n---\n*Sources: " + " - ".join(sources[:5])
            if len(sources) > 5:
                footer += f" ... (+{len(sources) - 5} more)"
            footer += "*"
        else:
            footer = ""

        full_answer = answer + footer
        answer_placeholder.markdown(full_answer)

        if _saved_attachments:
            _render_attachments(_saved_attachments, key_prefix="cur_")

    st.session_state.messages.append({"role": "assistant", "content": full_answer, "attachments": list(_saved_attachments)})
    st.session_state.conv_history.append({"role": "user", "content": prompt})
    _history_answer = answer
    if _saved_attachments and any(
        kw in answer for kw in ("open window", "screenshot", "Screenshot")
    ):
        _history_answer = "[Screen capture completed. Call inspect_windows or take_screenshot again to get fresh results.]"
    st.session_state.conv_history.append({"role": "assistant", "content": _history_answer})