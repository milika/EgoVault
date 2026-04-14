"""Agent core session — re-exports from chat.session plus the new AgentSession class.

During the Cycle-7.5 migration the canonical implementations live in
``egovault.chat.session``.  This module:

1. Re-exports every agent-level symbol so that
   ``from egovault.agent.session import _call_llm`` works.
2. Adds the new **AgentSession** public-API class that all frontends should
   use going forward instead of calling ``_call_llm_agent`` directly.

When Phase 4 completes, the code will move here and ``chat/session.py`` will
become the shim.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from typing import Callable
    from egovault.config import Settings
    from egovault.core.store import VaultStore

# ---------------------------------------------------------------------------
# Re-export canonical symbols from chat.session (shim direction)
# ---------------------------------------------------------------------------
# ruff: noqa: F401
from egovault.chat.session import (  # noqa: F811
    _call_llm,
    _call_llm_agent,
    _handle_schedule,
    _register_auto_schedules,
    BgProgress,
    _format_eta,
    _start_background_tasks,
    _ollama_ps,
    _fmt_bytes,
)

# ---------------------------------------------------------------------------
# AgentSession — new public API for all frontends
# ---------------------------------------------------------------------------


class AgentSession:
    """Public entry point for all EgoVault frontends.

    Wraps ``_call_llm_agent`` behind a stable, frontend-agnostic interface.
    All frontends (TUI, Streamlit, Telegram, MCP) should call this instead of
    importing ``_call_llm_agent`` directly.

    Usage::

        session = AgentSession(store, settings)
        answer, new_history = session.process(user_input, history, emit=print)
    """

    def __init__(
        self,
        store: "VaultStore",
        settings: "Settings",
    ) -> None:
        self._store = store
        self._settings = settings

    def process(
        self,
        user_input: str,
        history: "list[dict]",
        emit: "Callable[[str], None]",
        *,
        session_ctx: "dict | None" = None,
    ) -> "tuple[str, list[dict]]":
        """Run one turn of the agent pipeline.

        Parameters
        ----------
        user_input:
            The user's raw message text.
        history:
            Conversation history in OpenAI message format (list of
            ``{"role": ..., "content": ...}`` dicts).
        emit:
            Callback for streaming progress labels (tool-call updates).
            Each frontend implements this differently — Rich console.print
            in TUI, queue.put_nowait in Streamlit, bot.send_message in
            Telegram.
        session_ctx:
            Optional per-session state dict.  The agent reads/writes keys
            like ``last_sources``, ``saved_attachments``, ``owner_profile``,
            ``scheduler``, ``last_file``, etc.  Frontends should create this
            dict once and pass it on every call.

        Returns
        -------
        (answer, updated_history)
            The assistant's answer string, and the history extended with the
            new user + assistant turns.
        """
        # Lazy imports to avoid any potential circular-import issues.
        from egovault.chat.session import _call_llm as _llm, _call_llm_agent as _agent  # noqa: F811
        from egovault.chat.rag import build_prompt

        settings = self._settings
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

        if session_ctx is None:
            session_ctx = {}
        session_ctx.setdefault("settings", settings)
        session_ctx.setdefault("call_llm_fn", _llm)
        session_ctx.setdefault("hyde_llm_kwargs", {
            **llm_kwargs,
            "timeout": min(15, llm_cfg.timeout_seconds),
        })

        top_n: int = session_ctx.get("top_n", 10)
        owner_profile: str = session_ctx.get("owner_profile", "")
        output_dir: str = session_ctx.get(
            "output_dir",
            str(Path(settings.output_dir).expanduser().resolve()),
        )
        today: str = session_ctx.get("today", _dt.date.today().isoformat())
        upload_hint: str = session_ctx.pop("_upload_hint", "")

        initial_messages = build_prompt(
            user_input,
            upload_hint,
            history=history,
            owner_profile=owner_profile,
            output_dir=output_dir,
            today=today,
        )

        answer, _data, _chunks = _agent(
            initial_messages,
            self._store,
            top_n,
            llm_kwargs,
            None,
            session_ctx=session_ctx,
            progress_cb=emit,
        )

        updated_history = history + [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": answer},
        ]
        return answer, updated_history

    def process_turn(
        self,
        user_input: str,
        history: "list[dict]",
        emit: "Callable[[str], None]" = lambda _: None,
        *,
        session_ctx: "dict | None" = None,
    ) -> "TurnResult":
        """Unified input dispatcher for all frontends.

        Handles (in order):

        1. NL intent resolution — natural-language phrases are mapped to slash
           commands before dispatch (e.g. "show sources" → ``/sources``).
        2. Simple command dispatch via :func:`~egovault.agent.commands.handle_command`
           (exit, clear, restart, help, sources, profile, status, top_n).
        3. Heavy I/O commands (``/scan``, ``/schedule``, ``/gmail-*``,
           ``/telegram-*``, ``/open``) — returns
           ``TurnResult(action="_delegate", value=resolved_input)`` so the
           frontend can run the appropriate frontend-specific handler.
        4. Agent pipeline — everything else is routed through
           :meth:`process`, which calls ``_call_llm_agent``.

        Parameters
        ----------
        user_input:
            Raw text as received from the transport layer.
        history:
            Conversation history (OpenAI message-dict format).
        emit:
            Progress callback.  Called with short label strings by the agent
            during tool-calling iterations.  Frontends render these however
            they like (Rich dim-print, Streamlit ``st.write``, etc.).
        session_ctx:
            Mutable per-session state.  Recognised keys mirror those of
            :meth:`process`: ``last_sources``, ``owner_profile``,
            ``owner_profile_ref``, ``top_n``, ``bg_threads``,
            ``bg_progress``, ``scheduler``, ``notice_queue``, etc.

        Returns
        -------
        TurnResult
            Always returns a ``TurnResult``; never raises (LLM errors are
            caught and surfaced as ``TurnResult.text``).
            ``action="_delegate"`` means the frontend must handle the command
            in ``TurnResult.value`` itself.
        """
        from egovault.agent.intent import _resolve_intent
        from egovault.agent.commands import handle_command

        if session_ctx is None:
            session_ctx = {}

        # ------------------------------------------------------------------
        # 1. NL → slash-command intent resolution
        # ------------------------------------------------------------------
        resolved = user_input
        if not user_input.startswith("/"):
            candidate = _resolve_intent(user_input)
            if candidate:
                emit(f"→ {candidate}")
                resolved = candidate

        # ------------------------------------------------------------------
        # 2. Simple command dispatch
        # ------------------------------------------------------------------
        cmd_ctx: dict = {
            "settings": self._settings,
            "sources": session_ctx.get("last_sources", []),
            "owner_profile": session_ctx.get("owner_profile", ""),
            "top_n": session_ctx.get("top_n", 10),
            "bg_threads": session_ctx.get("bg_threads", []),
            "bg_progress": session_ctx.get("bg_progress", {}),
            "scheduler": session_ctx.get("scheduler"),
        }
        cmd_result = handle_command(resolved, cmd_ctx)
        if cmd_result is not None:
            if cmd_result.action == "top_n" and cmd_result.value is not None:
                session_ctx["top_n"] = cmd_result.value
            return TurnResult(
                text=cmd_result.text,
                updated_history=history,
                action=cmd_result.action,
                value=cmd_result.value,
                is_command=True,
            )

        # ------------------------------------------------------------------
        # 3. Heavy / I/O commands — delegate back to the frontend
        # ------------------------------------------------------------------
        lower = resolved.strip().lower()
        _HEAVY_PREFIXES = ("/scan", "/schedule", "/gmail", "/open", "/telegram-")
        if any(lower.startswith(p) for p in _HEAVY_PREFIXES):
            return TurnResult(
                text="",
                updated_history=history,
                action="_delegate",
                value=resolved,
                is_command=True,
            )

        # ------------------------------------------------------------------
        # 4. Agent pipeline
        # ------------------------------------------------------------------
        try:
            answer, updated_history = self.process(
                resolved, history, emit, session_ctx=session_ctx,
            )
        except Exception as exc:  # noqa: BLE001
            return TurnResult(
                text=f"**LLM error:** {exc}",
                updated_history=history,
                action=None,
                is_command=False,
            )

        sources: list[str] = list(session_ctx.get("last_sources", []))
        attachments: list[str] = list(session_ctx.get("saved_attachments", []))

        return TurnResult(
            text=answer,
            updated_history=updated_history,
            action=None,
            is_command=False,
            sources=sources,
            attachments=attachments,
        )


# ---------------------------------------------------------------------------
# TurnResult — return type of AgentSession.process_turn()
# ---------------------------------------------------------------------------

@dataclass
class TurnResult:
    """Result of one agent turn, returned by :meth:`AgentSession.process_turn`.

    Attributes
    ----------
    text:
        Markdown-formatted response string.  Empty for side-effect-only
        actions (``clear``, ``exit``, ``_delegate``).
    updated_history:
        Conversation history after this turn.  For commands that do not
        advance the dialogue (``clear``, ``exit``) this equals the input
        history unchanged.
    action:
        Optional side-effect signal for the frontend:

        - ``None``             — pure text response (render ``text``)
        - ``"exit"``           — end the session
        - ``"clear"``          — reset display + history
        - ``"restart"``        — reset history only
        - ``"top_n"``          — update retrieval depth to ``value``
        - ``"refresh_profile"``— re-extract owner profile (frontend calls LLM)
        - ``"_delegate"``      — heavy I/O command; ``value`` holds the
                                 resolved command string for the frontend to
                                 dispatch (e.g. ``"/scan inbox"``).
    value:
        Payload for the action (``int`` for ``"top_n"``, ``str`` for
        ``"_delegate"``).
    is_command:
        True when the turn was a slash command, False for agent/LLM turns.
    sources:
        Source-attribution strings for the last agent answer.
    attachments:
        File paths of attachments saved during this turn.
    """

    text: str
    updated_history: list
    action: "str | None" = None
    value: "Any" = None
    is_command: bool = False
    sources: list = field(default_factory=list)
    attachments: list = field(default_factory=list)
