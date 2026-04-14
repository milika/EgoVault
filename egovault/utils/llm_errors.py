"""Structured error classification for LLM endpoint failures.

``classify_llm_error(exc)`` maps any exception raised by the LLM call path
to a typed ``(error_code, user_message)`` pair.

Classification priority (first match wins):

    oom               → llama-server ran out of VRAM
    model_load        → model not yet loaded or still warming up
    context_overflow  → prompt too long for the model's context window
    server_error      → 5xx from llama-server / provider (transient)
    rate_limit        → 429 Too Many Requests
    auth_error        → 401 / 403 API key problem
    timeout           → request timed out
    network_error     → connection refused / DNS failure
    unknown_error     → anything else

All error codes are stable strings so callers can branch on them::

    code, msg = classify_llm_error(exc)
    if code == "network_error":
        # maybe retry after a short sleep
        ...
    emit(msg)  # show to user
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def classify_llm_error(exc: BaseException) -> tuple[str, str]:
    """Classify *exc* into ``(error_code, user_message)``.

    Checks structured attributes first (``status_code``, ``code``), then
    targeted string matching against the stringified exception.  Never raises.
    """
    exc_type = type(exc).__name__
    exc_str = str(exc).lower()

    # ── Structured HTTP status code ──────────────────────────────────────────
    # urllib.error.HTTPError carries a .code attribute.
    # httpx.HTTPStatusError and litellm exceptions carry .status_code.
    status_code: int | None = None
    for attr in ("status_code", "code"):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            status_code = val
            break

    if status_code is not None:
        if status_code == 400:
            # llama-server returns 400 with "context" in the body when kv-cache is full
            if any(k in exc_str for k in ("context", "token", "exceed", "limit")):
                return (
                    "context_overflow",
                    "The conversation is too long for the model. "
                    "Start a new session or reduce max_results.",
                )
        if status_code == 401:
            return (
                "auth_error",
                "API key error. Check [llm] api_key in egovault.toml.",
            )
        if status_code == 403:
            return (
                "auth_error",
                "Access denied. Check [llm] api_key and endpoint permissions.",
            )
        if status_code == 404:
            return (
                "model_load",
                "The model was not found at the endpoint. "
                "Is llama-server running with the correct model?",
            )
        if status_code == 429:
            return (
                "rate_limit",
                "Rate limited by the LLM endpoint. Wait a moment and try again.",
            )
        if status_code == 500:
            # OOM from llama-server typically includes "ggml" / "cuda" / "memory"
            if any(k in exc_str for k in ("out of memory", "oom", "cuda", "ggml", "alloc")):
                return (
                    "oom",
                    "The model ran out of VRAM. "
                    "Try a smaller model, reduce ctx_size, or close other GPU applications.",
                )
            return (
                "server_error",
                "llama-server returned an internal error. Check the server log for details.",
            )
        if status_code in (502, 503, 504):
            return (
                "server_error",
                "The LLM server is temporarily unavailable. "
                "Is llama-server running? Run `egovault chat` to restart it.",
            )

    # ── OOM (string-based — appears in URLError reason or raw exception) ──
    if any(k in exc_str for k in ("out of memory", "oom", "cuda error", "ggml_", "alloc failed")):
        return (
            "oom",
            "The model ran out of VRAM. "
            "Try a smaller model, reduce ctx_size, or close other GPU applications.",
        )

    # ── Model not loaded / still warming up ──────────────────────────────────
    if any(k in exc_str for k in ("model not loaded", "no model", "not ready", "loading model")):
        return (
            "model_load",
            "The model is still loading. Wait a moment and try again.",
        )

    # ── Context / token overflow ─────────────────────────────────────────────
    if any(
        k in exc_str
        for k in ("context length", "context_length", "token limit", "max_tokens", "kv cache full")
    ):
        return (
            "context_overflow",
            "The conversation is too long for the model's context window. "
            "Start a new session or reduce max_results.",
        )

    # ── Rate limiting ────────────────────────────────────────────────────────
    if ("rate" in exc_str and "limit" in exc_str) or "too many requests" in exc_str:
        return (
            "rate_limit",
            "Rate limited by the LLM endpoint. Wait a moment and try again.",
        )

    # ── Authentication ───────────────────────────────────────────────────────
    if any(k in exc_str for k in ("unauthorized", "unauthenticated", "invalid api key", "forbidden")):
        return (
            "auth_error",
            "API key error. Check [llm] api_key in egovault.toml.",
        )

    # ── Timeout ──────────────────────────────────────────────────────────────
    if exc_type in ("TimeoutError", "ReadTimeout", "ConnectTimeout", "socket.timeout"):
        return (
            "timeout",
            "The model took too long to respond. "
            "Try again or increase [llm] timeout_seconds in egovault.toml.",
        )
    if "timed out" in exc_str or "timeout" in exc_str:
        return (
            "timeout",
            "The model took too long to respond. "
            "Try again or increase [llm] timeout_seconds in egovault.toml.",
        )

    # ── Network / connection ─────────────────────────────────────────────────
    # urllib.error.URLError wraps ConnectionRefusedError when the server is down.
    if exc_type in ("URLError", "ConnectionRefusedError", "ConnectionError", "ConnectError"):
        return (
            "network_error",
            "Cannot reach the LLM server. "
            "Is llama-server running? Check [llm] base_url in egovault.toml.",
        )
    if any(
        k in exc_str
        for k in ("connection refused", "name resolution", "unreachable", "no route to host")
    ):
        return (
            "network_error",
            "Cannot reach the LLM server. "
            "Is llama-server running? Check [llm] base_url in egovault.toml.",
        )
    # urllib.error.URLError with ConnectionRefusedError as reason
    reason = getattr(exc, "reason", None)
    if reason is not None and type(reason).__name__ == "ConnectionRefusedError":
        return (
            "network_error",
            "Cannot reach the LLM server. "
            "Is llama-server running? Check [llm] base_url in egovault.toml.",
        )

    # ── Generic server error (phrases) ───────────────────────────────────────
    if any(k in exc_str for k in ("internal server error", "service unavailable", "bad gateway")):
        return (
            "server_error",
            "The LLM server is temporarily unavailable. Try again in a moment.",
        )

    # ── Fallback ─────────────────────────────────────────────────────────────
    logger.debug("classify_llm_error: unclassified %s: %s", exc_type, exc_str[:200])
    return (
        "unknown_error",
        f"LLM error ({exc_type}). Is llama-server running? Check logs for details.",
    )
