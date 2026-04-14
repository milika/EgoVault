"""Context memory processors for the EgoVault agentic tool-calling loop.

Two composable processors trim the message list passed to the LLM on every
iteration of ``_call_llm_agent``, preventing context overflow and silent
truncation on local models with limited context windows (8k–32k tokens).

Apply them in sequence before each LLM call::

    messages = ToolCallFilter(keep_recent=3).process(messages)
    messages = TokenLimiter(max_tokens=24_000, keep_recent=5).process(messages)

Both processors treat the first two messages (system + first user turn) as
sacred and will never truncate them.  All other ``role == "tool"`` messages
are candidates for trimming, oldest first.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Chars per estimated token — avoids importing a tokenizer dependency.
_CHARS_PER_TOKEN = 4

# Max chars to keep at start / end of a long tool result when stripping.
_KEEP_HEAD = 300
_KEEP_TAIL = 100


def _estimate_tokens(messages: list[dict]) -> int:
    """Rough token estimate: total chars across all message content // 4."""
    total = 0
    for m in messages:
        content = m.get("content") or ""
        total += len(content)
        # tool_calls arg strings also consume tokens
        for tc in m.get("tool_calls") or []:
            total += len(str(tc.get("function", {}).get("arguments", "")))
    return total // _CHARS_PER_TOKEN


def _tool_indices(messages: list[dict]) -> list[int]:
    """Return indices of all role=='tool' messages, oldest first."""
    return [i for i, m in enumerate(messages) if m.get("role") == "tool"]


def _summarise_tool_content(content: str) -> str:
    """Return a compact 1-line summary of a tool result string.

    Preserves the leading status token (e.g. 'VAULT DATA', 'SEARCH RESULTS',
    'Error') so the LLM knows the call succeeded or failed.
    """
    first_line = content.split("\n", 1)[0].strip()
    chars = len(content)
    return f"{first_line} [{chars} chars — truncated to save context]"


class ToolCallFilter:
    """Strip verbose content from old tool results.

    Keeps the ``keep_recent`` most recent tool results fully intact.
    For older results, truncates the content to the first ``_KEEP_HEAD``
    chars + last ``_KEEP_TAIL`` chars with a ``[…truncated…]`` marker.

    Only truncates if the result body exceeds ``min_length`` chars —
    short results (errors, confirmations) are always kept verbatim.
    """

    def __init__(self, keep_recent: int = 3, min_length: int = 800) -> None:
        self.keep_recent = keep_recent
        self.min_length = min_length

    def process(self, messages: list[dict]) -> list[dict]:
        indices = _tool_indices(messages)
        if len(indices) <= self.keep_recent:
            return messages

        old = indices[: -self.keep_recent]
        stripped = 0
        for i in old:
            content = messages[i].get("content") or ""
            if len(content) <= self.min_length:
                continue
            short = (
                content[:_KEEP_HEAD]
                + "\n…[truncated — start new session or use get_record to re-fetch]…\n"
                + content[-_KEEP_TAIL:]
            )
            messages[i] = {**messages[i], "content": short}
            stripped += 1

        if stripped:
            logger.debug("ToolCallFilter: truncated %d old tool results", stripped)
        return messages


class TokenLimiter:
    """Cap total context size by summarising old tool results.

    Uses character-based token estimation (chars // 4) — no tokenizer
    dependency needed.  When the estimated token count exceeds ``max_tokens``,
    old tool result bodies (beyond the most recent ``keep_recent``) are
    replaced with 1-line summaries.

    Set ``max_tokens = 0`` to disable.
    """

    def __init__(self, max_tokens: int = 24_000, keep_recent: int = 5) -> None:
        self.max_tokens = max_tokens
        self.keep_recent = keep_recent

    def process(self, messages: list[dict]) -> list[dict]:
        if not self.max_tokens:
            return messages

        estimated = _estimate_tokens(messages)
        if estimated <= self.max_tokens:
            return messages

        indices = _tool_indices(messages)
        if len(indices) <= self.keep_recent:
            return messages  # can't trim further

        old = indices[: -self.keep_recent]
        truncated = 0
        for i in old:
            content = messages[i].get("content") or ""
            if not content:
                continue
            messages[i] = {**messages[i], "content": _summarise_tool_content(content)}
            truncated += 1

        if truncated:
            new_est = _estimate_tokens(messages)
            logger.info(
                "TokenLimiter: summarised %d old results. tokens: ~%d → ~%d",
                truncated,
                estimated,
                new_est,
            )
        return messages
