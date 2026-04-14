"""Unit tests for egovault.utils.memory_processors."""
from __future__ import annotations

from egovault.utils.memory_processors import ToolCallFilter, TokenLimiter, _estimate_tokens


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _msgs(tool_contents: list[str]) -> list[dict]:
    """Build a minimal messages list with system + user + N tool results."""
    msgs: list[dict] = [
        {"role": "system", "content": "You are Ego."},
        {"role": "user", "content": "Find my emails about Barcelona."},
    ]
    for content in tool_contents:
        msgs.append({"role": "tool", "content": content})
    return msgs


# ---------------------------------------------------------------------------
# ToolCallFilter
# ---------------------------------------------------------------------------

class TestToolCallFilter:
    def test_no_truncation_when_few_results(self) -> None:
        msgs = _msgs(["short result"] * 3)
        out = ToolCallFilter(keep_recent=3).process(msgs)
        assert all(m["content"] == "short result" for m in out if m["role"] == "tool")

    def test_truncates_old_long_results(self) -> None:
        long = "x" * 2000
        msgs = _msgs([long, long, long, long])  # 4 tool results
        out = ToolCallFilter(keep_recent=3).process(msgs)
        tools = [m for m in out if m["role"] == "tool"]
        # First result should be truncated
        assert len(tools[0]["content"]) < len(long)
        assert "truncated" in tools[0]["content"]
        # Last 3 should be intact
        for m in tools[1:]:
            assert m["content"] == long

    def test_does_not_truncate_short_results(self) -> None:
        short = "x" * 100  # below min_length=800
        msgs = _msgs([short] * 5)
        out = ToolCallFilter(keep_recent=3).process(msgs)
        tools = [m for m in out if m["role"] == "tool"]
        assert all(m["content"] == short for m in tools)

    def test_system_and_user_messages_untouched(self) -> None:
        long = "x" * 2000
        msgs = _msgs([long] * 5)
        out = ToolCallFilter(keep_recent=3).process(msgs)
        assert out[0]["content"] == "You are Ego."
        assert out[1]["content"] == "Find my emails about Barcelona."

    def test_returns_same_list_object(self) -> None:
        msgs = _msgs(["result"] * 2)
        out = ToolCallFilter().process(msgs)
        assert out is msgs  # mutates in place and returns same list


# ---------------------------------------------------------------------------
# TokenLimiter
# ---------------------------------------------------------------------------

class TestTokenLimiter:
    def test_no_action_when_disabled(self) -> None:
        long = "x" * 10_000
        msgs = _msgs([long] * 10)
        out = TokenLimiter(max_tokens=0).process(msgs)
        tools = [m for m in out if m["role"] == "tool"]
        assert all(m["content"] == long for m in tools)

    def test_no_action_when_within_limit(self) -> None:
        msgs = _msgs(["short"] * 3)
        original_content = [m["content"] for m in msgs]
        out = TokenLimiter(max_tokens=100_000).process(msgs)
        assert [m["content"] for m in out] == original_content

    def test_summarises_old_results_when_over_limit(self) -> None:
        # 20 tool results × 5000 chars each = 100k chars ≈ 25k tokens
        big = "VAULT DATA: lots of records\n" + "record " * 500
        msgs = _msgs([big] * 20)
        out = TokenLimiter(max_tokens=1_000, keep_recent=5).process(msgs)
        tools = [m for m in out if m["role"] == "tool"]
        # Old results (first 15) should be summarised to 1 line
        for m in tools[:-5]:
            assert "\n" not in m["content"] or m["content"].count("\n") <= 1
            assert "truncated" in m["content"]
        # Most recent 5 should be intact
        for m in tools[-5:]:
            assert m["content"] == big

    def test_tokens_decrease_after_processing(self) -> None:
        big = "SEARCH RESULTS: 10 matches\n" + "detail " * 1000
        msgs = _msgs([big] * 15)
        before = _estimate_tokens(msgs)
        out = TokenLimiter(max_tokens=500, keep_recent=3).process(msgs)
        after = _estimate_tokens(out)
        assert after < before


# ---------------------------------------------------------------------------
# _estimate_tokens
# ---------------------------------------------------------------------------

class TestEstimateTokens:
    def test_empty_messages(self) -> None:
        assert _estimate_tokens([]) == 0

    def test_single_message(self) -> None:
        msgs = [{"role": "user", "content": "a" * 400}]
        assert _estimate_tokens(msgs) == 100  # 400 chars / 4

    def test_tool_calls_counted(self) -> None:
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"arguments": "a" * 400}}],
            }
        ]
        assert _estimate_tokens(msgs) == 100
