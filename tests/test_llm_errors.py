"""Unit tests for egovault.utils.llm_errors."""
from __future__ import annotations

import urllib.error


from egovault.utils.llm_errors import classify_llm_error


# ---------------------------------------------------------------------------
# Helpers — minimal exception stubs
# ---------------------------------------------------------------------------

class _HttpErr(Exception):
    def __init__(self, code: int, msg: str = "") -> None:
        super().__init__(msg)
        self.status_code = code


class _UrlErr(urllib.error.URLError):
    pass


# ---------------------------------------------------------------------------
# HTTP status codes
# ---------------------------------------------------------------------------

class TestStatusCodes:
    def test_401_auth_error(self) -> None:
        code, msg = classify_llm_error(_HttpErr(401))
        assert code == "auth_error"
        assert "api key" in msg.lower() or "check" in msg.lower()

    def test_429_rate_limit(self) -> None:
        code, msg = classify_llm_error(_HttpErr(429))
        assert code == "rate_limit"

    def test_500_oom(self) -> None:
        code, msg = classify_llm_error(_HttpErr(500, "out of memory: failed to alloc"))
        assert code == "oom"
        assert "VRAM" in msg or "vram" in msg.lower()

    def test_500_generic_server_error(self) -> None:
        code, msg = classify_llm_error(_HttpErr(500, "internal error"))
        assert code == "server_error"

    def test_503_server_unavailable(self) -> None:
        code, msg = classify_llm_error(_HttpErr(503))
        assert code == "server_error"

    def test_400_context_overflow(self) -> None:
        code, msg = classify_llm_error(_HttpErr(400, "context length exceeded"))
        assert code == "context_overflow"

    def test_404_model_not_found(self) -> None:
        code, msg = classify_llm_error(_HttpErr(404))
        assert code == "model_load"


# ---------------------------------------------------------------------------
# Network/connection errors
# ---------------------------------------------------------------------------

class TestNetworkErrors:
    def test_url_error_connection_refused(self) -> None:
        exc = _UrlErr(reason=ConnectionRefusedError())
        code, msg = classify_llm_error(exc)
        assert code == "network_error"
        assert "llama-server" in msg.lower() or "server" in msg.lower()

    def test_connection_refused_error_type(self) -> None:
        code, msg = classify_llm_error(ConnectionRefusedError("connection refused"))
        assert code == "network_error"

    def test_url_error_string_connection_refused(self) -> None:
        exc = _UrlErr("[Errno 111] connection refused")
        code, msg = classify_llm_error(exc)
        assert code == "network_error"


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------

class TestTimeout:
    def test_timeout_error_type(self) -> None:
        code, _msg = classify_llm_error(TimeoutError("timed out"))
        assert code == "timeout"

    def test_timeout_in_string(self) -> None:
        code, _msg = classify_llm_error(Exception("request timed out after 300s"))
        assert code == "timeout"


# ---------------------------------------------------------------------------
# OOM / model load (string-based detection)
# ---------------------------------------------------------------------------

class TestOomAndModelLoad:
    def test_oom_string(self) -> None:
        code, _msg = classify_llm_error(Exception("ggml_cuda_pool_alloc: out of memory"))
        assert code == "oom"

    def test_model_not_loaded(self) -> None:
        code, _msg = classify_llm_error(Exception("model not loaded"))
        assert code == "model_load"

    def test_context_overflow_string(self) -> None:
        code, _msg = classify_llm_error(Exception("kv cache full"))
        assert code == "context_overflow"


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------

class TestFallback:
    def test_unknown_returns_unknown_code(self) -> None:
        code, msg = classify_llm_error(Exception("something completely unexpected"))
        assert code == "unknown_error"
        assert len(msg) > 0
