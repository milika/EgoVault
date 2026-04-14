"""Shared low-level HTTP helpers for calling LLM endpoints.

All providers use the OpenAI-compatible wire protocol:
  POST /v1/chat/completions  →  response: ``choices[0].message.content``

This module is the single source of truth for that logic so it is not
duplicated across ``egovault.core.enrichment`` and ``egovault.chat.session``.
"""
from __future__ import annotations

import json
import logging
import urllib.request

logger = logging.getLogger(__name__)

_DEFAULT_TOP_N = 10


def auto_top_n(chunk_target_tokens: int = 2000) -> int:
    """Return a sensible default top-N chunk count for retrieval.

    Returns a fixed conservative default of ``_DEFAULT_TOP_N`` (10).
    Previously this was VRAM-adaptive; with llama-server the context window is
    set at server startup so a fixed value is appropriate.
    """
    return _DEFAULT_TOP_N


def query_total_vram_mb() -> int | None:
    """Return total GPU VRAM in MB via nvidia-smi, or None if unavailable."""
    import subprocess as _sp
    try:
        result = _sp.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            line = result.stdout.strip().splitlines()[0]
            return int(line.strip())
    except Exception:
        pass
    return None


def query_free_vram_mb() -> int | None:
    """Return currently free GPU VRAM in MB via nvidia-smi, or None if unavailable.

    Querying *free* (not total) VRAM means Windows/CUDA/other-program overhead is
    already subtracted — no manual correction needed when computing ctx-size.
    """
    import subprocess as _sp
    try:
        result = _sp.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            line = result.stdout.strip().splitlines()[0]
            return int(line.strip())
    except Exception:
        pass
    return None


def ctx_for_vram_budget(
    total_vram_mb: int,
    model_size_mb: int = 0,
    budget_pct: float = 0.80,
    flash_attn: bool = True,
    cuda_overhead_mb: int = 0,
) -> int:
    """Compute a ctx-size that fits within *budget_pct* of *total_vram_mb*.

    Pass *free* VRAM (from ``query_free_vram_mb()``) as *total_vram_mb* with
    ``model_size_mb=0`` and ``cuda_overhead_mb=0`` to let the model load
    unrestricted and allocate *budget_pct* of whatever VRAM is currently free
    to the KV cache::

        available_kv = free_vram_mb × budget_pct
        ctx          = available_kv / kv_mb_per_token  → nearest power of 2

    Empirical KV-cache cost per token:
      - without flash-attn : ~0.125 MB / token  (typical 7–13 B model)
      - with    flash-attn : ~0.0625 MB / token

    Returns the nearest power-of-2 ctx-size from 512 to 65536.

    Example — RTX 3080 Ti (free ≈ 10491 MB), e2b model, flash-attn on::

        available_kv = 10491 × 0.80 = 8393 MB
        raw_ctx      = 8393 / 0.0625 = 134288 → capped + rounded to 65536
    """
    budget_mb = total_vram_mb * budget_pct
    available_kv_mb = budget_mb - model_size_mb - cuda_overhead_mb
    if available_kv_mb <= 128:
        return 512  # not enough VRAM — use minimum safe value
    kv_mb_per_token = 0.0625 if flash_attn else 0.125
    raw_ctx = int(available_kv_mb / kv_mb_per_token)
    # Round down to nearest power of 2, clamped to [512, 65536]
    size = 512
    while size * 2 <= min(raw_ctx, 65536):
        size *= 2
    return size


def call_llm_simple(
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout: int,
    provider: str = "llama_cpp",
    api_key: str = "",
) -> str:
    """Send a single system+user turn to the LLM and return the text response.

    Used by the enrichment pipeline, context-prefix generation, and HyPE
    question generation — anywhere a one-shot prompt is sufficient.

    Returns the assistant content string.
    Raises ``urllib.error.URLError`` / ``KeyError`` on network or parse failure.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    return call_llm_chat(
        base_url=base_url,
        model=model,
        messages=messages,
        timeout=timeout,
        provider=provider,
        api_key=api_key,
    )


def call_llm_chat(
    base_url: str,
    model: str,
    messages: list[dict],
    timeout: int,
    provider: str = "llama_cpp",
    api_key: str = "",
    # Legacy params kept for call-site compatibility but ignored:
    num_gpu: int = -1,      # noqa: ARG001
    num_thread: int = 0,    # noqa: ARG001
    num_ctx: int = 0,       # noqa: ARG001
    auto_ctx: bool = False, # noqa: ARG001
) -> str:
    """Send a full message history to the LLM and return the text response.

    Uses the OpenAI-compatible POST /v1/chat/completions wire protocol.
    All providers (llama_cpp, openai, etc.) speak this protocol.

    Returns the assistant content string.
    Raises ``urllib.error.URLError`` / ``KeyError`` on network or parse failure.
    """
    body: dict = {"model": model, "messages": messages, "stream": False}
    payload = json.dumps(body).encode("utf-8")
    url = base_url.rstrip("/") + "/v1/chat/completions"
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        data = json.loads(resp.read().decode("utf-8"))

    return data["choices"][0]["message"]["content"] or ""

