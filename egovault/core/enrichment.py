"""LLM enrichment pipeline — chunks records, calls LLM, stores results."""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from egovault.config import LLMSettings, Settings, load_agent_prompts
from egovault.core.schema import EnrichmentStatus, NormalizedRecord
from egovault.core.store import VaultStore
from egovault.utils.llm import call_llm_simple as _call_llm_simple

if TYPE_CHECKING:
    pass  # kept for future conditional imports

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = load_agent_prompts()["enrichment"]

_GEM_PATTERN = re.compile(
    r"^-\s*\[(Link|Decision|Recommendation|Action)\]\s*(.+)$",
    re.IGNORECASE,
)
_URL_PATTERN = re.compile(r"https?://\S+")

_GEM_TYPE_MAP = {
    "link": "link",
    "decision": "decision",
    "recommendation": "recommendation",
    "action": "action",
}

_MAX_RETRIES: int = 3
_RETRY_BACKOFF_BASE: int = 2

# ---------------------------------------------------------------------------
# Contextual Retrieval (P1) — context prefix generation
# ---------------------------------------------------------------------------

_CONTEXT_PREFIX_PROMPT = (
    "Here is a document or message from the user's personal vault:\n"
    "<document>\n"
    "{body}\n"
    "</document>\n\n"
    "Write a SHORT context (2-3 sentences) that situates this chunk within the\n"
    "broader document for the purpose of improving search retrieval.\n"
    "Include: what kind of content it is, what topic or project it relates to,\n"
    "and any key identifiers (name, place, date, technology) present in the text.\n"
    "Reply with ONLY the context — no preamble, no explanation."
)


# ---------------------------------------------------------------------------
# HyPE (P2) — hypothetical question generation
# ---------------------------------------------------------------------------

_HYPE_QUESTIONS_PROMPT = (
    "Here is a document or message from the user's personal vault:\n"
    "<document>\n"
    "{body}\n"
    "</document>\n\n"
    "List exactly 3-5 short questions that a user might type to find THIS specific\n"
    "document when searching their vault.  Questions should vary in phrasing and\n"
    "cover different aspects of the content.\n"
    "Rules:\n"
    "- One question per line, no numbering, no bullet points.\n"
    "- Each question must be plausible given only the document above.\n"
    "- Reply with ONLY the questions \u2014 no preamble, no explanation."
)


def _generate_hype_questions(
    record: NormalizedRecord,
    llm: "LLMSettings",
) -> list[str]:
    """Generate 3-5 hypothetical search questions for *record* via the LLM.

    Returns a list of question strings on success, or an empty list on failure.
    Used by the HyPE (Hypothetical Prompt Embeddings) pipeline: each question
    is embedded and stored so that user queries can be matched against question
    embeddings rather than raw document embeddings.
    """
    body_excerpt = (record.body or "")[:3000]
    if not body_excerpt.strip():
        return []
    user_prompt = _HYPE_QUESTIONS_PROMPT.format(body=body_excerpt)
    try:
        raw = _call_llm_simple(
            base_url=llm.base_url,
            model=llm.model,
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            timeout=llm.timeout_seconds,
            provider=llm.provider,
            api_key=llm.api_key,
        )
    except Exception as exc:
        logger.warning("HyPE question generation failed for %s: %s", record.id, exc)
        return []

    questions = [line.strip() for line in raw.splitlines() if line.strip()]
    # Keep only lines that look like questions (contain a letter; skip headers)
    questions = [q for q in questions if any(c.isalpha() for c in q)][:5]
    return questions


def _generate_context_prefix(
    record: NormalizedRecord,
    llm: "LLMSettings",
) -> str:
    """Generate a short context blurb to prepend to *record.body* for indexing.

    Returns the contextual body (prefix + original body) on success, or an
    empty string on failure (the caller falls back to raw body in that case).
    """
    body_excerpt = (record.body or "")[:3000]  # keep prompt size bounded
    if not body_excerpt.strip():
        return ""
    user_prompt = _CONTEXT_PREFIX_PROMPT.format(body=body_excerpt)
    try:
        prefix = _call_llm_simple(
            base_url=llm.base_url,
            model=llm.model,
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            timeout=llm.timeout_seconds,
            provider=llm.provider,
            api_key=llm.api_key,
        )
        prefix = prefix.strip()
        if not prefix:
            return ""
        # Build the contextual body: context blurb + separator + original body
        return f"{prefix}\n\n{record.body}"
    except Exception as exc:
        logger.warning("Context prefix generation failed for %s: %s", record.id, exc)
        return ""


@dataclass
class EnrichmentResult:
    summary: str
    gems_raw: str
    token_count: int | None
    retries: int


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: 4 chars ≈ 1 token."""
    return max(1, len(text) // 4)


def _chunk_records(
    records: list[NormalizedRecord], target_tokens: int = 2000, overlap: int = 3
) -> list[list[NormalizedRecord]]:
    """Split *records* into chunks of ~*target_tokens* with *overlap* message overlap."""
    if not records:
        return []
    chunks: list[list[NormalizedRecord]] = []
    current: list[NormalizedRecord] = []
    current_tokens = 0

    for rec in records:
        rec_tokens = _estimate_tokens(rec.body)
        if current and current_tokens + rec_tokens > target_tokens:
            chunks.append(current)
            current = current[-overlap:] if overlap else []
            current_tokens = sum(_estimate_tokens(r.body) for r in current)
        current.append(rec)
        current_tokens += rec_tokens

    if current:
        chunks.append(current)
    return chunks


# Max chars sent to the LLM per record body (~8 000 chars ≈ 2 000 tokens).
# Prevents HTTP 400 from llama-server when a single record is very large.
_MAX_BODY_CHARS = 8_000


def _build_user_prompt(records: list[NormalizedRecord]) -> str:
    lines: list[str] = []
    for rec in records:
        ts = rec.timestamp.strftime("%Y-%m-%d %H:%M")
        body = (rec.body or "")[:_MAX_BODY_CHARS]
        if rec.sender_name and rec.sender_name != "user":
            lines.append(f"[{ts}] {rec.sender_name}: {body}")
        else:
            lines.append(f"[{ts}] {body}")
    return "\n".join(lines)


def _parse_response(response: str) -> tuple[str, str]:
    """Return (summary, gems_raw) extracted from LLM output."""
    summary = ""
    gems_lines: list[str] = []
    in_gems = False

    for line in response.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("SUMMARY:"):
            summary = stripped[len("SUMMARY:"):].strip()
            in_gems = False
        elif stripped.upper() == "GEMS:":
            in_gems = True
        elif in_gems and stripped.startswith("-"):
            gems_lines.append(stripped)

    return summary, "\n".join(gems_lines)


def _parse_gems(gems_raw: str) -> list[dict]:
    """Parse gem lines into structured dicts."""
    gems: list[dict] = []
    for line in gems_raw.splitlines():
        m = _GEM_PATTERN.match(line.strip())
        if not m:
            continue
        gem_type = _GEM_TYPE_MAP.get(m.group(1).lower(), "link")
        content = m.group(2).strip()
        url_match = _URL_PATTERN.search(content) if gem_type == "link" else None
        gems.append(
            {
                "gem_type": gem_type,
                "content": content,
                "url": url_match.group(0) if url_match else None,
                "attributed_to": None,
                "attributed_at": None,
            }
        )
    return gems


class EnrichmentPipeline:
    def __init__(self, store: VaultStore, settings: Settings) -> None:
        self._store = store
        self._settings = settings

    # ------------------------------------------------------------------
    # Private helpers (SRP: each has one job)
    # ------------------------------------------------------------------

    def _call_with_retry(self, llm: LLMSettings, user_prompt: str) -> tuple[str, int]:
        """Call the LLM with retry logic. Returns (response_text, retry_count)."""
        import urllib.error as _ue
        retries = 0
        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                response = _call_llm_simple(
                    base_url=llm.base_url,
                    model=llm.model,
                    system_prompt=_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    timeout=llm.timeout_seconds,
                    provider=llm.provider,
                    api_key=llm.api_key,
                )
                return response, retries
            except _ue.HTTPError as exc:
                last_error = exc
                retries = attempt + 1
                # 400 Bad Request is permanent (prompt too long / malformed).
                # Retrying the identical payload will never succeed.
                if exc.code == 400:
                    logger.warning(
                        "LLM call failed (HTTP 400 — prompt too long or malformed) "
                        "for record, skipping retries."
                    )
                    break
                wait = _RETRY_BACKOFF_BASE ** attempt
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s — retrying in %ds",
                    attempt + 1, _MAX_RETRIES, exc, wait,
                )
                time.sleep(wait)
            except Exception as exc:
                last_error = exc
                retries = attempt + 1
                wait = _RETRY_BACKOFF_BASE ** attempt
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s — retrying in %ds",
                    attempt + 1, _MAX_RETRIES, exc, wait,
                )
                time.sleep(wait)
        raise RuntimeError(
            f"LLM call failed after {_MAX_RETRIES} attempts"
        ) from last_error

    def _store_enrichment(
        self,
        record: NormalizedRecord,
        llm: LLMSettings,
        response: str,
        user_prompt: str,
        retries: int,
    ) -> None:
        """Write enrichment result and extracted gems to the store."""
        summary, gems_raw = _parse_response(response)
        token_count = _estimate_tokens(user_prompt)
        self._store.insert_enrichment_result(
            record_id=record.id,
            model=llm.model,
            summary=summary,
            gems_raw=gems_raw,
            token_count=token_count,
            retries=retries,
        )
        for gem in _parse_gems(gems_raw):
            self._store.insert_gem(record_id=record.id, **gem)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _enrich_image(self, record: NormalizedRecord) -> bool:
        """Describe an image record via the LLM vision API and store the result.

        Called from ``enrich_record`` for records with ``record_type == "image"``
        and an empty body.  Uses the same llama-server endpoint as text enrichment
        (multimodal message format with a base64 data-URI image_url part).

        Returns True on success (body updated, record marked DONE),
        False if vision is not supported or the file is missing/unreadable.
        """
        from pathlib import Path as _Path
        from egovault.adapters.local_inbox import _describe_image, _is_vision_supported

        llm = self._settings.llm
        if not _is_vision_supported(llm.base_url):
            # Vision not available — mark as skipped so we don't keep retrying.
            self._store.mark_enriched(record.id, EnrichmentStatus.SKIPPED)
            return False

        # Determine the image path — prefer file_path, fall back to first attachment.
        img_path_str = record.file_path or (
            record.attachments[0] if record.attachments else None
        )
        if not img_path_str:
            logger.warning("Image record %s has no file_path or attachments", record.id)
            self._store.mark_enriched(record.id, EnrichmentStatus.FAILED)
            return False

        img_path = _Path(img_path_str)
        if not img_path.exists():
            logger.warning("Image file not found, skipping vision: %s", img_path)
            self._store.mark_enriched(record.id, EnrichmentStatus.SKIPPED)
            return False

        description = _describe_image(img_path)
        if not description:
            logger.warning("Vision description returned empty for %s", img_path.name)
            self._store.mark_enriched(record.id, EnrichmentStatus.FAILED)
            return False

        # Prepend the file path header (same format as text records) and store.
        path_header = f"File: {img_path}\n---\n"
        full_body = path_header + description
        self._store.update_record_body(record.id, full_body)
        # Also store as enrichment result so the rest of the pipeline (embed,
        # chunk, etc.) treats it as fully enriched.
        self._store.insert_enrichment_result(
            record_id=record.id,
            model=llm.model,
            summary=description[:500],
            gems_raw="",
            token_count=len(description) // 4,
            retries=0,
        )
        self._store.mark_enriched(record.id, EnrichmentStatus.DONE)
        logger.info("Vision description stored for %s", img_path.name)
        return True

    def enrich_record(self, record: NormalizedRecord) -> bool:
        """Enrich a single record. Returns True on success, False on failure."""
        llm = self._settings.llm

        # ── Image records: generate vision description instead of text enrichment ──
        # Images have an empty body at ingest time (description is deferred here
        # so that scanning a folder is fast and non-blocking).
        if record.record_type == "image" and not (record.body or "").strip():
            return self._enrich_image(record)

        user_prompt = _build_user_prompt([record])

        try:
            response, retries = self._call_with_retry(llm, user_prompt)
        except RuntimeError as exc:
            logger.error("Enrichment failed for %s: %s", record.id, exc)
            self._store.mark_enriched(record.id, EnrichmentStatus.FAILED)
            return False

        self._store_enrichment(record, llm, response, user_prompt, retries)

        # Contextual Retrieval (P1): generate context prefix when enabled.
        # We do this inside enrich_record so the LLM is already warm.
        if getattr(self._settings.embeddings, "contextual_enabled", False):
            contextual_body = _generate_context_prefix(record, llm)
            if contextual_body:
                self._store.upsert_contextual_body(record.id, contextual_body)

        self._store.mark_enriched(record.id, EnrichmentStatus.DONE)
        return True

    def contextualize_record(self, record: NormalizedRecord) -> bool:
        """Generate and store a context prefix for a single already-enriched record.

        Used by `egovault context` to backfill contextual_body on existing records
        without re-running the full enrichment pipeline.
        Returns True on success.
        """
        contextual_body = _generate_context_prefix(record, self._settings.llm)
        if contextual_body:
            self._store.upsert_contextual_body(record.id, contextual_body)
            return True
        return False

    def enrich_all(self, limit: int = 500) -> tuple[int, int]:
        """Enrich all pending records. Returns (succeeded, failed)."""
        records = self._store.get_unenriched_records(limit=limit)
        ok = fail = 0
        for rec in records:
            if self.enrich_record(rec):
                ok += 1
            else:
                fail += 1
        return ok, fail

    def contextualize_all(self, limit: int = 5000) -> tuple[int, int]:
        """Backfill contextual_body for all records that lack one.

        Returns (succeeded, failed).  Safe to re-run: already-contextualized
        records are skipped by ``get_uncontextualized_record_ids``.
        """
        record_ids = self._store.get_uncontextualized_record_ids(limit=limit)
        if not record_ids:
            return 0, 0
        # Fetch full records for the IDs so we have the body available
        id_set = set(record_ids)
        all_records = self._store.get_records()
        pending = [r for r in all_records if r.id in id_set]
        ok = fail = 0
        for rec in pending:
            if self.contextualize_record(rec):
                ok += 1
            else:
                fail += 1
        return ok, fail
