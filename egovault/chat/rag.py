"""FTS5 retrieval and context assembly for the RAG chat pipeline."""
from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from egovault.config import EmbeddingSettings, RerankerSettings, load_agent_prompts
from egovault.core.schema import NormalizedRecord
from egovault.core.store import VaultStore, row_to_record

if TYPE_CHECKING:
    from egovault.config import CRAGSettings, SentenceWindowSettings

logger = logging.getLogger(__name__)

_LEVEL_0_SYSTEM_PROMPT = load_agent_prompts()["chat"]
_QUERY_PLANNER_PROMPT = load_agent_prompts().get("query_planner", "")


@dataclass
class RetrievedChunk:
    record: NormalizedRecord
    rank: float
    snippet: str = ""  # keyword-in-context excerpt; empty means use full body


def _kwic_snippet(body: str, keywords: list[str], window: int = 300, max_hits: int = 20) -> str:
    """Return ALL keyword hits in *body* as context windows, up to *max_hits*.

    Each hit is a ~*window* character excerpt centred on the match, with
    ``[…]`` markers. Consecutive overlapping hits are merged.
    Falls back to the first *window* chars when no keyword is found.
    """
    if not body:
        return ""
    lower_body = body.lower()

    # Collect (start, end) intervals for every keyword hit
    intervals: list[tuple[int, int]] = []
    for kw in keywords:
        kw_lower = kw.lower()
        kw_len = len(kw_lower)
        search_from = 0
        while True:
            pos = lower_body.find(kw_lower, search_from)
            if pos == -1:
                break
            start = max(0, pos - window // 2)
            end = min(len(body), pos + kw_len + window // 2)
            intervals.append((start, end))
            search_from = pos + kw_len
            if len(intervals) >= max_hits:
                break
        if len(intervals) >= max_hits:
            break

    if not intervals:
        # No keyword found — return start of body
        return body[:window].rstrip() + (" […]" if len(body) > window else "")

    # Sort and merge overlapping/adjacent intervals
    intervals.sort()
    merged: list[tuple[int, int]] = [intervals[0]]
    for s, e in intervals[1:]:
        if s <= merged[-1][1] + _KWIC_MERGE_GAP:  # merge if gap < _KWIC_MERGE_GAP chars
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    parts: list[str] = []
    for s, e in merged:
        prefix = "[…] " if s > 0 else ""
        suffix = " […]" if e < len(body) else ""
        parts.append(prefix + body[s:e].strip() + suffix)

    return "\n---\n".join(parts)


_STOP_WORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "shall", "can", "i", "me", "my",
        "we", "us", "our", "you", "your", "he", "she", "it", "they", "them",
        "their", "what", "which", "who", "whom", "this", "that", "these",
        "those", "am", "not", "no", "so", "if", "as", "up", "out", "about",
        "into", "than", "then", "just", "also", "all", "any", "how", "when",
        "where", "why", "there", "here", "get", "got", "like", "show", "tell",
        "find", "list", "give", "make", "know", "see",
        # Personal-question words that match noise in code files
        "name", "whats", "whos", "age", "call",
        # Meta-query words (the user's instruction, not the topic)
        "search", "file", "files", "mention", "mentions", "mentioned",
        "contains", "contain", "look", "looking", "check", "every", "everything",
        "please", "want", "need", "query", "scan", "across",
    }
)


def _sanitize_query(query: str) -> str:
    """Convert a natural-language query into a safe FTS5 OR expression.

    Removes FTS5 special characters and common English stop words, then joins
    the remaining meaningful tokens with OR so that records only need to
    contain *any* of the keywords rather than *all* of them.
    Returns an empty string when the query contains only stop words — in that
    case the caller skips the vault search and the LLM answers from its own
    knowledge.
    """
    # Remove FTS5 operators/quotes to avoid injection and syntax errors.
    # Hyphen must be included: FTS5 treats it as a NOT/column operator.
    cleaned = re.sub(r'["\*\^\(\)\{\}\[\]\|\\/:?!.,;\-]', " ", query)
    tokens = [t.lower() for t in cleaned.split() if len(t) > 1]
    keywords = [t for t in tokens if t not in _STOP_WORDS]
    # Return FTS5 OR expression: "arduino OR projects OR ..."
    return " OR ".join(keywords) if keywords else ""


# Path segments that indicate generated/library/cache content not useful as RAG context.
# Stored as POSIX (forward-slash) only — _path_score normalises the input path first.
_NOISE_PATH_SEGMENTS: tuple[str, ...] = (
    # Build artifacts
    "build/intermediates",
    "build/outputs",
    "build/tmp",
    "__content__",
    # Android generated sources / build dirs
    "build/generated",
    "build/transformed",
    "build/kotlin",
    "build/source",
    "not_namespaced_r_class",
    "r_class_sources",
    "processDebugResources",
    "processReleaseResources",
    "/r/android",
    "/r/androidx",
    "/r/com/google",
    "databinding",
    # Package managers / deps
    "node_modules",
    "site-packages",
    ".gradle",
    ".m2/repository",
    # IDE internals
    ".idea",
    "/.git/",
    "/__pycache__/",
    # Arduino library packages (downloaded, not user projects)
    "/libraries/",
    # Arduino IDE plugin dirs
    ".arduinoide",
    ".antigravity",
    "packages/arduino",
    "packages/digistump",
    "packages/esp",
    # Windows system / user cache dirs
    "appdata/local",
    "appdata/roaming",
    # VS Code / Cursor AI tooling dirs
    ".vscode/extensions",
    ".cursor/",
    # Language toolchain dirs
    ".rustup/",
    ".cargo/",
    ".pyenv/",
    # OneDrive synced library copies
    "onedrive/documents/arduino",
)

# Relative path-depth bonus: shorter paths → project root files → higher priority.
# Extension bonuses: .ino (project entry) > README/info > source code > other.
def _path_score(file_path: str | None) -> float:
    """Return a float bonus to add to FTS5 rank (rank is negative; closer to 0 = better).

    Positive return value improves a result's standing.
    """
    if not file_path:
        return 0.0
    lp = file_path.lower().replace("\\", "/")
    score = 0.0
    # Penalise noise paths heavily so they fall below the cutoff.
    for seg in _NOISE_PATH_SEGMENTS:
        if seg in lp:
            return -1000.0
    # Reward short path depth (fewer '/' → closer to project root)
    depth = lp.count("/")
    score += max(0, 15 - depth) * 0.5
    # Extension bonuses
    if lp.endswith(".ino"):
        score += 20.0
    elif any(lp.endswith(x) for x in ("/readme.md", "/readme.txt", "/info.txt", "/readme")):
        score += 10.0
    elif lp.endswith((".md", ".txt")):
        score += 3.0
    return score


# ---------------------------------------------------------------------------
# Two-stage reranking (Stage 2: score candidates after FTS5 retrieval)
# ---------------------------------------------------------------------------

# Cache loaded cross-encoder models so we pay the model-load cost only once.
_cross_encoder_cache: dict[str, object] = {}


def _bm25_rerank(chunks: list["RetrievedChunk"], query: str) -> list["RetrievedChunk"]:
    """BM25 re-rank *chunks* against *query* — pure Python, no extra dependencies.

    Computes BM25 scores using corpus statistics derived from the candidate set
    itself.  Better than raw FTS5 rank because it accounts for document-length
    normalisation and treats each unique query term independently.
    """
    import math

    K1 = 1.5
    B = 0.75

    q_tokens = re.findall(r"\w+", query.lower())
    if not q_tokens:
        return chunks

    corpus: list[list[str]] = []
    for chunk in chunks:
        # Include thread_name so records whose filename matches the query (e.g.
        # "Yoris-EmiratesTicket.pdf") rank correctly even when the body text
        # uses a concatenated form ("YORISMR") that doesn't tokenise to the query term.
        text = chunk.snippet or chunk.record.body or ""
        thread = chunk.record.thread_name or ""
        combined = text + " " + thread
        corpus.append(re.findall(r"\w+", combined.lower()))

    N = len(corpus)
    if N == 0:
        return chunks

    avgdl = sum(len(doc) for doc in corpus) / N

    # IDF per unique query term (smoothed BM25 IDF)
    idf: dict[str, float] = {}
    for term in set(q_tokens):
        df = sum(1 for doc in corpus if term in doc)
        idf[term] = math.log((N - df + 0.5) / (df + 0.5) + 1)

    for chunk, doc_tokens in zip(chunks, corpus):
        tf_map: dict[str, int] = {}
        for tok in doc_tokens:
            tf_map[tok] = tf_map.get(tok, 0) + 1
        dl = len(doc_tokens)
        score = 0.0
        for term in q_tokens:
            tf = tf_map.get(term, 0)
            score += idf.get(term, 0.0) * (tf * (K1 + 1)) / (
                tf + K1 * (1 - B + B * dl / max(1, avgdl))
            )
        chunk.rank = score

    return sorted(chunks, key=lambda c: c.rank, reverse=True)


def _crossencoder_rerank(
    chunks: list["RetrievedChunk"], query: str, model_name: str
) -> list["RetrievedChunk"]:
    """Cross-encoder re-rank using sentence-transformers.

    Included in ``pip install egovault`` (sentence-transformers + torch).
    Model is cached in memory after the first load.

    Raises ImportError when sentence-transformers is not installed.
    """
    from sentence_transformers import CrossEncoder  # type: ignore[import-untyped]

    global _cross_encoder_cache
    if model_name not in _cross_encoder_cache:
        _cross_encoder_cache[model_name] = CrossEncoder(model_name)
    model = _cross_encoder_cache[model_name]

    # Truncate documents to 512 characters to keep inference fast on CPU
    pairs = [
        (query, (chunk.snippet or chunk.record.body or "")[:512])
        for chunk in chunks
    ]
    scores: list[float] = model.predict(pairs)  # type: ignore[assignment]
    for chunk, score in zip(chunks, scores):
        chunk.rank = float(score)

    return sorted(chunks, key=lambda c: c.rank, reverse=True)


def rerank_chunks(
    chunks: list["RetrievedChunk"],
    query: str,
    cfg: "RerankerSettings",
) -> list["RetrievedChunk"]:
    """Apply Stage-2 reranking to *chunks* according to *cfg*.

    Backends:
    - ``"bm25"``           — pure-Python, always available.
    - ``"cross-encoder"``  — sentence-transformers CrossEncoder (needs optional dep).
    - ``"auto"``           — try cross-encoder, silently fall back to bm25.
    """
    if not chunks or not cfg.enabled:
        return chunks

    backend = cfg.backend.lower()

    if backend == "cross-encoder":
        return _crossencoder_rerank(chunks, query, cfg.model)

    if backend == "auto":
        try:
            return _crossencoder_rerank(chunks, query, cfg.model)
        except ImportError:
            pass  # fall through to bm25

    # Default / fallback
    return _bm25_rerank(chunks, query)


# ---------------------------------------------------------------------------
# Semantic (dense embedding) retrieval
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Embedding cache (feature: embedding cache)
# ---------------------------------------------------------------------------
# In-memory cache keyed by (model, text).  Eliminated redundant API round-trips
# for repeated or follow-up queries within the same session.  Evicts oldest
# entry when the cache exceeds _EMBED_CACHE_MAX to cap memory use.
_embed_cache: dict[tuple[str, str], list[float]] = {}
_EMBED_CACHE_MAX = 1_000  # entries; roughly 4 MB per 1000 × 768-dim float32 vecs

# ---------------------------------------------------------------------------
# Named retrieval constants
# ---------------------------------------------------------------------------
# FTS5 — cap the number of candidates fetched (keeps query fast on large vaults).
_FTS_MAX_CANDIDATES = 500
# Multiplier applied to top_n for semantic/HyPE/sentence-window oversample.
_SEMANTIC_OVERSAMPLE = 3
# Non-semantic oversample factor: FTS5 fetches top_n * this before reranking.
_FTS_OVERSAMPLE = 20
# Body-length threshold: above this we generate a KWIC snippet; below we use full body.
_KWIC_BODY_THRESHOLD = 600
# Path-score noise threshold: results below this are silently dropped.
_NOISE_SCORE_THRESHOLD = -500
# Deduplication fingerprint length in assemble_context (chars from record body).
_DEDUP_FINGERPRINT_LEN = 500
# KWIC snippet merge gap (chars): adjacent intervals closer than this are merged.
_KWIC_MERGE_GAP = 50
# Path-bonus scale factor applied when adding route bonus to rank.
_PATH_BONUS_SCALE = 0.01


def embed_text(text: str, base_url: str, model: str, timeout: int = 60) -> list[float]:
    """Embed *text* via the configured backend and return the embedding vector.

    Uses the OpenAI-compatible ``POST /v1/embeddings`` format served by
    llama-server (with ``--embedding`` flag) or any other OpenAI-compatible
    embedding endpoint.

    Results are cached in ``_embed_cache`` keyed by ``(model, text)`` so that
    repeated queries within a session bypass the network entirely.
    """
    import json
    import urllib.request

    cache_key = (model, text)
    if cache_key in _embed_cache:
        return _embed_cache[cache_key]

    payload = json.dumps({"model": model, "input": text}).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        data = json.loads(resp.read().decode())
    vec: list[float] = data["data"][0]["embedding"]

    # Store in cache; evict oldest entry when cap is reached
    if len(_embed_cache) >= _EMBED_CACHE_MAX:
        _embed_cache.pop(next(iter(_embed_cache)))
    _embed_cache[cache_key] = vec
    return vec


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion (feature: RRF hybrid score fusion)
# ---------------------------------------------------------------------------

def _rrf_fuse(
    fts_chunks: list["RetrievedChunk"],
    sem_chunks: list["RetrievedChunk"],
    hype_chunks: "list[RetrievedChunk] | None" = None,
    sw_chunks: "list[RetrievedChunk] | None" = None,
    k: int = 60,
) -> list["RetrievedChunk"]:
    """Reciprocal Rank Fusion across up to four ranked candidate lists.

    Uses rank positions rather than raw scores, so FTS5 and cosine scales never
    need to be normalised against each other.  The RRF score for a document is::

        score = Σ  1 / (k + position_in_list_i)

    where *position* is 0-indexed and *k*=60 is the standard Cormack (2009) default.
    Documents appearing in multiple lists receive proportional contributions —
    the strongest signal of genuine relevance.

    Lists
    -----
    - *fts_chunks*  — FTS5 keyword search results (preferred for KWIC snippets)
    - *sem_chunks*  — dense semantic search results
    - *hype_chunks* — HyPE question-embedding lane (§6, optional)
    - *sw_chunks*   — Sentence Window chunk-level lane (§8, optional)
    """
    id_to_chunk: dict[str, "RetrievedChunk"] = {}
    scores: dict[str, float] = {}

    # Collect IDs that appear in any semantic lane so we can identify
    # FTS5-only records (those that miss all embedding lanes).
    sem_ids: set[str] = set()
    for chunk in sem_chunks:
        sem_ids.add(chunk.record.id)
    if hype_chunks:
        for chunk in hype_chunks:
            sem_ids.add(chunk.record.id)
    if sw_chunks:
        for chunk in sw_chunks:
            sem_ids.add(chunk.record.id)

    for pos, chunk in enumerate(fts_chunks):
        rid = chunk.record.id
        id_to_chunk.setdefault(rid, chunk)  # prefer FTS5 version (has KWIC snippet)
        score = 1.0 / (k + pos)
        # Local files without embeddings appear only in the FTS5 lane.  Add a
        # synthetic boost equivalent to a second-lane position-0 hit so they
        # are not systematically outranked by email records that have 4 lanes.
        fp = chunk.record.file_path
        if fp and rid not in sem_ids:
            lp = fp.lower().replace("\\", "/")
            if "/inbox/" in lp:
                score += 1.0 / k  # equivalent to position-0 bonus in a 2nd lane
        scores[rid] = scores.get(rid, 0.0) + score

    for pos, chunk in enumerate(sem_chunks):
        rid = chunk.record.id
        if rid not in id_to_chunk:
            id_to_chunk[rid] = chunk
        scores[rid] = scores.get(rid, 0.0) + 1.0 / (k + pos)

    if hype_chunks:
        for pos, chunk in enumerate(hype_chunks):
            rid = chunk.record.id
            if rid not in id_to_chunk:
                id_to_chunk[rid] = chunk
            scores[rid] = scores.get(rid, 0.0) + 1.0 / (k + pos)

    if sw_chunks:
        for pos, chunk in enumerate(sw_chunks):
            rid = chunk.record.id
            # Sentence-window chunks carry precise snippets — prefer them over
            # semantic/HyPE placeholders but not over FTS5 KWIC.
            if rid not in id_to_chunk:
                id_to_chunk[rid] = chunk
            elif chunk.snippet and not id_to_chunk[rid].snippet:
                id_to_chunk[rid] = chunk  # upgrade to richer snippet
            scores[rid] = scores.get(rid, 0.0) + 1.0 / (k + pos)

    result = list(id_to_chunk.values())
    for chunk in result:
        chunk.rank = scores[chunk.record.id]
    result.sort(key=lambda c: c.rank, reverse=True)
    return result


# ---------------------------------------------------------------------------
# HyDE — Hypothetical Document Embeddings (feature: HyDE)
# ---------------------------------------------------------------------------

_HYDE_PROMPT = (
    "Write a SHORT passage (2-4 sentences) that reads like real content from "
    "someone's personal files, notes, messages, or code projects — content that "
    "would DIRECTLY answer the following question.\n"
    "Rules:\n"
    "- Write as natural personal notes, not as an AI response.\n"
    "- Be specific: include realistic names, dates, file names, technologies, or places.\n"
    "- Do NOT start with 'According to' or 'In this document'.  Just write the passage.\n"
    "- Match the vocabulary and phrasing of files and notes, not questions.\n\n"
    "Question: {query}"
)


def _hyde_query(
    query: str,
    call_llm_fn: Callable,
    llm_kwargs: dict,
) -> str:
    """Generate a hypothetical vault document passage that would answer *query*.

    HyDE (Hypothetical Document Embeddings, Gao et al. 2022): embedding a
    synthetic answer rather than the raw question exploits document-to-document
    similarity rather than question-to-document similarity, materially improving
    dense retrieval recall — especially for paraphrase-heavy personal notes.

    Falls back to the original query on any LLM error so the pipeline degrades
    gracefully rather than silently dropping semantic search.
    """
    prompt = _HYDE_PROMPT.format(query=query)
    try:
        result, _ = call_llm_fn(
            messages=[{"role": "user", "content": prompt}],
            **llm_kwargs,
        )
        return result.strip() or query
    except Exception:
        return query


def _cosine_sims_blobs(
    query_vec: list[float],
    blobs: list[bytes],
) -> list[float]:
    """Return cosine similarity scores in the same order as *blobs*.

    Uses numpy when available for speed; falls back to pure Python.
    Returns an empty list when *query_vec* has zero norm or *blobs* is empty.
    Callers must pre-filter ``blobs`` so all entries share the same dimension
    as *query_vec* (``len(blob) == len(query_vec) * 4``).
    """
    import math
    import struct

    if not blobs:
        return []
    q_norm = math.sqrt(sum(x * x for x in query_vec))
    if q_norm == 0.0:
        return []
    try:
        import numpy as np

        q_arr = np.array(query_vec, dtype=np.float32) / q_norm
        dim = len(q_arr)
        matrix = np.frombuffer(b"".join(blobs), dtype=np.float32).reshape(-1, dim)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (matrix / norms @ q_arr).tolist()
    except ImportError:
        q_unit = [x / q_norm for x in query_vec]
        sims: list[float] = []
        for blob in blobs:
            n = len(blob) // 4
            vec = struct.unpack(f"{n}f", blob)
            dot = sum(a * b for a, b in zip(q_unit, vec))
            v_norm = math.sqrt(sum(x * x for x in vec))
            sims.append(dot / v_norm if v_norm else 0.0)
        return sims


def retrieve_semantic(
    store: VaultStore,
    query: str,
    top_n: int,
    embed_cfg: "EmbeddingSettings",
    resolved_base_url: str,
) -> list["RetrievedChunk"]:
    """Return up to *top_n* candidates ranked by cosine similarity to *query*.

    Reads all stored embeddings for the configured model, computes cosine
    similarity in pure Python (uses numpy when available for large vaults),
    and returns matching ``RetrievedChunk`` objects.

    Returns an empty list when:
    - No embeddings are stored yet (run ``egovault embed`` first).
    - The embedding endpoint is unreachable.
    - The query produces a zero vector.
    """
    try:
        query_vec = embed_text(query, resolved_base_url, embed_cfg.model)
    except Exception as exc:  # noqa: BLE001
        logger.warning("retrieve_semantic: embed_text failed: %s", exc)
        return []

    if not query_vec:
        return []

    stored = store.get_all_embeddings(embed_cfg.model)
    if not stored:
        return []

    dim = len(query_vec)
    compatible = [(r[0], r[1]) for r in stored if len(r[1]) == dim * 4]
    if not compatible:
        return []
    sims = _cosine_sims_blobs(query_vec, [r[1] for r in compatible])
    if not sims:
        return []
    scores: list[tuple[float, str]] = sorted(
        zip(sims, [r[0] for r in compatible]), key=lambda t: t[0], reverse=True
    )
    top_ids = [rec_id for _, rec_id in scores[: top_n * _SEMANTIC_OVERSAMPLE]]

    # Fetch full records for the top candidates
    if not top_ids:
        return []
    rows = store.fetch_records_by_ids(top_ids)
    id_to_row = {row["id"]: row for row in rows}

    chunks: list[RetrievedChunk] = []
    for sim, rec_id in scores[: top_n * _SEMANTIC_OVERSAMPLE]:
        row = id_to_row.get(rec_id)
        if row is None:
            continue
        fp = row["file_path"]
        path_bonus = _path_score(fp)
        if path_bonus <= _NOISE_SCORE_THRESHOLD:
            continue
        body = row["body"] or ""
        # No KWIC snippet here — the reranker will score by semantic similarity
        chunks.append(RetrievedChunk(record=row_to_record(row), rank=sim + path_bonus * _PATH_BONUS_SCALE))
    return chunks


# ---------------------------------------------------------------------------
# HyPE — Hypothetical Prompt Embeddings retrieval lane (feature: HyPE)
# ---------------------------------------------------------------------------


def retrieve_hype(
    store: VaultStore,
    query: str,
    top_n: int,
    embed_cfg: "EmbeddingSettings",
    resolved_base_url: str,
) -> list["RetrievedChunk"]:
    """Return up to *top_n* candidates by matching the query against stored
    question embeddings (HyPE \u00a76).

    At index time each record has 3-5 hypothetical questions embedded and stored
    in ``record_question_embeddings``.  At retrieval time the user's query is
    matched against those question vectors rather than document vectors.
    Question-to-question cosine similarity is much tighter than
    question-to-document similarity, so this lane catches records that are
    topically relevant even when their body text uses different vocabulary.

    Returns an empty list when no question embeddings are stored yet (run
    ``egovault embed`` with ``hype_enabled = true`` first).
    """
    try:
        query_vec = embed_text(query, resolved_base_url, embed_cfg.model)
    except Exception as exc:  # noqa: BLE001
        logger.warning("retrieve_hype: embed_text failed: %s", exc)
        return []

    if not query_vec:
        return []

    stored = store.get_all_question_embeddings(embed_cfg.model)
    if not stored:
        return []

    dim = len(query_vec)
    compatible = [r for r in stored if len(r[2]) == dim * 4]
    if not compatible:
        return []
    sims = _cosine_sims_blobs(query_vec, [r[2] for r in compatible])
    if not sims:
        return []

    # Aggregate: for each record_id keep the maximum similarity across all its questions
    best_score: dict[str, float] = {}
    for sim, row in zip(sims, compatible):
        rec_id = row[0]
        if sim > best_score.get(rec_id, -1.0):
            best_score[rec_id] = sim

    ranked = sorted(best_score.items(), key=lambda t: t[1], reverse=True)
    top_ids = [rec_id for rec_id, _ in ranked[: top_n * _SEMANTIC_OVERSAMPLE]]

    if not top_ids:
        return []

    rows = store.fetch_records_by_ids(top_ids)
    id_to_row = {row["id"]: row for row in rows}

    chunks: list[RetrievedChunk] = []
    for rec_id, sim in ranked[: top_n * _SEMANTIC_OVERSAMPLE]:
        row = id_to_row.get(rec_id)
        if row is None:
            continue
        if _path_score(row["file_path"]) <= _NOISE_SCORE_THRESHOLD:
            continue
        chunks.append(RetrievedChunk(record=row_to_record(row), rank=sim))
    return chunks


def plan_search_queries(
    query: str,
    history: list[dict] | None,
    call_llm_fn: Callable,
    llm_kwargs: dict,
) -> str | None:
    """Use the LLM to extract vault search keywords from the user's query.

    Returns:
      - A comma-separated keyword string (e.g. "arduino,ino,sketch") to search with.
      - None when the LLM says to skip vault search ("SKIP") or on any error.

    Falls back to stop-word keyword extraction if the LLM call fails, so the
    pipeline degrades gracefully even when the model is unreachable.
    """
    if not _QUERY_PLANNER_PROMPT:
        return _sanitize_query(query) or None

    messages: list[dict] = [{"role": "system", "content": _QUERY_PLANNER_PROMPT}]
    # Include recent conversation context (last 6 turns) so the planner can
    # resolve references like "those files" or "the last project".
    if history:
        messages.extend(history[-6:])
    messages.append({"role": "user", "content": query})

    try:
        raw, _ = call_llm_fn(messages=messages, **llm_kwargs)
        result = raw.strip().upper()
        if result == "SKIP" or not result:
            return None
        # Normalise: lowercase, strip spaces around commas, deduplicate
        raw_lower = raw.strip().lower()
        terms = [t.strip() for t in raw_lower.split(",") if t.strip()]
        # Convert to FTS5 OR expression
        return " OR ".join(dict.fromkeys(terms)) or None  # dict.fromkeys preserves order + deduplicates
    except Exception as exc:  # noqa: BLE001
        # LLM unreachable or bad response — fall back to stop-word extraction
        logger.debug("plan_search_queries: LLM failed (%s), using fallback", exc)
        return _sanitize_query(query) or None


def retrieve(
    store: VaultStore,
    query: str,
    top_n: int = 10,
    planned_query: str | None = None,
    reranker_cfg: RerankerSettings | None = None,
    embed_cfg: EmbeddingSettings | None = None,
    llm_base_url: str = "",
    call_llm_fn: Callable | None = None,
    llm_kwargs: dict | None = None,
    crag_cfg: "CRAGSettings | None" = None,
    sw_cfg: "SentenceWindowSettings | None" = None,
) -> list[RetrievedChunk]:
    """Run FTS5 search (+ optional semantic search), apply noise-filtering,
    RRF fusion, Stage-2 reranking, and optional CRAG-lite re-retrieval.

    Pipeline stages:
    1. FTS5 keyword search + supplemental LIKE search → *fts_chunks
    2. Dense semantic search (when embed_cfg.enabled) → *sem_chunks*
       - If embed_cfg.hyde_enabled and call_llm_fn is provided, the semantic
         query is replaced by a HyDE-generated hypothetical document passage.
    3. HyPE question-embedding retrieval (when hype_enabled) → *hype_chunks*
    4. Sentence window retrieval (when sw_cfg.enabled) → *sw_chunks*
    5. Reciprocal Rank Fusion of all non-empty lists → fused pool
    6. Stage-2 reranking (BM25 or cross-encoder) on the fused pool
    7. CRAG-lite: if top score < threshold, trigger corrective re-retrieval
    8. Trim to top_n
    """
    if planned_query is not None:
        sanitized = planned_query
    else:
        sanitized = _sanitize_query(query)
    if not sanitized:
        return []

    # Keywords extracted for supplemental lookup (first meaningful token)
    keywords = [k.strip() for k in sanitized.split(" OR ") if k.strip()]

    # Fetch many more candidates than top_n so the re-ranker has room to reorder.
    # We cap at _FTS_MAX_CANDIDATES to keep the query fast even on large vaults.
    fetch_n = min(_FTS_MAX_CANDIDATES, top_n * _FTS_OVERSAMPLE)

    # ── Stage 1a: FTS5 keyword match ──────────────────────────────────────────
    fts_chunks: list[RetrievedChunk] = []
    fts_seen_ids: set[str] = set()

    try:
        rows = store._con.execute(
            """
            SELECT nr.*, fts.rank
            FROM records_fts fts
            JOIN normalized_records nr ON nr.rowid = fts.rowid
            WHERE records_fts MATCH ?
            ORDER BY fts.rank
            LIMIT ?
            """,
            (sanitized, fetch_n),
        ).fetchall()

        for row in rows:
            rec_id = row["id"]
            if rec_id not in fts_seen_ids:
                fts_seen_ids.add(rec_id)
                fp = row["file_path"]
                bonus = _path_score(fp)
                if bonus <= _NOISE_SCORE_THRESHOLD:
                    continue
                body = row["body"] or ""
                snippet = _kwic_snippet(body, keywords, max_hits=3) if len(body) > _KWIC_BODY_THRESHOLD else ""
                fts_chunks.append(RetrievedChunk(record=row_to_record(row), rank=row["rank"] + bonus, snippet=snippet))
    except sqlite3.OperationalError:
        pass

    # ── Stage 1b: supplemental LIKE search for project-entry files ────────────
    # Ensures .ino / .py files always surface by path match regardless of FTS5 rank.
    if keywords:
        like_clauses = " OR ".join("file_path LIKE ?" for _ in keywords)
        like_params = [f"%{kw}%" for kw in keywords]
        sup_rows = store._con.execute(
            f"""
            SELECT *
            FROM normalized_records
            WHERE record_type = 'code'
            AND ({like_clauses})
            ORDER BY length(file_path) ASC
            LIMIT ?
            """,
            (*like_params, fetch_n),
        ).fetchall()
        for row in sup_rows:
            rec_id = row["id"]
            if rec_id not in fts_seen_ids:
                fts_seen_ids.add(rec_id)
                fp = row["file_path"]
                bonus = _path_score(fp)
                if bonus <= _NOISE_SCORE_THRESHOLD:
                    continue
                body = row["body"] or ""
                snippet = _kwic_snippet(body, keywords, max_hits=3) if len(body) > _KWIC_BODY_THRESHOLD else ""
                fts_chunks.append(RetrievedChunk(record=row_to_record(row), rank=bonus, snippet=snippet))

    # ── Stage 1b': supplemental attachment search ─────────────────────────────
    # Finds emails/messages whose *attachments* list contains image (or other
    # media) filenames even when the message body doesn't spell out the word
    # "image".  Only triggered when the query includes image-related keywords.
    _IMAGE_KEYWORDS: frozenset[str] = frozenset(
        {"image", "images", "photo", "photos", "img", "jpg", "jpeg",
         "png", "gif", "webp", "heic", "screenshot", "screenshots", "picture", "pictures"}
    )
    _PERSON_KEYWORDS: list[str] = [
        tok
        for k in keywords
        for tok in k.lower().split()
        if tok not in _IMAGE_KEYWORDS
    ]
    if any(k.lower() in _IMAGE_KEYWORDS for k in keywords) or any(
        tok in _IMAGE_KEYWORDS for k in keywords for tok in k.lower().split()
    ):
        att_clauses: list[str] = [
            "lower(attachments) LIKE '%.jpg%'",
            "lower(attachments) LIKE '%.jpeg%'",
            "lower(attachments) LIKE '%.png%'",
            "lower(attachments) LIKE '%.gif%'",
            "lower(attachments) LIKE '%.webp%'",
            "lower(attachments) LIKE '%.heic%'",
            "lower(attachments) LIKE '%image%'",
        ]
        att_where = " OR ".join(att_clauses)
        if _PERSON_KEYWORDS:
            person_clauses = " OR ".join(
                "(lower(body) LIKE ? OR lower(thread_name) LIKE ? OR lower(sender_name) LIKE ?)"
                for _ in _PERSON_KEYWORDS
            )
            person_params: list[str] = []
            for pk in _PERSON_KEYWORDS:
                like_pk = f"%{pk}%"
                person_params.extend([like_pk, like_pk, like_pk])
            att_sql = f"""
                SELECT * FROM normalized_records
                WHERE ({att_where})
                AND ({person_clauses})
                ORDER BY timestamp DESC
                LIMIT ?
            """
            att_rows = store._con.execute(att_sql, (*person_params, fetch_n)).fetchall()
        else:
            att_rows = store._con.execute(
                f"SELECT * FROM normalized_records WHERE ({att_where}) ORDER BY timestamp DESC LIMIT ?",
                (fetch_n,),
            ).fetchall()
        for row in att_rows:
            rec_id = row["id"]
            if rec_id not in fts_seen_ids:
                fts_seen_ids.add(rec_id)
                fp = row["file_path"]
                bonus = _path_score(fp)
                if bonus <= _NOISE_SCORE_THRESHOLD:
                    continue
                body = row["body"] or ""
                snippet = _kwic_snippet(body, keywords, max_hits=3) if len(body) > _KWIC_BODY_THRESHOLD else ""
                # Slightly lower rank than FTS5 hits but above fallback
                fts_chunks.append(RetrievedChunk(record=row_to_record(row), rank=bonus - 0.5, snippet=snippet))

    # Sort FTS list by path-adjusted rank so position 0 = best keyword hit
    fts_chunks.sort(key=lambda c: c.rank, reverse=True)

    # ── Stage 1c: dense semantic search ──────────────────────────────────────
    sem_chunks: list[RetrievedChunk] = []
    if embed_cfg is not None and embed_cfg.enabled:
        base = embed_cfg.base_url.strip() or llm_base_url
        # HyDE: replace the raw query with a hypothetical document passage
        # so we match document-to-document instead of question-to-document.
        semantic_query = query
        if (
            embed_cfg.hyde_enabled
            and call_llm_fn is not None
            and llm_kwargs is not None
        ):
            semantic_query = _hyde_query(query, call_llm_fn, llm_kwargs)
        raw_sem = retrieve_semantic(store, semantic_query, top_n, embed_cfg, base)
        for sc in raw_sem:
            if _path_score(sc.record.file_path) > _NOISE_SCORE_THRESHOLD:
                sem_chunks.append(sc)

    # ── Stage 1d: HyPE question-embedding retrieval lane ─────────────────────
    # Matches the user's query against per-record question embeddings generated
    # at index time.  Question-to-question similarity is tighter than
    # question-to-document, so this lane catches records that use different
    # vocabulary from the query.  No extra LLM call at query time.
    hype_chunks: list[RetrievedChunk] = []
    if (
        embed_cfg is not None
        and embed_cfg.enabled
        and embed_cfg.hype_enabled
    ):
        base = embed_cfg.base_url.strip() or llm_base_url
        raw_hype = retrieve_hype(store, query, top_n, embed_cfg, base)
        for hc in raw_hype:
            if _path_score(hc.record.file_path) > _NOISE_SCORE_THRESHOLD:
                hype_chunks.append(hc)

    # ── Stage 1e: Sentence Window retrieval lane ──────────────────────────────
    # Each record is pre-split into overlapping sentence windows at index time.
    # The query is matched against window embeddings for sub-record precision.
    # The best-matching window is expanded ±window_size/2 for context richness.
    sw_chunks: list[RetrievedChunk] = []
    if (
        sw_cfg is not None
        and sw_cfg.enabled
        and embed_cfg is not None
        and embed_cfg.enabled
    ):
        base = embed_cfg.base_url.strip() or llm_base_url
        raw_sw = retrieve_sentence_window(
            store, query, top_n, embed_cfg, base,
            window_size=sw_cfg.window_size,
        )
        for sc in raw_sw:
            if _path_score(sc.record.file_path) > _NOISE_SCORE_THRESHOLD:
                sw_chunks.append(sc)

    # ── Stage 2: Reciprocal Rank Fusion ──────────────────────────────────────
    # RRF is rank-position-based, so FTS5 and cosine scores never need to be
    # brought to a common scale.  Documents in both lists get a double boost.
    all_lists = [fts_chunks, sem_chunks, hype_chunks, sw_chunks]
    if any(all_lists):
        chunks = _rrf_fuse(
            fts_chunks,
            sem_chunks,
            hype_chunks=hype_chunks or None,
            sw_chunks=sw_chunks or None,
        )
    else:
        chunks = []

    # ── Stage 3: cross-list dedup (RRF already handles this via dict, but keep
    #    a safety set so fallback paths below stay consistent) ─────────────────
    seen_ids: set[str] = {c.record.id for c in chunks}

    # Stage-4 reranking: re-score the fused pool using BM25 or cross-encoder.
    # We rerank *all* candidates before trimming so the reranker can surface
    # lower-ranked-but-more-relevant documents into the top_n window.
    if reranker_cfg is not None:
        chunks = rerank_chunks(chunks, query, reranker_cfg)

    # ── Stage 5: CRAG-lite corrective re-retrieval ───────────────────────────
    # When all results score below threshold the retrieval is low-quality.
    # One corrective re-retrieval pass is triggered using the configured strategy.
    if (
        crag_cfg is not None
        and crag_cfg.enabled
        and chunks
    ):
        top_score = chunks[0].rank
        if top_score < crag_cfg.threshold:
            strategy = crag_cfg.strategy

            if strategy == "empty":
                # Signal to the LLM explicitly that nothing relevant was found
                return []

            elif strategy == "broaden" or (
                strategy == "hyde"
                and (call_llm_fn is None or embed_cfg is None or not embed_cfg.enabled)
            ):
                # Broaden: strip the most specific keyword and retry
                broad_terms = [k for k in keywords[1:] if k]
                broad_query = " OR ".join(broad_terms) if broad_terms else sanitized
                chunks = retrieve(
                    store=store,
                    query=query,
                    top_n=top_n,
                    planned_query=broad_query,
                    reranker_cfg=reranker_cfg,
                    embed_cfg=embed_cfg,
                    llm_base_url=llm_base_url,
                    call_llm_fn=call_llm_fn,
                    llm_kwargs=llm_kwargs,
                    crag_cfg=None,  # no recursive CRAG on the retry
                )

            elif strategy == "hyde" and call_llm_fn is not None and embed_cfg is not None:
                # Force HyDE-augmented semantic re-retrieval
                hyde_embed_cfg = EmbeddingSettings(
                    enabled=embed_cfg.enabled,
                    model=embed_cfg.model,
                    base_url=embed_cfg.base_url,
                    hyde_enabled=True,
                    contextual_enabled=embed_cfg.contextual_enabled,
                )
                chunks = retrieve(
                    store=store,
                    query=query,
                    top_n=top_n,
                    planned_query=planned_query,
                    reranker_cfg=reranker_cfg,
                    embed_cfg=hyde_embed_cfg,
                    llm_base_url=llm_base_url,
                    call_llm_fn=call_llm_fn,
                    llm_kwargs=llm_kwargs,
                    crag_cfg=None,  # no recursive CRAG on the retry
                )

    chunks = chunks[:top_n]

    # Fallback: LIKE search if nothing survived noise filtering.
    # Run in two passes: prefer non-noise paths, include noise paths only
    # if they are the sole evidence (e.g. explicit keyword like "Malta").
    if not chunks:
        first_kw = keywords[0] if keywords else sanitized
        like_term = f"%{first_kw}%"
        rows = store._con.execute(
            """
            SELECT * FROM normalized_records
            WHERE body LIKE ? OR thread_name LIKE ? OR sender_name LIKE ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (like_term, like_term, like_term, top_n * 4),
        ).fetchall()
        clean_rows = [r for r in rows if _path_score(r["file_path"]) > _NOISE_SCORE_THRESHOLD]
        use_rows = clean_rows if clean_rows else rows  # fall back to noisy results if nothing else
        for row in use_rows[:top_n]:
            rec_id = row["id"]
            if rec_id not in seen_ids:
                seen_ids.add(rec_id)
                body = row["body"] or ""
                snippet = _kwic_snippet(body, keywords, max_hits=3) if len(body) > _KWIC_BODY_THRESHOLD else ""
                chunks.append(RetrievedChunk(record=row_to_record(row), rank=0.0, snippet=snippet))

    # Apply Stage-2 reranking to fallback results as well, then trim.
    if chunks and reranker_cfg is not None:
        chunks = rerank_chunks(chunks, query, reranker_cfg)
    chunks = chunks[:top_n]

    return chunks


# ---------------------------------------------------------------------------
# Sentence Window Retrieval (P4) — sub-record chunk-level semantic search
# ---------------------------------------------------------------------------


def retrieve_sentence_window(
    store: VaultStore,
    query: str,
    top_n: int,
    embed_cfg: "EmbeddingSettings",
    resolved_base_url: str,
    window_size: int = 3,
) -> list["RetrievedChunk"]:
    """Return up to *top_n* candidates by matching the query against sentence-window
    chunk embeddings (Sentence Window Retrieval, §8).

    At index time each record body is split into overlapping windows of
    ``window_size`` sentences and each window is embedded separately.  At
    retrieval time the query is matched against chunk embeddings.  The winning
    chunk's *parent record* is returned with a snippet that shows the matching
    window expanded by up to ``window_size`` surrounding chunks.

    Returns an empty list when no chunk embeddings are stored yet (run
    ``egovault embed`` with ``sentence_window.enabled = true`` first).
    """
    try:
        query_vec = embed_text(query, resolved_base_url, embed_cfg.model)
    except Exception as exc:  # noqa: BLE001
        logger.warning("retrieve_sentence_window: embed_text failed: %s", exc)
        return []

    if not query_vec:
        return []

    stored = store.get_all_chunk_embeddings(embed_cfg.model)
    if not stored:
        return []

    # Score every chunk against the query
    dim = len(query_vec)
    compatible = [r for r in stored if len(r[3]) == dim * 4]
    if not compatible:
        return []
    sims = _cosine_sims_blobs(query_vec, [r[3] for r in compatible])
    if not sims:
        return []

    # For each record keep only the best-scoring chunk
    best: dict[str, tuple[float, int, str]] = {}  # record_id → (score, chunk_idx, chunk_text)
    for sim, row in zip(sims, compatible):
        rec_id, cidx, ctext = row[0], row[1], row[2]
        if sim > best.get(rec_id, (-1.0, 0, ""))[0]:
            best[rec_id] = (sim, cidx, ctext)

    ranked = sorted(best.items(), key=lambda t: t[1][0], reverse=True)
    top_ids = [rec_id for rec_id, _ in ranked[: top_n * 3]]

    if not top_ids:
        return []

    # Fetch parent records
    rows = store.fetch_records_by_ids(top_ids)
    id_to_row = {row["id"]: row for row in rows}

    # Build context window snippet: best chunk ± (window_size-1) surrounding chunks
    chunks_out: list[RetrievedChunk] = []
    for rec_id, (sim, best_cidx, _best_text) in ranked[: top_n * 3]:
        row = id_to_row.get(rec_id)
        if row is None:
            continue
        if _path_score(row["file_path"]) <= _NOISE_SCORE_THRESHOLD:
            continue

        # Fetch neighbouring chunks to build the context window
        all_chunks = store.get_chunks_for_record(rec_id, embed_cfg.model)
        if not all_chunks:
            # Fall back to full body
            body = row["body"] or ""
            chunks_out.append(RetrievedChunk(record=row_to_record(row), rank=sim, snippet=body[:600]))
            continue

        half = max(1, window_size // 2)
        lo = max(0, best_cidx - half)
        hi = min(len(all_chunks) - 1, best_cidx + half)
        window_texts = [t for (_, t, _) in all_chunks[lo : hi + 1]]
        snippet = " ".join(window_texts)
        if lo > 0:
            snippet = "[\u2026] " + snippet
        if hi < len(all_chunks) - 1:
            snippet = snippet + " [\u2026]"

        chunks_out.append(RetrievedChunk(record=row_to_record(row), rank=sim, snippet=snippet))

    return chunks_out


def assemble_context(chunks: list[RetrievedChunk], max_total_chars: int = 12_000) -> str:
    """Convert retrieved chunks into a context block for the LLM prompt.

    Caps total output at *max_total_chars* and deduplicates entries with
    near-identical snippets so repeated log files don't flood the context.
    """
    if not chunks:
        return ""

    _CODE_BODY_LIMIT = 400

    lines: list[str] = ["[VAULT CONTEXT]"]
    used_chars = 0
    seen_snippets: set[str] = set()  # dedup by first-300-char fingerprint
    entry_num = 0
    for chunk in chunks:
        rec = chunk.record
        ts = rec.timestamp.strftime("%Y-%m-%d")
        source = f"{rec.platform} / {rec.thread_name or rec.file_path or 'unknown'}"
        sender = f"{rec.sender_name}: " if rec.sender_name and rec.sender_name != "user" else ""

        if rec.record_type == "code":
            # Prefer a KWIC snippet (keyword-in-context excerpt) over naive
            # head-of-file truncation — this surfaces relevant functions/blocks
            # even when they appear deep in a large file.
            if chunk.snippet:
                body = chunk.snippet
            else:
                body = rec.body
                if len(body) > _CODE_BODY_LIMIT:
                    body = body[:_CODE_BODY_LIMIT].rstrip() + " …"
        elif rec.file_path and len(rec.body or "") <= 12000:
            # Local files are structured documents (PDFs, tickets, notes) where
            # ALL sections are relevant — not just the lines near a keyword hit.
            # Using a KWIC snippet would cut off e.g. flight departure dates that
            # appear far from the passenger-name field the keyword matched.
            # Show the full body when it fits within a generous per-record budget.
            body = rec.body
        elif chunk.snippet:
            body = chunk.snippet
        else:
            body = rec.body

        # Skip entries whose snippet is identical to an already-emitted one
        dedup_key = body[:_DEDUP_FINGERPRINT_LEN]
        if dedup_key in seen_snippets:
            continue
        seen_snippets.add(dedup_key)

        att_names = [a for a in (rec.attachments or []) if a]
        att_suffix = (
            f"\n  [Attachments: {', '.join(att_names[:15])}]" if att_names else ""
        )

        entry_num += 1
        entry = f"[{entry_num}] ({source}, {ts}) {sender}{body}{att_suffix}"
        if used_chars + len(entry) > max_total_chars:
            lines.append("[… additional results omitted — refine keywords to focus results]")
            break
        lines.append(entry)
        used_chars += len(entry)

    return "\n".join(lines)


def build_prompt(
    query: str,
    context: str,
    history: list[dict] | None = None,
    owner_profile: str = "",
    output_dir: str = "",
    today: str = "",
) -> list[dict]:
    """Build the messages list for an LLM /chat call, optionally injecting vault context and history."""
    system = _LEVEL_0_SYSTEM_PROMPT
    if owner_profile:
        system = f"{system}\n\n[OWNER PROFILE]\n{owner_profile}"
    if output_dir:
        # Resolve common folder paths so the LLM can write to user-named locations.
        from egovault.utils.folders import resolve_folder as _rf
        _known_path_lines: list[str] = []
        for _alias in ("desktop", "downloads", "documents"):
            try:
                _known_path_lines.append(f"  {_alias}: {_rf(_alias)}")
            except Exception:
                pass
        _paths_hint = (
            "\nKnown folder paths (use these when the user specifies a location): "
            + " | ".join(_known_path_lines)
            if _known_path_lines else ""
        )
        system = (
            f"{system}\n\n[OUTPUT DIRECTORY]\n{output_dir}\n"
            "When the user asks to save, export, or create a file, use write_file with a path inside this directory unless the user specifies otherwise."
            f"{_paths_hint}"
        )
    if today:
        # Pre-compute week and month boundaries so the LLM never has to calculate dates.
        from datetime import date as _d, timedelta as _td
        _t = _d.fromisoformat(today)
        _dow = _t.weekday()  # 0=Mon … 6=Sun
        _this_week_start = _t - _td(days=_dow)
        _this_week_end = _this_week_start + _td(days=6)
        _last_week_start = _this_week_start - _td(days=7)
        _last_week_end = _this_week_start - _td(days=1)
        _yesterday = _t - _td(days=1)
        _month_start = _t.replace(day=1)
        _last_month_end = _month_start - _td(days=1)
        _last_month_start = _last_month_end.replace(day=1)
        _date_ctx = (
            f"{today} ({_t.strftime('%A')})\n"
            f"Yesterday      : {_yesterday.isoformat()}\n"
            f"This week (Mon–Sun): {_this_week_start.isoformat()} to {_this_week_end.isoformat()}\n"
            f"Last week (Mon–Sun): {_last_week_start.isoformat()} to {_last_week_end.isoformat()}\n"
            f"This month     : {_month_start.isoformat()} to {today}\n"
            f"Last month     : {_last_month_start.isoformat()} to {_last_month_end.isoformat()}"
        )
        system = (
            f"{system}\n\n[TODAY]\n{_date_ctx}\n"
            "Use these pre-computed ranges directly for since/until — do not recalculate."
        )
    messages: list[dict] = [{"role": "system", "content": system}]
    if history:
        messages.extend(history)
    user_content = f"{context}\n\n---\n{query}" if context else query
    messages.append({"role": "user", "content": user_content})
    return messages


_PROFILE_EXTRACTOR_PROMPT = """\
You are a personal fact extractor. Given snippets from a user's personal files, extract key facts about the vault OWNER only.

Output ONLY this format (omit lines you cannot determine):
name: <full name>
username: <login/handle>
email: <email>
location: <city/country>
occupation: <role>

Rules:
- Only include facts you are confident about from the provided text.
- Do NOT invent or guess. If you cannot determine a fact, skip that line.
- Output nothing else — no explanation, no preamble.
"""


def extract_owner_profile(
    store: VaultStore,
    call_llm_fn: Callable,
    llm_kwargs: dict,
) -> str:
    """Use the LLM to extract personal facts from vault records and update the stored profile.

    Searches for promising records (git config, about/profile files, author mentions),
    passes snippets to the LLM, and caches the result in the settings table.
    Returns the extracted profile string.
    """
    import os
    from pathlib import Path as _Path

    snippets: list[str] = []

    # 1. System-level identity hints (always available)
    windows_user = os.environ.get("USERNAME") or os.environ.get("USER") or ""
    computer = os.environ.get("COMPUTERNAME") or ""
    if windows_user:
        snippets.append(f"Windows username: {windows_user}")
    if computer:
        snippets.append(f"Computer name: {computer}")

    # 2. Directly read well-known identity files from disk (no scan required)
    _IDENTITY_FILES = [
        _Path.home() / ".gitconfig",
        _Path.home() / ".gitconfig_global",
        _Path.home() / ".npmrc",
        _Path.home() / ".config" / "git" / "config",
    ]
    for idfile in _IDENTITY_FILES:
        if idfile.exists():
            try:
                content = idfile.read_text(encoding="utf-8", errors="replace")[:600]
                snippets.append(f"--- {idfile} ---\n{content}")
            except OSError:
                pass

    # 3. Vault records that likely contain owner info
    rows = store._con.execute(
        """
        SELECT file_path, body FROM normalized_records
        WHERE (
            lower(file_path) LIKE '%.gitconfig'
            OR lower(file_path) LIKE '%gitconfig%'
            OR lower(body) LIKE '%user.name%'
            OR lower(body) LIKE '%user.email%'
            OR lower(body) LIKE '%registered owner%'
            OR lower(body) LIKE '%full name%'
            OR lower(body) LIKE '%author:%'
        )
        AND lower(file_path) NOT LIKE '%node_modules%'
        AND lower(file_path) NOT LIKE '%build%'
        AND lower(file_path) NOT LIKE '%appdata%'
        ORDER BY length(file_path) ASC
        LIMIT 8
        """
    ).fetchall()

    for row in rows:
        body = row["body"] or ""
        # Trim to first 400 chars to keep the prompt short
        snippets.append(f"--- {row['file_path']} ---\n{body[:400]}")

    if not snippets:
        return ""

    user_content = "\n\n".join(snippets)
    messages = [
        {"role": "system", "content": _PROFILE_EXTRACTOR_PROMPT},
        {"role": "user", "content": user_content},
    ]

    try:
        profile, _ = call_llm_fn(messages=messages, **llm_kwargs)
        profile = profile.strip()
        if profile:
            store.set_owner_profile(profile)
        return profile
    except Exception as exc:  # noqa: BLE001
        logger.warning("extract_owner_profile: LLM call failed: %s", exc)
        return store.get_owner_profile()



def vault_summary_context(store: VaultStore) -> str:
    """Return a [VAULT SUMMARY] block describing vault contents when no search results matched.

    Returns an empty string when the vault is empty.
    """
    stats = store.vault_stats()
    if not stats["total_records"]:
        return ""

    lines = ["[VAULT SUMMARY — no search results matched, but the vault contains:]"]
    lines.append(f"Total records: {stats['total_records']}")
    if stats["date_min"] and stats["date_max"]:
        lines.append(f"Date range: {stats['date_min']} to {stats['date_max']}")
    if stats["sources"]:
        lines.append("Sources / files:")
        for label in stats["sources"]:
            lines.append(f"  - {label}")
    lines.append(
        "Use this information to acknowledge what is in the vault and "
        "guide the user towards a more specific query or /scan if they need to add new files."
    )
    return "\n".join(lines)


def source_attribution(chunks: list[RetrievedChunk]) -> list[str]:
    """Return human-readable source labels for the retrieved chunks."""
    labels: list[str] = []
    for chunk in chunks:
        rec = chunk.record
        ts = rec.timestamp.strftime("%Y-%m-%d")
        name = rec.thread_name or (rec.file_path or "unknown")
        labels.append(f"[{rec.platform}] {name} ({ts})")
    return labels
