"""Sentence-window chunking utilities for Sentence Window Retrieval (P4).

Splits a text body into overlapping sentence windows so that each chunk can
be embedded independently.  At retrieval time the best-matching chunk's
surrounding window is returned, giving precise semantic matching with
full surrounding context.

Design constraints
------------------
- Pure Python — no NLTK or spaCy dependency.
- Handles plain text, Markdown, code, and HTML adequately.
- Overlapping windows: each window shares *overlap* sentences with its
  neighbour so that relevant context is never split across a boundary.
"""
from __future__ import annotations

import re

# Sentence boundary: end-of-sentence punctuation followed by whitespace and
# an uppercase letter OR end-of-string.  Handles "Mr.", "Dr.", URLs etc. by
# requiring the follow-up character to be uppercase (heuristic; not perfect
# but very fast and dependency-free).
_SENTENCE_END = re.compile(
    r"""
    (?<!\w\.\w)          # not in the middle of an abbreviation like "U.S."
    (?<![A-Z][a-z]\.)    # not after honorific like "Mr."
    (?<=[.!?…])          # must come right after sentence-ending punctuation
    (?=\s+[A-Z]|\s*$)    # must be followed by whitespace+capitalised or EOL
    """,
    re.VERBOSE,
)


def split_sentences(text: str) -> list[str]:
    """Split *text* into a list of individual sentences.

    Uses a lightweight regex heuristic: sentence boundaries are detected by
    end-of-sentence punctuation followed by whitespace and an uppercase letter.
    Newlines are treated as soft sentence breaks too.

    Returns a list of non-empty stripped sentence strings.
    """
    if not text:
        return []

    # First normalise newlines: treat blank lines as sentence boundaries.
    text = re.sub(r"\n{2,}", ".\n", text)

    # Split on detected sentence boundaries first
    parts: list[str] = _SENTENCE_END.split(text)

    # Further split on remaining newlines within each part
    sentences: list[str] = []
    for part in parts:
        for line in part.splitlines():
            stripped = line.strip()
            if stripped:
                sentences.append(stripped)

    return sentences


def make_sentence_windows(
    text: str,
    window_size: int = 3,
    overlap: int = 1,
) -> list[tuple[int, str]]:
    """Split *text* into overlapping sentence windows.

    Parameters
    ----------
    text:
        The full record body to chunk.
    window_size:
        Number of sentences per window.
    overlap:
        Number of sentences shared between consecutive windows.

    Returns
    -------
    A list of ``(chunk_index, chunk_text)`` tuples, where *chunk_text* is
    the concatenation of ``window_size`` consecutive sentences separated by
    a space.  *chunk_index* starts at 0.

    For very short texts (fewer sentences than *window_size*), a single
    window containing the whole text is returned.
    """
    sentences = split_sentences(text)
    if not sentences:
        return []

    step = max(1, window_size - overlap)
    windows: list[tuple[int, str]] = []
    idx = 0
    pos = 0
    while pos < len(sentences):
        window_sents = sentences[pos : pos + window_size]
        windows.append((idx, " ".join(window_sents)))
        idx += 1
        pos += step

    return windows
