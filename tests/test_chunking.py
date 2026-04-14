"""Unit tests for the sentence-window chunking utility (P4)."""
from __future__ import annotations


from egovault.utils.chunking import make_sentence_windows, split_sentences


# ---------------------------------------------------------------------------
# split_sentences
# ---------------------------------------------------------------------------


def test_split_empty_string() -> None:
    assert split_sentences("") == []


def test_split_single_sentence() -> None:
    result = split_sentences("Hello world.")
    assert len(result) >= 1
    assert any("Hello" in s for s in result)


def test_split_multiple_sentences() -> None:
    text = "First sentence. Second sentence! Third sentence?"
    result = split_sentences(text)
    assert len(result) >= 2


def test_split_multiline_text() -> None:
    text = "Line one.\nLine two.\nLine three."
    result = split_sentences(text)
    assert len(result) >= 3


def test_split_blank_lines_as_boundaries() -> None:
    text = "Para one sentence one. Para one sentence two.\n\nPara two sentence one."
    result = split_sentences(text)
    assert len(result) >= 2


def test_split_strips_whitespace() -> None:
    result = split_sentences("  Leading spaces.  ")
    assert all(s == s.strip() for s in result)
    assert all(s for s in result)  # no empty strings


# ---------------------------------------------------------------------------
# make_sentence_windows
# ---------------------------------------------------------------------------


def test_windows_empty_text() -> None:
    assert make_sentence_windows("") == []


def test_windows_short_text_single_window() -> None:
    text = "Just one sentence."
    windows = make_sentence_windows(text, window_size=3, overlap=1)
    assert len(windows) >= 1
    assert windows[0][0] == 0  # chunk_index starts at 0


def test_windows_returns_index_and_text() -> None:
    text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
    windows = make_sentence_windows(text, window_size=3, overlap=1)
    for idx, text_w in windows:
        assert isinstance(idx, int)
        assert isinstance(text_w, str)
        assert text_w  # non-empty


def test_windows_index_starts_at_zero() -> None:
    text = "A. B. C. D. E."
    windows = make_sentence_windows(text, window_size=2, overlap=0)
    assert windows[0][0] == 0


def test_windows_overlap_produces_more_chunks() -> None:
    text = "S1. S2. S3. S4. S5. S6."
    no_overlap = make_sentence_windows(text, window_size=2, overlap=0)
    with_overlap = make_sentence_windows(text, window_size=2, overlap=1)
    assert len(with_overlap) >= len(no_overlap)


def test_windows_cover_all_content() -> None:
    """All sentences from the original text appear in at least one window."""
    text = "Alpha beta. Gamma delta. Epsilon zeta. Eta theta."
    windows = make_sentence_windows(text, window_size=2, overlap=1)
    all_text = " ".join(w for _, w in windows).lower()
    for word in ["alpha", "gamma", "epsilon", "eta"]:
        assert word in all_text


def test_windows_window_size_1() -> None:
    text = "One. Two. Three."
    windows = make_sentence_windows(text, window_size=1, overlap=0)
    assert len(windows) >= 3
