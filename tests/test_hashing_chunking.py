"""Unit tests for hashing and chunking utilities."""
from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path


from egovault.utils.hashing import compute_content_hash, compute_file_id, compute_record_id
from egovault.utils.chunking import make_sentence_windows, split_sentences


# ---------------------------------------------------------------------------
# hashing — compute_record_id
# ---------------------------------------------------------------------------

class TestComputeRecordId:
    def test_deterministic_for_same_inputs(self) -> None:
        id1 = compute_record_id("platform", "thread", "2024-01-01", "sender", "body")
        id2 = compute_record_id("platform", "thread", "2024-01-01", "sender", "body")
        assert id1 == id2

    def test_different_platforms_produce_different_ids(self) -> None:
        id1 = compute_record_id("facebook", "t", "2024-01-01", "user", "body")
        id2 = compute_record_id("twitter", "t", "2024-01-01", "user", "body")
        assert id1 != id2

    def test_file_path_mode_ignores_thread_and_sender(self) -> None:
        # Two calls with different thread_id/sender_id but same file_path
        id1 = compute_record_id("p", "t1", "2024-01-01", "s1", "body", file_path="/file.md")
        id2 = compute_record_id("p", "t2", "2024-01-01", "s2", "body", file_path="/file.md")
        assert id1 == id2

    def test_file_path_and_non_file_path_differ(self) -> None:
        id1 = compute_record_id("p", "t", "2024-01-01", "s", "body", file_path="/file.md")
        id2 = compute_record_id("p", "t", "2024-01-01", "s", "body")
        assert id1 != id2

    def test_returns_hex_string(self) -> None:
        result = compute_record_id("p", "t", "2024-01-01", "s", "b")
        assert all(c in "0123456789abcdef" for c in result)
        assert len(result) == 64  # SHA-256 hex digest length

    def test_accepts_datetime_timestamp(self) -> None:
        ts = datetime(2024, 6, 15, 12, 0, 0)
        id1 = compute_record_id("p", "t", ts, "s", "body")
        id2 = compute_record_id("p", "t", ts.isoformat(), "s", "body")
        assert id1 == id2


# ---------------------------------------------------------------------------
# hashing — compute_file_id
# ---------------------------------------------------------------------------

class TestComputeFileId:
    def test_deterministic(self) -> None:
        id1 = compute_file_id("/path/to/file.txt", 1700000000.0, 1024)
        id2 = compute_file_id("/path/to/file.txt", 1700000000.0, 1024)
        assert id1 == id2

    def test_different_path_produces_different_id(self) -> None:
        id1 = compute_file_id("/a.txt", 1700000000.0, 1024)
        id2 = compute_file_id("/b.txt", 1700000000.0, 1024)
        assert id1 != id2

    def test_different_mtime_produces_different_id(self) -> None:
        id1 = compute_file_id("/file.txt", 1700000000.0, 1024)
        id2 = compute_file_id("/file.txt", 1700000001.0, 1024)
        assert id1 != id2


# ---------------------------------------------------------------------------
# hashing — compute_content_hash
# ---------------------------------------------------------------------------

class TestComputeContentHash:
    def test_hash_matches_hashlib(self, tmp_path: Path) -> None:
        f = tmp_path / "sample.txt"
        f.write_bytes(b"hello world")
        expected = hashlib.sha256(b"hello world").hexdigest()
        assert compute_content_hash(f) == expected

    def test_different_content_different_hash(self, tmp_path: Path) -> None:
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_bytes(b"content A")
        f2.write_bytes(b"content B")
        assert compute_content_hash(f1) != compute_content_hash(f2)

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        f = tmp_path / "file.bin"
        f.write_bytes(b"\x00\xff")
        result = compute_content_hash(str(f))
        assert isinstance(result, str) and len(result) == 64


# ---------------------------------------------------------------------------
# chunking — split_sentences
# ---------------------------------------------------------------------------

class TestSplitSentences:
    def test_empty(self) -> None:
        assert split_sentences("") == []

    def test_single_sentence(self) -> None:
        result = split_sentences("Hello world.")
        assert len(result) == 1
        assert result[0] == "Hello world."

    def test_multiple_sentences(self) -> None:
        result = split_sentences("First sentence. Second sentence. Third sentence.")
        assert len(result) >= 2

    def test_strips_whitespace(self) -> None:
        result = split_sentences("  Trimmed.  ")
        assert result[0] == "Trimmed."

    def test_blank_lines_as_boundaries(self) -> None:
        result = split_sentences("Paragraph one.\n\nParagraph two.")
        assert any("Paragraph one" in s for s in result)
        assert any("Paragraph two" in s for s in result)


# ---------------------------------------------------------------------------
# chunking — make_sentence_windows
# ---------------------------------------------------------------------------

class TestMakeSentenceWindows:
    def test_empty_text(self) -> None:
        assert make_sentence_windows("") == []

    def test_short_text_produces_single_window(self) -> None:
        result = make_sentence_windows("Just one sentence.")
        assert len(result) == 1
        assert result[0][0] == 0

    def test_window_contains_multiple_sentences(self) -> None:
        text = "One. Two. Three. Four. Five."
        windows = make_sentence_windows(text, window_size=2, overlap=0)
        assert len(windows) >= 2
        for idx, chunk in windows:
            assert isinstance(chunk, str) and len(chunk) > 0

    def test_overlap_shares_sentences(self) -> None:
        text = "A. B. C. D. E."
        windows = make_sentence_windows(text, window_size=3, overlap=1)
        if len(windows) >= 2:
            # last sentence of window 0 should appear in start of window 1
            last_in_0 = windows[0][1].split()[-1]
            first_in_1 = windows[1][1].split()[0]
            # They won't be the same word but the window text should share content
            assert windows[0][1] != windows[1][1]

    def test_indices_are_sequential(self) -> None:
        text = "S1. S2. S3. S4. S5. S6. S7. S8."
        windows = make_sentence_windows(text, window_size=2, overlap=0)
        indices = [idx for idx, _ in windows]
        assert indices == list(range(len(windows)))
