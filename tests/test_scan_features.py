"""Tests for folder resolution, intent mapping, and the /scan chat command."""
from __future__ import annotations

import platform
import shutil
from pathlib import Path
from io import StringIO
from unittest.mock import patch

import pytest

from egovault.core.store import VaultStore
from egovault.config import Settings

FIXTURES = Path(__file__).parent / "fixtures" / "local_inbox"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def store() -> VaultStore:
    s = VaultStore(":memory:")
    s.init_db()
    yield s
    s.close()


@pytest.fixture()
def settings() -> Settings:
    return Settings()


# ===========================================================================
# resolve_folder
# ===========================================================================


class TestResolveFolder:
    def test_home_alias_returns_home_dir(self) -> None:
        from egovault.utils.folders import resolve_folder
        assert resolve_folder("home") == Path.home()

    def test_home_alias_case_insensitive(self) -> None:
        from egovault.utils.folders import resolve_folder
        assert resolve_folder("HOME") == Path.home()

    def test_tilde_path_expanded(self, tmp_path: Path) -> None:
        from egovault.utils.folders import resolve_folder
        # ~/  should expand to the home directory
        result = resolve_folder("~/")
        assert result == Path.home()

    def test_absolute_path_returned_as_is(self, tmp_path: Path) -> None:
        from egovault.utils.folders import resolve_folder
        result = resolve_folder(str(tmp_path))
        assert result == tmp_path.resolve()

    def test_plain_nonexistent_path_returned_without_error(self, tmp_path: Path) -> None:
        from egovault.utils.folders import resolve_folder
        nonexistent = tmp_path / "does_not_exist"
        # plain paths are returned even if they don't exist — caller decides
        result = resolve_folder(str(nonexistent))
        assert result == nonexistent.resolve()

    def test_unknown_alias_raises_value_error(self, tmp_path: Path) -> None:
        from egovault.utils.folders import resolve_folder
        import unittest.mock as mock
        # Patch all platform resolvers and Path.home to force every known alias
        # to return None, then confirm ValueError is raised.
        alias = "desktop"
        with mock.patch("egovault.utils.folders._resolve_windows", return_value=None), \
             mock.patch("egovault.utils.folders._resolve_macos", return_value=None), \
             mock.patch("egovault.utils.folders._resolve_linux", return_value=None), \
             mock.patch("egovault.utils.folders._default_folder_name", return_value=tmp_path / "nonexistent"):
            with pytest.raises(ValueError, match="not locate well-known folder"):
                resolve_folder(alias)

    def test_documents_resolves_to_existing_dir(self) -> None:
        from egovault.utils.folders import resolve_folder
        try:
            path = resolve_folder("documents")
            assert path.is_dir()
        except ValueError:
            pytest.skip("No Documents folder found on this system")

    def test_downloads_resolves_to_existing_dir(self) -> None:
        from egovault.utils.folders import resolve_folder
        try:
            path = resolve_folder("downloads")
            assert path.is_dir()
        except ValueError:
            pytest.skip("No Downloads folder found on this system")

    @pytest.mark.skipif(platform.system() != "Windows", reason="Windows-only")
    def test_windows_registry_returns_path(self) -> None:
        from egovault.utils.folders import _resolve_windows
        path = _resolve_windows("documents")
        assert path is None or path.is_dir()

    @pytest.mark.skipif(platform.system() == "Windows", reason="Linux/macOS only")
    def test_linux_xdg_env_var_used_when_set(self, tmp_path: Path) -> None:
        from egovault.utils.folders import _resolve_linux
        with patch.dict("os.environ", {"XDG_DOCUMENTS_DIR": str(tmp_path)}):
            result = _resolve_linux("documents")
        assert result == tmp_path

    @pytest.mark.skipif(platform.system() == "Windows", reason="Linux/macOS only")
    def test_linux_falls_back_to_capitalised_home_subdir(self, tmp_path: Path) -> None:
        from egovault.utils.folders import _resolve_linux
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        docs = fake_home / "Documents"
        docs.mkdir()
        with patch("egovault.utils.folders.Path.home", return_value=fake_home), \
             patch.dict("os.environ", {}, clear=False):
            result = _resolve_linux("documents")
        assert result == docs


# ===========================================================================
# list_known_folders
# ===========================================================================


class TestListKnownFolders:
    def test_returns_all_aliases(self) -> None:
        from egovault.utils.folders import list_known_folders, KNOWN_FOLDERS
        results = list_known_folders()
        returned_aliases = {alias for alias, _ in results}
        assert returned_aliases == set(KNOWN_FOLDERS)

    def test_each_entry_is_tuple_of_alias_and_path_or_none(self) -> None:
        from egovault.utils.folders import list_known_folders
        for alias, path in list_known_folders():
            assert isinstance(alias, str)
            assert path is None or isinstance(path, Path)

    def test_home_always_resolves(self) -> None:
        from egovault.utils.folders import list_known_folders
        home_entry = next(p for a, p in list_known_folders() if a == "home")
        assert home_entry is not None
        assert home_entry == Path.home()


# ===========================================================================
# _resolve_intent
# ===========================================================================


class TestResolveIntent:
    """All intent-map patterns plus the folder-capturing scan pattern."""

    @pytest.mark.parametrize("text,expected", [
        # status
        ("what's the gpu usage?",          "/status"),
        ("show model status",               "/status"),
        ("how much vram is used",           "/status"),
        # sources
        ("show sources",                    "/sources"),
        ("where did that come from?",       "/sources"),
        ("list sources",                    "/sources"),
        # help
        ("help",                            "/help"),
        ("what can you do?",                "/help"),
        ("show commands",                   "/help"),
        # clear
        ("clear the screen",                "/clear"),
        ("clean screen",                    "/clear"),
        # restart
        ("start over",                      "/restart"),
        ("reset chat",                      "/restart"),
        ("new conversation",                "/restart"),
        # exit
        ("please quit",                     "/exit"),
        ("bye",                             "/exit"),
        ("end session",                     "/exit"),
        ("goodbye",                         "/exit"),
        # scan — folder captured
        ("scan my desktop",                 "/scan desktop"),
        ("scan desktop",                    "/scan desktop"),
        ("index my documents",              "/scan documents"),
        ("please ingest my downloads",      "/scan downloads"),
        ("import the pictures folder",      "/scan pictures"),
        ("scan home",                       "/scan home"),
        ("index my videos",                 "/scan videos"),
        ("scan the movies folder",          "/scan movies"),
        # no match
        ("what is the weather?",            None),
        ("tell me about Python",            None),
    ])
    def test_intent(self, text: str, expected: str | None) -> None:
        from egovault.chat.session import _resolve_intent
        assert _resolve_intent(text) == expected

    def test_intent_case_insensitive(self) -> None:
        from egovault.chat.session import _resolve_intent
        assert _resolve_intent("SCAN MY DESKTOP") == "/scan desktop"
        assert _resolve_intent("PLEASE QUIT") == "/exit"


# ===========================================================================
# _handle_scan
# ===========================================================================


class TestHandleScan:
    """Tests for the /scan chat command handler."""

    def _run(self, command: str, store: VaultStore, settings: Settings) -> str:
        """Run _handle_scan and capture console output."""
        from egovault.chat.session import _handle_scan
        from rich.console import Console
        buf = StringIO()
        con = Console(file=buf, highlight=False, markup=False)
        with patch("egovault.chat.session.console", con):
            _handle_scan(command, store, settings)
        return buf.getvalue()

    def test_scan_no_arg_prints_usage(self, store: VaultStore, settings: Settings) -> None:
        out = self._run("/scan", store, settings)
        assert "Usage" in out or "alias" in out.lower()

    def test_scan_list_flag_shows_aliases(self, store: VaultStore, settings: Settings) -> None:
        out = self._run("/scan --list", store, settings)
        assert "home" in out.lower()
        assert "documents" in out.lower()

    def test_scan_bad_path_prints_error(self, store: VaultStore, settings: Settings) -> None:
        out = self._run("/scan /tmp/__egovault_does_not_exist_xyz__", store, settings)
        assert "error" in out.lower() or "not" in out.lower()

    def test_scan_file_not_dir_prints_error(
        self, store: VaultStore, settings: Settings, tmp_path: Path
    ) -> None:
        f = tmp_path / "file.txt"
        f.write_text("hello")
        out = self._run(f"/scan {f}", store, settings)
        assert "error" in out.lower()

    def test_scan_empty_dir_reports_no_files(
        self, store: VaultStore, settings: Settings, tmp_path: Path
    ) -> None:
        out = self._run(f"/scan {tmp_path}", store, settings)
        assert "no supported" in out.lower() or "no" in out.lower()

    def test_scan_dir_with_only_unsupported_files(
        self, store: VaultStore, settings: Settings, tmp_path: Path
    ) -> None:
        (tmp_path / "archive.xyz").write_bytes(b"garbage")
        out = self._run(f"/scan {tmp_path}", store, settings)
        assert "no supported" in out.lower()

    def test_scan_fixture_dir_inserts_records(
        self, store: VaultStore, settings: Settings
    ) -> None:
        out = self._run(f"/scan {FIXTURES}", store, settings)
        assert "new" in out.lower()
        records = store.get_records()
        assert len(records) >= 3

    def test_scan_fixture_dir_twice_skips_all(
        self, store: VaultStore, settings: Settings
    ) -> None:
        # First scan
        self._run(f"/scan {FIXTURES}", store, settings)
        # Second scan — all files already known
        out = self._run(f"/scan {FIXTURES}", store, settings)
        assert "already in the vault" in out.lower() or "already known" in out.lower()

    def test_scan_pre_filter_counts_match(
        self, store: VaultStore, settings: Settings
    ) -> None:
        """After first scan, second scan should report 0 pending (bar shows 0 files)."""
        from egovault.adapters.local_inbox import SUPPORTED_SUFFIXES
        from egovault.utils.hashing import compute_file_id

        self._run(f"/scan {FIXTURES}", store, settings)

        all_files = [
            f for f in FIXTURES.rglob("*")
            if f.is_file() and f.suffix.lower() in SUPPORTED_SUFFIXES
        ]
        pending = []
        for f in all_files:
            st = f.stat()
            fid = compute_file_id(str(f), st.st_mtime, st.st_size)
            if not store.is_file_known(fid):
                pending.append(f)

        assert pending == [], "All fixture files should be in ingested_files after first scan"

    def test_scan_content_hash_dedup_skips_renamed_copy(
        self, store: VaultStore, settings: Settings, tmp_path: Path
    ) -> None:
        """A renamed copy of an already-ingested file is skipped via content hash."""
        src_file = FIXTURES / "meeting_notes.md"
        dir_a = tmp_path / "a"
        dir_a.mkdir()
        dir_b = tmp_path / "b"
        dir_b.mkdir()
        shutil.copy2(src_file, dir_a / "meeting_notes.md")
        shutil.copy2(src_file, dir_b / "renamed_copy.md")

        # Scan dir_a first
        self._run(f"/scan {dir_a}", store, settings)
        records_before = len(store.get_records())
        assert records_before == 1

        # Scan dir_b — identical content, different name + path → skipped
        out = self._run(f"/scan {dir_b}", store, settings)
        records_after = len(store.get_records())
        assert records_after == 1, "Renamed copy should be deduplicated by content hash"

    def test_scan_subfolder_files_are_discovered(
        self, store: VaultStore, settings: Settings, tmp_path: Path
    ) -> None:
        """Files nested in subdirectories should be found and scanned."""
        sub = tmp_path / "level1" / "level2"
        sub.mkdir(parents=True)
        (sub / "deep_note.md").write_text("# Deep Note\nSome content here.")

        out = self._run(f"/scan {tmp_path}", store, settings)
        assert "new" in out.lower()
        records = store.get_records()
        assert any("deep_note.md" in (r.file_path or "") for r in records)
