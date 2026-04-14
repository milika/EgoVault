from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from egovault.chat.session import _resolve_write_target_path


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only path handling")
def test_resolve_write_target_path_maps_linux_desktop_prefix_on_windows() -> None:
    desktop = Path(r"C:\Users\tester\Desktop")
    with patch("egovault.chat.session.os.name", "nt"), patch(
        "egovault.chat.session.resolve_folder", return_value=desktop
    ):
        resolved = _resolve_write_target_path("/home/user/Desktop/all_emails.txt")

    assert resolved == desktop / "all_emails.txt"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only path handling")
def test_resolve_write_target_path_maps_backslash_linux_style_on_windows() -> None:
    desktop = Path(r"C:\Users\tester\Desktop")
    with patch("egovault.chat.session.os.name", "nt"), patch(
        "egovault.chat.session.resolve_folder", return_value=desktop
    ):
        resolved = _resolve_write_target_path(r"\home\user\Desktop\notes.txt")

    assert resolved == desktop / "notes.txt"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only path handling")
def test_resolve_write_target_path_keeps_non_desktop_paths() -> None:
    with patch("egovault.chat.session.os.name", "nt"):
        resolved = _resolve_write_target_path(r"D:\exports\all_emails.txt")

    assert resolved == Path(r"D:\exports\all_emails.txt")
