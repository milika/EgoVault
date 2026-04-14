"""Cross-platform resolution of well-known user folder aliases.

Supports Windows (with registry fallback for OneDrive-redirected paths),
macOS, and Linux (XDG user-dirs).

Usage
-----
    from egovault.utils.folders import resolve_folder, KNOWN_FOLDERS

    path = resolve_folder("desktop")   # returns a resolved Path
    path = resolve_folder("~/notes")   # plain paths are expanded and resolved
    path = resolve_folder("/tmp/work") # absolute paths are returned as-is
"""
from __future__ import annotations

import os
import platform
from pathlib import Path

# ---------------------------------------------------------------------------
# Alias table — canonical name → human label (used in help text and errors)
# ---------------------------------------------------------------------------

KNOWN_FOLDERS: dict[str, str] = {
    "home":       "Home directory",
    "desktop":    "Desktop",
    "documents":  "Documents",
    "downloads":  "Downloads",
    "pictures":   "Pictures",
    "music":      "Music",
    "videos":     "Videos",
    "movies":     "Movies/Videos",
}

# macOS uses "Movies" instead of "Videos"
_MACOS_FOLDER_NAMES: dict[str, str] = {
    "videos": "Movies",
    "movies": "Movies",
}

# Linux XDG environment variables that map to each alias
_XDG_VARS: dict[str, str] = {
    "desktop":   "XDG_DESKTOP_DIR",
    "documents": "XDG_DOCUMENTS_DIR",
    "downloads": "XDG_DOWNLOAD_DIR",
    "pictures":  "XDG_PICTURES_DIR",
    "music":     "XDG_MUSIC_DIR",
    "videos":    "XDG_VIDEOS_DIR",
    "movies":    "XDG_VIDEOS_DIR",
}

# Windows CSIDLs / KnownFolder GUIDs for the registry key path lookup
# HKCU\Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders
_WINDOWS_SHELL_KEYS: dict[str, str] = {
    "desktop":   "Desktop",
    "documents": "Personal",
    "downloads": "{374DE290-123F-4565-9164-39C4925E467B}",
    "pictures":  "My Pictures",
    "music":     "My Music",
    "videos":    "My Video",
    "movies":    "My Video",
}


# ---------------------------------------------------------------------------
# Platform helpers
# ---------------------------------------------------------------------------


def _resolve_windows(alias: str) -> Path | None:
    """Try to read the real path from the Windows registry (handles OneDrive)."""
    import winreg  # only available on Windows

    reg_key = _WINDOWS_SHELL_KEYS.get(alias)
    if reg_key is None:
        return None

    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders",
        ) as key:
            raw_value, _ = winreg.QueryValueEx(key, reg_key)
            # Values may contain %USERPROFILE% or other env vars
            expanded = os.path.expandvars(raw_value)
            path = Path(expanded)
            if path.exists():
                return path
    except (OSError, FileNotFoundError):
        pass

    # Fallback: USERPROFILE\<FolderName>
    userprofile = Path(os.environ.get("USERPROFILE", Path.home()))
    folder_name = alias.capitalize()
    fallback = userprofile / folder_name
    if fallback.exists():
        return fallback
    return None


def _resolve_linux(alias: str) -> Path | None:
    """Try XDG_*_DIR env vars, then parse ~/.config/user-dirs.dirs."""
    xdg_var = _XDG_VARS.get(alias)
    if xdg_var:
        val = os.environ.get(xdg_var)
        if val:
            p = Path(val)
            if p.exists():
                return p

    # Parse ~/.config/user-dirs.dirs (set by xdg-user-dirs on most distros)
    user_dirs_file = Path.home() / ".config" / "user-dirs.dirs"
    if user_dirs_file.exists() and xdg_var:
        try:
            for line in user_dirs_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith(xdg_var + "="):
                    raw = line.split("=", 1)[1].strip().strip('"')
                    expanded = os.path.expandvars(raw.replace("$HOME", str(Path.home())))
                    p = Path(expanded)
                    if p.exists():
                        return p
        except OSError:
            pass

    # Fallback: ~/FolderName
    folder_name = alias.capitalize()
    fallback = Path.home() / folder_name
    if fallback.exists():
        return fallback
    return None


def _resolve_macos(alias: str) -> Path | None:
    """Resolve standard macOS user folder (~/Desktop, ~/Documents, etc.)."""
    folder_name = _MACOS_FOLDER_NAMES.get(alias, alias.capitalize())
    p = Path.home() / folder_name
    return p if p.exists() else None


def _default_folder_name(alias: str) -> Path:
    """Last-resort: ~/alias-capitalised (e.g., ~/Downloads)."""
    return Path.home() / alias.capitalize()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_folder(folder: str) -> Path:
    """Resolve *folder* to an absolute ``Path``.

    *folder* may be:
    - A well-known alias:   ``"desktop"``, ``"documents"``, ``"downloads"``,
      ``"pictures"``, ``"music"``, ``"videos"`` / ``"movies"``, ``"home"``
    - A ``~``-prefixed path: ``"~/notes"``
    - Any absolute or relative path: ``"/tmp/work"``, ``"./data"``

    Raises ``ValueError`` if an alias cannot be resolved to an existing
    directory.  Plain paths are returned after ``expanduser`` + ``resolve``
    without an existence check (the caller can decide).
    """
    alias = folder.strip().lower()

    # home alias
    if alias == "home":
        return Path.home()

    if alias in KNOWN_FOLDERS:
        resolved: Path | None = None
        system = platform.system()

        if system == "Windows":
            resolved = _resolve_windows(alias)
        elif system == "Darwin":
            resolved = _resolve_macos(alias)
        else:  # Linux and other Unix-likes
            resolved = _resolve_linux(alias)

        if resolved is None:
            # Final fallback common to all platforms
            candidate = _default_folder_name(alias)
            if candidate.exists():
                resolved = candidate

        if resolved is None:
            known = ", ".join(sorted(KNOWN_FOLDERS))
            raise ValueError(
                f"Could not locate well-known folder '{alias}' on this system. "
                f"Known aliases: {known}. Pass an explicit path instead."
            )
        return resolved.resolve()

    # Plain path — expand ~ and resolve
    return Path(os.path.expanduser(folder)).resolve()


def list_known_folders() -> list[tuple[str, Path | None]]:
    """Return (alias, resolved_path_or_None) for every known alias.

    Useful for the ``scan-folder --list`` flag.
    """
    results: list[tuple[str, Path | None]] = []
    for alias in KNOWN_FOLDERS:
        try:
            results.append((alias, resolve_folder(alias)))
        except ValueError:
            results.append((alias, None))
    return results
