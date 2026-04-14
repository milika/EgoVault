"""Auto-update helper for EgoVault.

On every interactive startup EgoVault checks whether a newer version is
available.  If one is found it shows a prompt::

    Update available  (3 new commits on origin/main)
    Apply update? [y/N]  (auto-skip in 15 s):

The prompt times out after 15 seconds and defaults to **N**.
If the user answers Y the update is applied and the process restarts.

Detection strategy
------------------
* Git repo (dev / editable install)
    Compare local HEAD against ``origin/main`` via ``git fetch`` + diff.
* pipx install
    Query PyPI JSON API for the latest published version and compare.
* Plain pip / venv
    Same PyPI check.

All network calls use short timeouts and fail silently on any error so a
missing network connection never blocks startup.
"""
from __future__ import annotations

import importlib.metadata
import logging
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import NamedTuple

from rich.console import Console

logger = logging.getLogger(__name__)

# Project root: utils/ → egovault/ → project root
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# ── data ─────────────────────────────────────────────────────────────────────

class UpdateInfo(NamedTuple):
    mode: str          # "git" | "pipx" | "pip"
    description: str   # human-readable, e.g. "3 new commits" or "0.1.0 → 0.2.0"


# ── internal helpers ─────────────────────────────────────────────────────────

def _run(
    *args: str,
    cwd: Path | None = None,
    timeout: int = 10,
) -> subprocess.CompletedProcess:  # type: ignore[type-arg]
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=cwd,
    )


def _detect_mode() -> str:
    """Return "git" if inside a git repo, "pipx" if installed via pipx, else "pip"."""
    try:
        r = _run("git", "rev-parse", "--is-inside-work-tree", cwd=_REPO_ROOT, timeout=5)
        if r.returncode == 0:
            return "git"
    except Exception:
        pass

    exe_lower = sys.executable.lower()
    if "pipx" in exe_lower or "pipx" in str(Path(sys.executable).parent).lower():
        return "pipx"

    return "pip"


def _check_git() -> UpdateInfo | None:
    """Return UpdateInfo if origin/main is ahead of HEAD, else None."""
    try:
        _run("git", "fetch", "origin", "main", "--quiet", cwd=_REPO_ROOT, timeout=10)
        local = _run("git", "rev-parse", "HEAD", cwd=_REPO_ROOT).stdout.strip()
        remote = _run("git", "rev-parse", "origin/main", cwd=_REPO_ROOT).stdout.strip()
        if local == remote:
            return None
        count_r = _run(
            "git", "rev-list", "--count", f"{local}..origin/main", cwd=_REPO_ROOT
        )
        count = count_r.stdout.strip() or "?"
        noun = "commit" if count == "1" else "commits"
        return UpdateInfo("git", f"{count} new {noun} on origin/main")
    except Exception as exc:
        logger.debug("Git update check failed: %s", exc)
        return None


def _check_pypi(mode: str) -> UpdateInfo | None:
    """Return UpdateInfo if PyPI has a newer version than what is installed, else None."""
    import json
    import urllib.request

    try:
        current = importlib.metadata.version("egovault")
    except importlib.metadata.PackageNotFoundError:
        return None

    try:
        req = urllib.request.Request(
            "https://pypi.org/pypi/egovault/json",
            headers={"User-Agent": "EgoVault-updater/1.0"},
        )
        with urllib.request.urlopen(req, timeout=5) as r:  # noqa: S310
            data = json.loads(r.read())
        latest: str = data["info"]["version"]
    except Exception as exc:
        logger.debug("PyPI update check failed: %s", exc)
        return None

    if latest == current:
        return None

    def _ver(v: str) -> tuple[int, ...]:
        try:
            return tuple(int(x) for x in v.split("."))
        except ValueError:
            return (0,)

    if _ver(latest) <= _ver(current):
        return None

    return UpdateInfo(mode, f"{current} → {latest}")


# ── public API ────────────────────────────────────────────────────────────────

def check_for_update() -> UpdateInfo | None:
    """Return an UpdateInfo if a newer version is available, else None.

    All network calls are guarded and fail silently on any error.
    """
    mode = _detect_mode()
    if mode == "git":
        return _check_git()
    return _check_pypi(mode)


def apply_update(info: UpdateInfo, console: Console) -> bool:
    """Apply the update described in *info*.  Returns True on success."""
    try:
        if info.mode == "git":
            console.print("[dim]  Running git pull…[/dim]")
            r = _run(
                "git", "pull", "--ff-only", "origin", "main",
                cwd=_REPO_ROOT, timeout=60,
            )
            if r.returncode != 0:
                console.print(f"[red]  git pull failed:[/red] {r.stderr.strip()}")
                return False
            # Reinstall in case pyproject.toml / dependencies changed (editable install).
            console.print("[dim]  Reinstalling dependencies…[/dim]")
            _run(
                sys.executable, "-m", "pip", "install", "-e", ".", "-q",
                cwd=_REPO_ROOT, timeout=120,
            )
            return True

        if info.mode == "pipx":
            console.print("[dim]  Running pipx upgrade egovault…[/dim]")
            r = _run("pipx", "upgrade", "egovault", timeout=120)
            if r.returncode != 0:
                console.print(f"[red]  pipx upgrade failed:[/red] {r.stderr.strip()}")
                return False
            return True

        # plain pip
        console.print("[dim]  Running pip install --upgrade egovault…[/dim]")
        r = _run(
            sys.executable, "-m", "pip", "install", "--upgrade", "egovault", "-q",
            timeout=120,
        )
        if r.returncode != 0:
            console.print(f"[red]  pip upgrade failed:[/red] {r.stderr.strip()}")
            return False
        return True

    except Exception as exc:
        console.print(f"[red]  Update error:[/red] {exc}")
        return False


def restart(console: Console) -> None:
    """Re-exec the current process with the same arguments.

    On POSIX this calls ``os.execv`` and never returns.
    On Windows this spawns a new process and exits the current one.
    """
    console.print("[dim]Restarting EgoVault…[/dim]")
    if sys.platform == "win32":
        result = subprocess.run(sys.argv)  # noqa: S603
        sys.exit(result.returncode)
    else:
        os.execv(sys.argv[0], sys.argv)  # noqa: S606


def _timed_input(prompt: str, timeout: float) -> str:
    """Display *prompt*, wait up to *timeout* seconds for a line.

    Returns the stripped input string, or "" on timeout or EOF.
    """
    result: list[str] = []
    done = threading.Event()

    def _read() -> None:
        try:
            result.append(input(prompt))
        except (EOFError, OSError):
            pass
        done.set()

    t = threading.Thread(target=_read, daemon=True)
    t.start()
    timed_out = not done.wait(timeout)
    if timed_out:
        # The input() call is still blocking in the daemon thread; move to a
        # new line so subsequent Rich output is not mangled.
        print()
    return result[0].strip() if result else ""


def prompt_and_maybe_update(console: Console) -> bool:
    """Check for an update and prompt the user if one is available.

    Returns True if an update was applied and the caller should restart the
    process.  Skips silently when stdout is not a tty (piped / scripted).
    """
    if not sys.stdout.isatty():
        return False

    try:
        info = check_for_update()
    except Exception:
        return False

    if info is None:
        return False

    console.print(
        f"\n[bold yellow]  Update available[/bold yellow]  "
        f"[dim]({info.description})[/dim]"
    )
    answer = _timed_input(
        "  Apply update? [y/N]  (auto-skip in 15 s): ",
        timeout=15.0,
    )

    if answer.lower() not in ("y", "yes"):
        console.print("[dim]  Skipped.[/dim]\n")
        return False

    ok = apply_update(info, console)
    if ok:
        console.print("[green]  Update applied successfully.[/green]")
        return True

    return False
