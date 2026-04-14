"""Bootstrap — check/start an LLM server before enrichment/chat.

EgoVault can either connect to an externally-started server
(``[llama_cpp] manage = false``) or manage the server lifecycle itself
when ``[llama_cpp] manage = true``.

When manage = true, EgoVault:
  1. Auto-downloads the GGUF from HuggingFace if model_path is missing.
  2. Queries FREE GPU VRAM via nvidia-smi and computes ctx-size so the
     KV cache uses at most ``vram_budget_pct`` (default 80 %) of free VRAM.
  3. Tries to start the ``llama-server`` binary (in PATH or ./bin/).
  4. If llama-server is not found, downloads it automatically from the
     latest llama.cpp GitHub release (Windows CUDA build preferred).
  5. If the binary download fails, installs ``llama-cpp-python[server]``
     via pip (CUDA pre-built wheels first, then CPU fallback) and starts
     ``python -m llama_cpp.server`` instead.
  6. Registers an atexit handler to stop the server when Python exits.
"""
from __future__ import annotations

import atexit
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console

if TYPE_CHECKING:
    from egovault.config import Settings

logger = logging.getLogger(__name__)

# Module-level handle kept so atexit can kill it.
_server_proc: subprocess.Popen | None = None  # type: ignore[type-arg]


# ── private helpers ───────────────────────────────────────────────────────────

def _is_reachable(base_url: str, timeout: int = 3) -> bool:
    """Return True if llama-server answers GET /health or /v1/models."""
    for path in ("/health", "/v1/models"):
        try:
            with urllib.request.urlopen(  # noqa: S310
                f"{base_url.rstrip('/')}{path}", timeout=timeout
            ) as r:
                if r.status == 200:
                    logger.debug("llama-server reachable at %s%s", base_url, path)
                    return True
        except Exception:
            continue
    return False


def _stop_server() -> None:
    """atexit hook: terminate the managed server process."""
    global _server_proc
    if _server_proc is not None and _server_proc.poll() is None:
        logger.info("Stopping managed LLM server (pid %d)\u2026", _server_proc.pid)
        _server_proc.terminate()
        try:
            _server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _server_proc.kill()
        _server_proc = None


def _detect_cuda_tag() -> str | None:
    """Return nearest available CUDA wheel tag (e.g. ``'cu124'``), or None.

    Parses ``nvidia-smi`` for the installed CUDA version and maps it to the
    nearest pre-built wheel tag available on abetlen.github.io.  If the exact
    version has no wheel, the closest older supported tag is returned.
    """
    import re as _re
    # Pre-built wheel tags available on abetlen.github.io/llama-cpp-python/whl/
    _SUPPORTED = [121, 122, 123, 124, 125, 126]
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=10
        )
        m = _re.search(r"CUDA Version:\s*(\d+)\.(\d+)", result.stdout)
        if m:
            our_num = int(m.group(1)) * 10 + int(m.group(2))
            best = max((n for n in _SUPPORTED if n <= our_num), default=None)
            if best:
                return f"cu{best}"
    except Exception:
        pass
    return None


def _ensure_llama_cpp_python(console: Console) -> bool:
    """Install ``llama-cpp-python[server]`` into the running venv if needed.

    Tries CUDA pre-built wheels first (detected via nvidia-smi and mapped to
    the nearest supported wheel tag), using ``--only-binary`` to avoid slow
    source compilation.  Falls back to a plain ``pip install`` from PyPI which
    will use a binary wheel if one exists for the running Python version.
    Returns True if the package is importable after the attempt.
    """
    import importlib

    try:
        import llama_cpp  # noqa: F401
        return True
    except ImportError:
        pass

    pkg = "llama-cpp-python[server]"
    cuda_tag = _detect_cuda_tag()

    if cuda_tag:
        wheel_url = f"https://abetlen.github.io/llama-cpp-python/whl/{cuda_tag}"
        console.print(
            f"[dim]Installing [bold]{pkg}[/bold] with CUDA pre-built wheels "
            f"({cuda_tag})\u2026[/dim]"
        )
        result = subprocess.run(  # noqa: S603
            [sys.executable, "-m", "pip", "install", pkg,
             "--extra-index-url", wheel_url,
             "--only-binary=llama-cpp-python",  # never compile from source
             "--quiet"],
        )
        if result.returncode == 0:
            importlib.invalidate_caches()
            try:
                import llama_cpp  # noqa: F401
                return True
            except ImportError:
                pass
        console.print("[dim]CUDA wheel not available \u2014 trying PyPI\u2026[/dim]")

    # Fall back to plain PyPI install with --only-binary (no source compile).
    console.print(f"[dim]Installing [bold]{pkg}[/bold] from PyPI\u2026[/dim]")
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "pip", "install", pkg,
         "--only-binary=llama-cpp-python", "--quiet"],
    )
    if result.returncode != 0:
        return False
    importlib.invalidate_caches()
    try:
        import llama_cpp  # noqa: F401
        return True
    except ImportError:
        return False


# Bin dir for auto-downloaded llama-server binary (project-local, gitignored).
_BIN_DIR = Path("./bin").resolve()


def _llama_server_exe() -> str | None:
    """Return absolute path to an existing llama-server executable, or None."""
    if p := shutil.which("llama-server"):
        return p
    local = _BIN_DIR / ("llama-server.exe" if sys.platform == "win32" else "llama-server")
    if local.exists():
        return str(local)
    return None


# Sidecar file that records which release asset is currently in _BIN_DIR.
_ASSET_SOURCE = _BIN_DIR / ".llama-server-source"


def _auto_download_llama_server(
    console: Console,
    blacklist: frozenset[str] = frozenset(),
) -> str | None:
    """Download the llama-server binary from the latest llama.cpp GitHub release.

    Picks a Windows x64 CUDA build when available, otherwise the CPU build.
    Extracts ``llama-server.(exe)`` to ``./bin/`` and returns its path.
    *blacklist* is a set of asset names to never pick (used after a crash-retry).
    Returns None on any error.
    """
    from rich.progress import (
        BarColumn, DownloadColumn, Progress, TimeRemainingColumn, TransferSpeedColumn,
    )

    _BIN_DIR.mkdir(parents=True, exist_ok=True)
    exe_name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
    dest = _BIN_DIR / exe_name

    api_url = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
    try:
        req = urllib.request.Request(
            api_url, headers={"Accept": "application/vnd.github.v3+json",
                               "User-Agent": "EgoVault/1.0"}
        )
        with urllib.request.urlopen(req, timeout=15) as r:  # noqa: S310
            release = json.loads(r.read())
    except Exception as exc:
        logger.error("Failed to fetch llama.cpp release info: %s", exc)
        return None

    assets = release.get("assets", [])

    import platform as _platform
    _is_arm = _platform.machine().lower() in ("arm64", "aarch64")

    def _score(name: str) -> int:
        n = name.lower()
        is_archive = n.endswith(".zip") or n.endswith(".tar.gz")
        if not is_archive:
            return -1
        if not n.startswith("llama-") or "-bin-" not in n:
            return -1  # skip cudart, source tarballs, xcframework, etc.
        if sys.platform == "win32":
            if "win" not in n or not n.endswith(".zip"):
                return -1
            if "x64" not in n and "arm64" not in n:
                return -1
        elif sys.platform == "darwin":
            if "macos" not in n and "osx" not in n and "apple" not in n:
                return -1
        else:  # linux — llama.cpp uses "ubuntu" not "linux" in asset names
            if "linux" not in n and "ubuntu" not in n:
                return -1
        score = 0
        if sys.platform == "darwin":
            if _is_arm and "arm64" in n:
                score += 5  # prefer native arm64 on Apple Silicon
            elif not _is_arm and "x64" in n:
                score += 5
        else:  # linux: prefer matching arch
            if _is_arm and "arm64" in n:
                score += 5
            elif not _is_arm and ("x64" in n or "x86_64" in n):
                score += 5
        if "cuda" in n:
            score += 10
        if "avx2" in n:
            score += 2
        elif "avx" in n:
            score += 1
        # Penalise specialised builds that need specific hardware/drivers
        if "kleidiai" in n or "no-metal" in n or "openvino" in n or "rocm" in n or "vulkan" in n:
            score -= 3
        return score

    # Filter out any assets the caller wants to skip (e.g. previously crashed).
    candidates = [a for a in assets if a["name"] not in blacklist]
    best = max(candidates, key=lambda a: _score(a["name"]), default=None)
    if not best or _score(best["name"]) < 0:
        logger.error("No suitable llama-server asset found for this platform")
        return None

    def _download_archive(url: str, name: str) -> Path | None:
        """Download an archive (.zip or .tar.gz) and return its local path."""
        dest_arc = _BIN_DIR / name
        try:
            with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310
                total = int(t) if (t := response.headers.get("Content-Length")) else None
                with Progress(
                    "[progress.description]{task.description}",
                    BarColumn(), DownloadColumn(), TransferSpeedColumn(), TimeRemainingColumn(),
                    console=console, transient=True,
                ) as progress:
                    task = progress.add_task(f"  {name}", total=total)
                    with open(dest_arc, "wb") as fh:
                        while chunk := response.read(1 << 20):
                            fh.write(chunk)
                            progress.update(task, advance=len(chunk))
        except Exception as exc:
            logger.error("Download failed: %s", exc)
            dest_arc.unlink(missing_ok=True)
            return None
        return dest_arc

    def _extract_to_bin(arc_path: Path) -> None:
        """Extract llama-server binary (and DLLs on Windows) to _BIN_DIR."""
        try:
            name_lower = arc_path.name.lower()
            if name_lower.endswith(".tar.gz"):
                import tarfile
                with tarfile.open(arc_path) as tf:
                    for member in tf.getmembers():
                        fname = Path(member.name).name.lower()
                        if fname in ("llama-server", "llama-server.exe"):
                            src = tf.extractfile(member)
                            if src:
                                dest_file = _BIN_DIR / Path(member.name).name
                                dest_file.write_bytes(src.read())
                                dest_file.chmod(0o755)
            else:
                with zipfile.ZipFile(arc_path) as zf:
                    for member in zf.namelist():
                        fname = Path(member).name.lower()
                        is_binary = (
                            fname in ("llama-server", "llama-server.exe")
                            or fname.endswith(".dll")
                        )
                        if is_binary:
                            d = _BIN_DIR / Path(member).name
                            d.write_bytes(zf.read(member))
                            if sys.platform != "win32":
                                d.chmod(0o755)
        except Exception as exc:
            logger.error("Extraction error for %s: %s", arc_path.name, exc)
        finally:
            arc_path.unlink(missing_ok=True)

    console.print(
        f"[bold]Auto-downloading[/bold] [cyan]{best['name']}[/cyan] "
        f"from llama.cpp releases\u2026"
    )
    arc_path = _download_archive(best["browser_download_url"], best["name"])
    if not arc_path:
        return None
    _extract_to_bin(arc_path)

    # Also download the matching CUDA runtime DLLs (cudart-*) on Windows.
    import re as _re
    m = _re.search(r"cuda[-_]([\d.]+)", best["name"].lower())
    if m and sys.platform == "win32":
        cuda_ver = m.group(1)
        cudart_name = f"cudart-llama-bin-win-cuda-{cuda_ver}-x64.zip"
        cudart_asset = next(
            (a for a in assets if a["name"].lower() == cudart_name.lower()), None
        )
        if cudart_asset:
            console.print(f"[dim]Fetching CUDA runtime DLLs ({cudart_name})\u2026[/dim]")
            cudart_arc = _download_archive(cudart_asset["browser_download_url"], cudart_name)
            if cudart_arc:
                _extract_to_bin(cudart_arc)

    if not dest.exists():
        logger.error("%s not found in zip archive", exe_name)
        return None

    # Record which asset is installed so crash-retry can exclude it.
    try:
        _ASSET_SOURCE.parent.mkdir(parents=True, exist_ok=True)
        _ASSET_SOURCE.write_text(best["name"], encoding="utf-8")
    except OSError:
        pass

    console.print(
        f"[green]\u2713[/green] llama-server installed at [bold]{dest}[/bold]"
    )
    return str(dest)


def _download_model(model_path: Path, hf_repo: str, console: Console) -> bool:
    """Download *model_path.name* from ``hf_repo`` (HuggingFace) to *model_path*.

    Uses a ``.part`` temp file so an interrupted download never leaves a
    corrupt GGUF in place.  Returns True on success.
    """
    from rich.progress import (
        BarColumn,
        DownloadColumn,
        Progress,
        TimeRemainingColumn,
        TransferSpeedColumn,
    )

    filename = model_path.name
    url = f"https://huggingface.co/{hf_repo}/resolve/main/{filename}"
    tmp_path = model_path.with_suffix(".gguf.part")

    model_path.parent.mkdir(parents=True, exist_ok=True)

    console.print(
        f"[bold]Auto-downloading[/bold] [cyan]{filename}[/cyan] "
        f"from [link={url}]{hf_repo}[/link] …"
    )

    try:
        with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310
            content_length = response.headers.get("Content-Length")
            total = int(content_length) if content_length else None

            with Progress(
                "[progress.description]{task.description}",
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
                console=console,
                transient=True,
            ) as progress:
                task = progress.add_task(f"  {filename}", total=total)
                with open(tmp_path, "wb") as fh:
                    while chunk := response.read(1 << 20):  # 1 MB chunks
                        fh.write(chunk)
                        progress.update(task, advance=len(chunk))

        tmp_path.rename(model_path)
        size_gb = model_path.stat().st_size / 1_073_741_824
        console.print(
            f"[green]✓[/green] Downloaded [bold]{filename}[/bold]  "
            f"({size_gb:.1f} GB) → {model_path.parent}"
        )
        return True
    except Exception as exc:
        logger.error("Model download failed: %s", exc)
        tmp_path.unlink(missing_ok=True)
        console.print(
            f"[red]EgoVault:[/red] Download failed: {exc}\n"
            f"[dim]URL tried: {url}[/dim]\n"
            "[dim]Place the GGUF manually or update [bold]llama_cpp.model_hf_repo[/bold] "
            "in egovault.toml.[/dim]"
        )
        return False


# ── public API ────────────────────────────────────────────────────────────────

def ensure_llama_server(settings: "Settings", console: Console) -> bool:
    """Ensure llama-server is reachable, starting it automatically if manage = true.

    Returns True if the server is (or becomes) reachable.
    """
    from egovault.utils.llm import ctx_for_vram_budget, query_free_vram_mb  # lazy import

    lcpp = settings.llama_cpp
    base_url = settings.llm.base_url

    # Already running — nothing to do.
    if _is_reachable(base_url):
        return True

    # Not managed: show the startup hint and report failure.
    if not lcpp.manage:
        return check_llama_server(base_url, console)

    # ── Managed startup ──────────────────────────────────────────────────────
    model_path = Path(lcpp.model_path).expanduser().resolve() if lcpp.model_path else None
    if not model_path or not model_path.exists():
        # Auto-download from HuggingFace when a repo is configured.
        if model_path and lcpp.model_hf_repo:
            if not _download_model(model_path, lcpp.model_hf_repo, console):
                return False
        else:
            console.print(
                "[red]EgoVault:[/red] [bold]llama_cpp.manage = true[/bold] but the model file "
                f"was not found: [bold]{model_path or 'model_path not set'}[/bold]\n"
                "[dim]Set [bold]llama_cpp.model_hf_repo[/bold] in egovault.toml for auto-download, "
                "or place the GGUF at [bold]./models/[/bold] manually.[/dim]"
            )
            return False

    # Compute ctx_size using 80 % of currently-free VRAM (model loads unrestricted).
    ctx = lcpp.ctx_size
    if ctx == 0:
        free_vram = query_free_vram_mb()
        if free_vram:
            ctx = ctx_for_vram_budget(
                total_vram_mb=free_vram,
                model_size_mb=0,
                budget_pct=lcpp.vram_budget_pct,
                flash_attn=lcpp.flash_attn,
                cuda_overhead_mb=0,
            )
            console.print(
                f"[dim]llama-server: auto ctx-size = [bold]{ctx}[/bold] tokens  "
                f"({free_vram} MB free VRAM \u00d7 {lcpp.vram_budget_pct:.0%} = "
                f"{int(free_vram * lcpp.vram_budget_pct)} MB → KV cache)[/dim]"
            )
        else:
            ctx = 8192  # conservative fallback when nvidia-smi unavailable

    # Parse host and port from base_url.
    parsed = urllib.parse.urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8080

    # ── Choose backend: llama-server binary or llama_cpp.server Python module ─
    llama_exe = _llama_server_exe()
    if not llama_exe:
        console.print(
            "[dim]llama-server not found in PATH \u2014 "
            "auto-downloading from llama.cpp releases\u2026[/dim]"
        )
        llama_exe = _auto_download_llama_server(console)

    if llama_exe:
        cmd = [
            llama_exe,
            "-m", str(model_path),
            "--n-gpu-layers", str(lcpp.n_gpu_layers),
            "--ctx-size", str(ctx),
            "--host", host,
            "--port", str(port),
        ]
        if lcpp.flash_attn:
            cmd += ["--flash-attn", "on"]  # new llama.cpp takes on|off|auto, not bare flag
        if lcpp.embed:
            cmd += ["--embedding", "--pooling", "mean"]
        backend = "llama-server"
    else:
        # Binary download failed — try llama-cpp-python as last resort.
        console.print("[dim]Binary download failed \u2014 trying llama-cpp-python\u2026[/dim]")
        if not _ensure_llama_cpp_python(console):
            py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
            if sys.platform == "darwin":
                hint = (
                    "[dim]On macOS, install llama-server via Homebrew:\n"
                    "  brew install llama.cpp\n"
                    "Then re-run: ego chat[/dim]"
                )
            else:
                hint = (
                    f"[dim]Python {py_ver} may not yet have llama-cpp-python wheels.\n"
                    "Get llama-server directly: "
                    "https://github.com/ggml-org/llama.cpp/releases[/dim]"
                )
            console.print(
                "[red]EgoVault:[/red] Could not start an LLM server.\n" + hint
            )
            return False
        cmd = [
            sys.executable, "-m", "llama_cpp.server",
            "--model", str(model_path),
            "--n_gpu_layers", str(lcpp.n_gpu_layers),
            "--n_ctx", str(ctx),
            "--host", host,
            "--port", str(port),
        ]
        if lcpp.flash_attn:
            cmd += ["--flash_attn", "true"]
        if lcpp.embed:
            cmd += ["--embedding", "true"]
        backend = "llama-cpp-python"

    console.print(
        f"[dim]Starting {backend}  model={model_path.name}  "
        f"ctx={ctx}  flash-attn={'on' if lcpp.flash_attn else 'off'}  "
        f"vram-budget={lcpp.vram_budget_pct:.0%}[/dim]"
    )

    import tempfile as _tempfile
    _stderr_file = _tempfile.NamedTemporaryFile(
        mode="w", suffix="-llama-stderr.txt", delete=False
    )
    global _server_proc
    _server_proc = subprocess.Popen(  # noqa: S603
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=_stderr_file,
        env={**os.environ},
    )
    _stderr_file.close()
    atexit.register(_stop_server)

    # Wait up to 60 s for the server to become reachable.
    deadline = time.monotonic() + 60
    last_dot = 0
    while time.monotonic() < deadline:
        if _server_proc.poll() is not None:
            rc = _server_proc.returncode
            # Show the last few lines of stderr to aid debugging.
            try:
                with open(_stderr_file.name, encoding="utf-8", errors="replace") as _fh:
                    _tail = _fh.read()[-2000:].strip()
            except OSError:
                _tail = ""
            hint = f"\n[dim]{_tail}[/dim]" if _tail else ""

            # ── Crash-retry: if the binary was auto-downloaded and crashed
            # with a signal (negative code on Unix = SIGABRT / SIGILL etc.),
            # delete it and try an alternative release asset. ────────────────
            _local_exe = _BIN_DIR / ("llama-server.exe" if sys.platform == "win32" else "llama-server")
            _is_local = cmd[0] == str(_local_exe) and _local_exe.exists()
            if rc < 0 and _is_local:
                bad_asset = ""
                try:
                    bad_asset = _ASSET_SOURCE.read_text(encoding="utf-8").strip()
                except OSError:
                    pass
                if bad_asset:
                    console.print(
                        f"[yellow]llama-server crashed (signal {-rc}) "
                        f"\u2014 {bad_asset} is incompatible with this CPU. "
                        f"Retrying with a different build\u2026[/yellow]"
                    )
                    _local_exe.unlink(missing_ok=True)
                    _ASSET_SOURCE.unlink(missing_ok=True)
                    llama_exe = _auto_download_llama_server(
                        console, blacklist=frozenset({bad_asset})
                    )
                    if llama_exe:
                        # Update cmd to use new binary and restart.
                        cmd[0] = llama_exe
                        _stderr_file2 = _tempfile.NamedTemporaryFile(  # noqa: SIM115
                            mode="w", suffix="-llama-stderr.txt", delete=False
                        )
                        _server_proc = subprocess.Popen(  # noqa: S603
                            cmd,
                            stdout=subprocess.DEVNULL,
                            stderr=_stderr_file2,
                            env={**os.environ},
                        )
                        _stderr_file2.close()
                        _stderr_file = _stderr_file2
                        deadline = time.monotonic() + 60
                        last_dot = 0
                        continue

            console.print(
                f"[red]EgoVault:[/red] llama-server exited early "
                f"(code {rc}). "
                f"Check the model path and GPU drivers.{hint}"
            )
            return False
        if _is_reachable(base_url, timeout=1):
            console.print(
                f"[green]\u2713[/green] llama-server ready at [bold]{base_url}[/bold]  "
                f"(ctx={ctx}, flash-attn={'on' if lcpp.flash_attn else 'off'})"
            )
            return True
        elapsed = int(time.monotonic() - (deadline - 60))
        if elapsed - last_dot >= 10:
            console.print(f"[dim]  waiting for llama-server\u2026 ({elapsed}s)[/dim]")
            last_dot = elapsed
        time.sleep(2)

    console.print(
        "[red]EgoVault:[/red] llama-server did not become reachable within 60 s."
    )
    return False


def check_llama_server(base_url: str, console: Console) -> bool:
    """Return True if llama-server is reachable, False (with a startup hint) otherwise.

    Tries GET /health (llama-server standard endpoint) then falls back to
    GET /v1/models.  Either returning HTTP 200 is sufficient.

    For automatic server management set ``[llama_cpp] manage = true`` in
    egovault.toml and use :func:`ensure_llama_server` instead.
    """
    if _is_reachable(base_url):
        return True

    console.print(
        f"[red]EgoVault:[/red] llama-server not reachable at [bold]{base_url}[/bold]\n"
        "[yellow]Start it first, for example:[/yellow]\n"
        "  llama-server -m /path/to/model.gguf \\\\\n"
        "      --n-gpu-layers 99 --flash-attn --ctx-size 16384 \\\\\n"
        "      --host 127.0.0.1 --port 8080 --embedding --pooling mean\n"
        "[dim]Or set [bold]llama_cpp.manage = true[/bold] and "
        "[bold]llama_cpp.model_path[/bold] in egovault.toml for auto-start "
        "(EgoVault computes ctx-size for 80 \u0025 of your GPU VRAM automatically).\n"
        "Download Unsloth GGUFs from https://huggingface.co/unsloth[/dim]"
    )
    return False