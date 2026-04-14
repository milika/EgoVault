"""Gmail OAuth2 authentication helpers for EgoVault.

All heavy imports (google-auth-oauthlib, google-api-python-client) are done
lazily inside each function so that EgoVault starts without error when the
optional ``[gmail]`` extras are not installed.

Missing packages are **auto-installed** the first time a Gmail command is
used — no manual ``pip install`` step required.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# The single Gmail scope EgoVault needs — read-only access to the mailbox.
SCOPES: list[str] = ["https://www.googleapis.com/auth/gmail.readonly"]

# Token file name stored in data/.
TOKEN_FILENAME: str = "gmail_token.json"


_GMAIL_PACKAGES: list[str] = [
    "google-auth",
    "google-auth-oauthlib",
    "google-api-python-client",
]

# Maps PyPI package name → importable module name for existence checks.
_IMPORT_NAMES: dict[str, str] = {
    "google-auth": "google.auth",
    "google-auth-oauthlib": "google_auth_oauthlib",
    "google-api-python-client": "googleapiclient",
}


def _find_missing() -> list[str]:
    """Return a list of PyPI package names that are not yet importable."""
    import importlib
    missing: list[str] = []
    for pkg, mod in _IMPORT_NAMES.items():
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(pkg)
    return missing


def _require_deps() -> None:
    """Ensure all Gmail dependencies are installed, auto-installing if needed.

    On the first call where deps are absent, runs ``pip install`` silently
    and retries the imports.  If installation fails, raises ``ImportError``
    with a clear message.
    """
    missing = _find_missing()
    if not missing:
        return

    # Auto-install into the running interpreter's environment.
    print(
        f"[EgoVault] Gmail extras not installed — installing: {', '.join(missing)} …",
        flush=True,
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "pip", "install", "--quiet", *missing],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ImportError(
            "Failed to install Gmail dependencies automatically.\n"
            f"Packages tried: {', '.join(missing)}\n"
            f"pip output:\n{result.stderr.strip()}\n\n"
            "Fix manually:  pip install egovault\n"
            "or install just these packages: "
            f"pip install {' '.join(missing)}"
        )

    # Retry imports after installation.
    still_missing = _find_missing()
    if still_missing:
        raise ImportError(
            "Gmail dependencies installed but still not importable — "
            "try restarting EgoVault.\n"
            f"Missing: {', '.join(still_missing)}"
        )


def get_token_path(data_dir: str | Path) -> Path:
    """Return the canonical path for the OAuth token file in *data_dir*."""
    return Path(data_dir) / TOKEN_FILENAME


def run_oauth_flow(credentials_path: Path, token_path: Path) -> object:
    """Open the browser for Google OAuth2 and save the resulting token.

    Uses a randomly-assigned localhost port for the redirect — no fixed
    port needs to be opened or configured by the user.

    Args:
        credentials_path: Path to the ``client_secret_*.json`` file downloaded
            from Google Cloud Console (Desktop app OAuth credentials).
        token_path: Where to save the resulting token JSON.  The parent
            directory is created automatically if needed.

    Returns:
        The authenticated ``google.oauth2.credentials.Credentials`` object.

    Raises:
        ImportError: If Gmail dependencies are missing in the current environment.
        FileNotFoundError: If *credentials_path* does not exist.
    """
    _require_deps()
    from google_auth_oauthlib.flow import InstalledAppFlow  # type: ignore[import-untyped]

    if not credentials_path.exists():
        raise FileNotFoundError(
            f"Credentials file not found: {credentials_path}\n"
            "Download it from Google Cloud Console → APIs & Services → Credentials."
        )

    flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path), SCOPES)
    # port=0 → OS picks a free port automatically; no manual firewall config needed.
    creds = flow.run_local_server(port=0, open_browser=True)

    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(creds.to_json())
    return creds


def run_oauth_flow_embedded(
    client_id: str,
    client_secret: str,
    token_path: Path,
) -> object:
    """Run the OAuth2 browser flow using an embedded ``client_id``/``client_secret``.

    This is the zero-friction path: no ``client_secret_*.json`` download needed.
    The caller supplies the OAuth client credentials directly (e.g. read from
    ``egovault.toml`` ``[gmail]`` or the ``EGOVAULT_GMAIL_CLIENT_ID`` /
    ``EGOVAULT_GMAIL_CLIENT_SECRET`` env vars).

    Args:
        client_id: OAuth2 client ID from Google Cloud Console.
        client_secret: OAuth2 client secret from Google Cloud Console.
        token_path: Where to save the resulting token JSON.

    Returns:
        The authenticated ``google.oauth2.credentials.Credentials`` object.

    Raises:
        ImportError: If Gmail dependencies are missing in the current environment.
    """
    _require_deps()
    from google_auth_oauthlib.flow import InstalledAppFlow  # type: ignore[import-untyped]

    # Build the same dict structure that a ``client_secret_*.json`` contains.
    client_config = {
        "installed": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": ["http://localhost"],
        }
    }
    flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
    creds = flow.run_local_server(port=0, open_browser=True)

    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(creds.to_json())
    return creds


def load_credentials(token_path: Path) -> object | None:
    """Load and (if necessary) refresh the saved OAuth token.

    Returns:
        A valid ``google.oauth2.credentials.Credentials`` object, or ``None``
        if no token file exists or the token cannot be refreshed.
    """
    _require_deps()
    from google.auth.transport.requests import Request  # type: ignore[import-untyped]
    from google.oauth2.credentials import Credentials  # type: ignore[import-untyped]

    if not token_path.exists():
        return None

    try:
        creds: Credentials = Credentials.from_authorized_user_file(
            str(token_path), SCOPES
        )
    except Exception:
        return None

    if not creds.valid:
        if creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                token_path.write_text(creds.to_json())
            except Exception:
                return None
        else:
            return None

    return creds


def build_service(creds: object) -> object:
    """Return an authenticated Gmail API service resource.

    Args:
        creds: A valid ``google.oauth2.credentials.Credentials`` object
            (returned by :func:`load_credentials` or :func:`run_oauth_flow`).

    Returns:
        A ``googleapiclient.discovery.Resource`` for ``gmail v1``.
    """
    _require_deps()
    from googleapiclient.discovery import build  # type: ignore[import-untyped]

    return build("gmail", "v1", credentials=creds)
