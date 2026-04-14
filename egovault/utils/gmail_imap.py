"""Gmail IMAP helpers — authenticate with an App Password, no OAuth or GCP needed.

One-time setup (~2 minutes):
  1. Enable 2-Step Verification: https://myaccount.google.com/security
  2. Create an App Password:     https://myaccount.google.com/apppasswords
     (choose "Mail" or any custom name → copy the 16-character code)
  3. Enable IMAP in Gmail:
     Gmail → Settings (⚙) → See all settings →
     Forwarding and POP/IMAP → IMAP access: Enable → Save Changes

No Google Cloud project, no credentials JSON, no browser OAuth flow.
Everything runs over a plain imaplib.IMAP4_SSL connection.
"""
from __future__ import annotations

import imaplib
import json
from datetime import date
from pathlib import Path

# ── connection constants ────────────────────────────────────────────────────

IMAP_HOST: str = "imap.gmail.com"
IMAP_PORT: int = 993
CREDENTIALS_FILENAME: str = "gmail_imap.json"


# ── credential helpers ──────────────────────────────────────────────────────

def get_credentials_path(data_dir: str | Path) -> Path:
    """Return the path to the stored IMAP credentials file."""
    return Path(data_dir) / CREDENTIALS_FILENAME


def save_credentials(
    data_dir: str | Path,
    gmail_address: str,
    app_password: str,
) -> Path:
    """Persist IMAP credentials to *data_dir* and restrict file permissions.

    Args:
        data_dir: Directory that holds EgoVault's data files.
        gmail_address: Full Gmail address (e.g. ``you@gmail.com``).
        app_password: 16-character App Password from Google Account settings.

    Returns:
        Path of the written credentials file.
    """
    path = get_credentials_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"email": gmail_address, "app_password": app_password}),
        encoding="utf-8",
    )
    # Restrict to owner read/write — best-effort; silently ignored on Windows.
    try:
        path.chmod(0o600)
    except NotImplementedError:
        pass
    return path


def load_credentials(data_dir: str | Path) -> tuple[str, str] | None:
    """Return ``(email, app_password)`` from the saved credentials file.

    Returns:
        A ``(email, app_password)`` tuple, or ``None`` if no credentials file
        exists or the file is malformed.
    """
    path = get_credentials_path(data_dir)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data["email"], data["app_password"]
    except (json.JSONDecodeError, KeyError):
        return None


# ── connection helpers ──────────────────────────────────────────────────────

def verify_connection(gmail_address: str, app_password: str) -> None:
    """Open an SSL connection to Gmail IMAP, log in, and log out immediately.

    Raises:
        imaplib.IMAP4.error: If authentication fails (bad address / password /
            IMAP not enabled in Gmail settings).
    """
    mail = imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT)
    try:
        mail.login(gmail_address, app_password)
    finally:
        try:
            mail.logout()
        except Exception:
            pass


def connect(gmail_address: str, app_password: str) -> imaplib.IMAP4_SSL:
    """Open and return an authenticated IMAP4_SSL connection to Gmail."""
    mail = imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT)
    mail.login(gmail_address, app_password)
    return mail


def fetch_attachment_bytes(
    data_dir: str | "Path",
    message_id: str,
    attachment_name: str,
) -> bytes | None:
    """Download a single attachment from Gmail via IMAP and return its raw bytes.

    Searches ``[Gmail]/All Mail`` for the message whose ``Message-ID`` header
    matches *message_id*, then walks its MIME parts to find a part whose
    filename matches *attachment_name* (case-insensitive).

    Args:
        data_dir: Directory holding ``gmail_imap.json`` credentials.
        message_id: RFC 2822 Message-ID header value, e.g.
            ``<CADeJxu4...@mail.gmail.com>``.
        attachment_name: Filename of the attachment to retrieve, e.g.
            ``image.png``.

    Returns:
        Raw attachment bytes, or ``None`` when not found or credentials
        are unavailable.
    """
    import email as _email

    creds = load_credentials(data_dir)
    if creds is None:
        return None
    gmail_address, app_password = creds

    mail = connect(gmail_address, app_password)
    try:
        # Search [Gmail]/All Mail; fall back to INBOX on failure.
        for folder in ('"[Gmail]/All Mail"', "INBOX"):
            status, _ = mail.select(folder, readonly=True)
            if status == "OK":
                break
        else:
            return None

        # IMAP SEARCH by Message-ID header
        search_term = message_id.strip("<>")
        status, data = mail.search(None, f'HEADER Message-ID "{search_term}"')
        if status != "OK" or not data or not data[0]:
            return None

        msg_nums = data[0].split()
        if not msg_nums:
            return None

        # Fetch the RFC 822 payload of the first (and typically only) hit.
        status, msg_data = mail.fetch(msg_nums[0], "(RFC822)")
        if status != "OK" or not msg_data or not msg_data[0]:
            return None

        raw: bytes = msg_data[0][1]  # type: ignore[index]
        msg = _email.message_from_bytes(raw)

        # Walk MIME parts and find the matching attachment.
        att_lower = attachment_name.lower()
        for part in msg.walk():
            disposition = str(part.get("Content-Disposition", ""))
            if "attachment" not in disposition and "inline" not in disposition:
                continue
            filename = part.get_filename() or ""
            if filename.lower() == att_lower or filename.lower().endswith(att_lower.lstrip("*")):
                payload = part.get_payload(decode=True)
                if isinstance(payload, bytes) and payload:
                    return payload

        return None
    finally:
        try:
            mail.close()
            mail.logout()
        except Exception:
            pass


# ── query helpers ───────────────────────────────────────────────────────────

def imap_since_date(iso_date: str) -> str:
    """Convert an ISO-8601 date string to IMAP ``SINCE`` criterion format.

    IMAP expects ``DD-Mon-YYYY`` (e.g. ``01-Jan-2025``).

    Args:
        iso_date: Date in ``YYYY-MM-DD`` format.

    Returns:
        Date string in IMAP ``SINCE`` criterion format.
    """
    d = date.fromisoformat(iso_date)
    return d.strftime("%d-%b-%Y")


def imap_before_date(iso_date: str) -> str:
    """Convert an ISO-8601 date string to IMAP ``BEFORE`` criterion format.

    IMAP expects ``DD-Mon-YYYY`` (e.g. ``01-Jan-2025``).

    Args:
        iso_date: Date in ``YYYY-MM-DD`` format.

    Returns:
        Date string in IMAP ``BEFORE`` criterion format.
    """
    d = date.fromisoformat(iso_date)
    return d.strftime("%d-%b-%Y")
