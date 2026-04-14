"""Encoding utilities — Meta platform mojibake fix."""
from __future__ import annotations


def fix_mojibake(text: str) -> str:
    """Fix Meta (Facebook/Instagram) mojibake.

    Meta's export encodes UTF-8 text as latin-1 byte values, then stores them
    as a string.  Re-encoding as latin-1 bytes and decoding as UTF-8 recovers
    the original characters.  Non-mojibake strings are returned unchanged.
    """
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return text
