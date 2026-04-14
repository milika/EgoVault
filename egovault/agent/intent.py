"""Agent intent detection shim.

During the Cycle-7.5 migration the canonical code lives in
`egovault.chat.session`.  This module re-exports the intent-detection
symbols so that `from egovault.agent.intent import _resolve_intent` works.

When the migration completes (Phase 4), the code will move here and
`chat/session.py` will only re-export from here.
"""
# ruff: noqa: F401
from egovault.chat.session import (
    _BANNER,
    _HELP,
    _INTENT_MAP,
    _SCHEDULE_TIME_RE,
    _resolve_intent,
    _resolve_schedule_intent,
)
