"""Agent tool executor shim — imports from chat.session during Cycle-7.5 migration."""
# ruff: noqa: F401
from egovault.chat.session import (
    _open_with_default_app,
    _resolve_write_target_path,
    _build_file_export,
    _expand_abbreviation,
    _rank_search_results,
    _extract_matching_lines,
    _list_windows_cross_platform,
    _enumerate_window_elements,
    _execute_tool,
    _answer_needs_retry,
)
