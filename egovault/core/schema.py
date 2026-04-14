"""NormalizedRecord — the single data schema all adapters must produce."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum

from egovault.utils.hashing import compute_record_id


class EnrichmentStatus(IntEnum):
    PENDING = 0
    DONE = 1
    FAILED = 2
    SKIPPED = 3

VALID_RECORD_TYPES: frozenset[str] = frozenset(
    {
        "message",
        "link",
        "event",
        "media_ref",
        "document",
        "note",
        "image",
        "spreadsheet",
        "code",
    }
)


@dataclass
class NormalizedRecord:
    platform: str
    record_type: str
    timestamp: datetime
    sender_id: str
    sender_name: str
    thread_id: str
    thread_name: str
    body: str
    attachments: list[str] = field(default_factory=list)
    raw: dict = field(default_factory=dict)
    file_path: str | None = None
    mime_type: str | None = None
    # computed in __post_init__, not accepted as a constructor arg
    id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.record_type not in VALID_RECORD_TYPES:
            raise ValueError(
                f"Invalid record_type {self.record_type!r}. "
                f"Must be one of: {sorted(VALID_RECORD_TYPES)}"
            )
        self.id = compute_record_id(
            platform=self.platform,
            thread_id=self.thread_id,
            timestamp=self.timestamp,
            sender_id=self.sender_id,
            body=self.body,
            file_path=self.file_path,
        )
