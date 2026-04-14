"""BasePlatformAdapter — ABC that all platform adapters must implement."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

from egovault.core.schema import NormalizedRecord


class BasePlatformAdapter(ABC):
    """All platform adapters must implement this interface."""

    @property
    @abstractmethod
    def platform_id(self) -> str:
        """Short lowercase identifier, e.g. 'facebook'."""
        ...

    @classmethod
    @abstractmethod
    def can_handle(cls, source_path: Path) -> bool:
        """Return True if this adapter recognises the given export path/file."""
        ...

    @abstractmethod
    def ingest(self, source_path: Path) -> Iterator[NormalizedRecord]:
        """Parse the export and yield normalised records one at a time."""
        ...
