"""AdapterRegistry — tracks all registered adapters and selects one by path."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from egovault.core.adapter import BasePlatformAdapter

# Populated by the @register decorator defined below.
_registry: list[type["BasePlatformAdapter"]] = []
_adapters_loaded: bool = False


def register(cls: type["BasePlatformAdapter"]) -> type["BasePlatformAdapter"]:
    """Class decorator — adds the adapter to the global registry."""
    _registry.append(cls)
    return cls


def _load_adapters() -> None:
    """Import egovault.adapters, which auto-imports every adapter module."""
    global _adapters_loaded
    if _adapters_loaded:
        return
    import egovault.adapters  # noqa: F401  — side-effect: runs __init__.py
    _adapters_loaded = True


class AdapterRegistry:
    @classmethod
    def get_adapter(cls, source_path: Path, **kwargs: object) -> "BasePlatformAdapter":
        """Return the first registered adapter that can handle *source_path*.

        Extra *kwargs* are forwarded to the adapter's constructor (e.g. store=…).
        Raises ValueError if no adapter matches.
        """
        _load_adapters()
        for adapter_cls in _registry:
            if adapter_cls.can_handle(source_path):
                return adapter_cls(**kwargs)
        raise ValueError(
            f"No adapter found for path: {source_path}\n"
            f"Registered adapters: {[c.__name__ for c in _registry]}"
        )

    @classmethod
    def all_adapters(cls) -> list[type["BasePlatformAdapter"]]:
        """Return all registered adapter classes."""
        _load_adapters()
        return list(_registry)
