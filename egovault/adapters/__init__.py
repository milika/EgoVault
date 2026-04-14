"""Auto-imports all adapter modules so their @register decorators fire."""
from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

_adapters_dir = str(Path(__file__).parent)

for _mod in pkgutil.iter_modules([_adapters_dir]):
    importlib.import_module(f"egovault.adapters.{_mod.name}")
