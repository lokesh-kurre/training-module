"""Task package auto-discovery.

Any subpackage under training/tasks that imports and registers a task class
will be auto-imported here, so no manual edits are needed when adding tasks.
"""

from __future__ import annotations

import importlib
import pkgutil
from types import ModuleType


def _import_task_modules() -> list[ModuleType]:
    imported: list[ModuleType] = []
    for module_info in pkgutil.iter_modules(__path__):
        module_name = module_info.name
        if module_name.startswith("_") or module_name == "base" or module_name == "registry":
            continue

        # Import package-level module first (supports explicit package exports).
        package_module = importlib.import_module(f"{__name__}.{module_name}")
        imported.append(package_module)

        # Also import optional `<task>.task` module to ensure registration side effects
        # happen even when the package __init__.py does not import task symbols.
        try:
            task_module = importlib.import_module(f"{__name__}.{module_name}.task")
            imported.append(task_module)
        except ModuleNotFoundError:
            # Some task packages may register from __init__.py or alternate module layout.
            pass
    return imported


_TASK_MODULES = _import_task_modules()
__all__ = [module.__name__.split(".")[-1] for module in _TASK_MODULES]

