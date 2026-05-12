from __future__ import annotations

from typing import Any, Callable, Type, cast

from training.registry import available, get, register


def register_task(name: str) -> Callable[[Type[Any]], Type[Any]]:
    """Register a task class by name."""

    def decorator(task_cls: Type[Any]) -> Type[Any]:
        return cast(Type[Any], register("tasks", name)(task_cls))

    return decorator


def get_task_class(name: str) -> Type[Any]:
    """Fetch a registered task class by name."""
    return cast(Type[Any], get("tasks", name))


def available_tasks() -> list[str]:
    """List all registered task names."""
    return available("tasks")
