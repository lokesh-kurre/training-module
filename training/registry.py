from collections import defaultdict
from typing import Any, Callable, Dict


RegistryStore = Dict[str, Dict[str, Callable[..., Any]]]
_REGISTRIES: RegistryStore = defaultdict(dict)


def register(kind: str, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a callable under a registry kind and name."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        if name in _REGISTRIES[kind]:
            raise ValueError(f"Duplicate registration for {kind}:{name}")
        _REGISTRIES[kind][name] = fn
        return fn

    return decorator


def get(kind: str, name: str) -> Callable[..., Any]:
    """Fetch a registered callable by kind and name."""
    try:
        return _REGISTRIES[kind][name]
    except KeyError as exc:
        available = ", ".join(sorted(_REGISTRIES.get(kind, {}).keys()))
        raise KeyError(f"Unknown {kind}:{name}. Available: [{available}]") from exc


def available(kind: str) -> list[str]:
    """List all registered names for a registry kind."""
    return sorted(_REGISTRIES.get(kind, {}).keys())
