from __future__ import annotations

from typing import Any

from training.callbacks.base import Callback
from training.callbacks.defaults import CheckpointCallback, MetricsCallback
from training.utils.importer import get_obj_by_name


def build_callbacks(cfg: dict[str, Any]) -> list[Callback]:
    callback_cfg = cfg.get("callbacks", {})
    callback_names = callback_cfg.get("enabled", ["checkpoint", "metrics"])

    callbacks: list[Callback] = []
    for name in callback_names:
        if name == "checkpoint":
            callbacks.append(
                CheckpointCallback(
                    monitor=str(callback_cfg.get("monitor", "val_loss")),
                    mode=str(callback_cfg.get("mode", "min")),
                    save_freq=int(callback_cfg.get("save_freq", 1)),
                )
            )
        elif name == "metrics":
            callbacks.append(MetricsCallback())
        elif isinstance(name, str) and "." in name:
            cls = get_obj_by_name(name)
            callbacks.append(cls())
        else:
            raise ValueError(f"Unknown callback '{name}'")

    return callbacks
