from __future__ import annotations

from typing import Any

from training.callbacks.base import Callback
from training.callbacks.defaults import CheckpointCallback, GANSampleGridCallback, MetricsCallback
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
            metrics_cfg = callback_cfg.get("metrics", {})
            if not isinstance(metrics_cfg, dict):
                metrics_cfg = {}
            callbacks.append(
                MetricsCallback(
                    write_json=bool(metrics_cfg.get("write_json", True)),
                    write_csv=bool(metrics_cfg.get("write_csv", True)),
                    json_filename=str(metrics_cfg.get("json_filename", "metrics.json")),
                    csv_filename=str(metrics_cfg.get("csv_filename", "metrics.csv")),
                )
            )
        elif name == "gan_samples":
            gan_cfg = callback_cfg.get("gan_samples", {})
            if not isinstance(gan_cfg, dict):
                gan_cfg = {}
            callbacks.append(
                GANSampleGridCallback(
                    save_freq=int(gan_cfg.get("save_freq", callback_cfg.get("save_freq", 1))),
                    num_samples=int(gan_cfg.get("num_samples", 16)),
                    nrow=int(gan_cfg.get("nrow", 4)),
                    normalize=bool(gan_cfg.get("normalize", True)),
                    with_headers=bool(gan_cfg.get("with_headers", False)),
                    row_header_prefix=str(gan_cfg.get("row_header_prefix", "Style")),
                    col_header_prefix=str(gan_cfg.get("col_header_prefix", "Content")),
                )
            )
        elif isinstance(name, str) and "." in name:
            cls = get_obj_by_name(name)
            callbacks.append(cls())
        else:
            raise ValueError(f"Unknown callback '{name}'")

    return callbacks
