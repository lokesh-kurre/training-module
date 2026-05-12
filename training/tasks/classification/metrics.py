"""Classification metrics."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute classification accuracy.

    Args:
        logits: model outputs (batch_size, num_classes)
        targets: ground truth labels (batch_size,)

    Returns:
        accuracy as float [0, 1]
    """
    preds = torch.argmax(logits, dim=1)
    return float(torch.mean((preds == targets).float()))


def compute_metrics(logits: torch.Tensor, targets: torch.Tensor, cfg: dict) -> dict[str, float]:
    """Compute all configured metrics.

    Args:
        logits: model outputs
        targets: ground truth labels
        cfg: config dict with metrics list

    Returns:
        dict of metric_name -> value
    """
    task_cfg = cfg.get("task", {})
    metrics_list = task_cfg.get("metrics", ["accuracy"])

    result = {}
    for metric_name in metrics_list:
        normalized = str(metric_name).lower()
        if normalized in {"accuracy", "categoricalaccuracy", "categorical_accuracy"}:
            result["accuracy"] = compute_accuracy(logits, targets)
        else:
            raise ValueError(f"Unknown metric: {metric_name}")

    return result
