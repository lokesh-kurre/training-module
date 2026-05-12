"""Loss functions for classification."""

from __future__ import annotations

from typing import Any

import torch.nn as nn


def build_loss(cfg: dict[str, Any]) -> nn.Module:
    """Build classification loss function from config.

    Supported losses: cross_entropy, bce, bce_with_logits

    Args:
        cfg: config dict with loss settings

    Returns:
        PyTorch loss module
    """
    loss_cfg = cfg.get("loss", {})
    loss_name = loss_cfg.get("name", "cross_entropy").lower()

    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif loss_name == "bce":
        return nn.BCELoss()
    elif loss_name == "bce_with_logits":
        return nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")
