"""Optimizer builders."""

from __future__ import annotations

from typing import Any

import torch.nn as nn
import torch.optim as optim


def build_optimizer(model: nn.Module, cfg: dict[str, Any]) -> optim.Optimizer:
    """Build optimizer from config.

    Supported optimizers: adam, sgd, adamw

    Args:
        model: PyTorch model
        cfg: config dict with optimizer settings

    Returns:
        PyTorch optimizer instance
    """
    optimizer_cfg = cfg.get("optimizer", {})
    optimizer_name = optimizer_cfg.get("name", "adam").lower()
    lr = float(optimizer_cfg.get("lr", 0.001))
    weight_decay = float(optimizer_cfg.get("weight_decay", 0.0))

    if optimizer_name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        momentum = float(optimizer_cfg.get("momentum", 0.9))
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def build_scheduler(optimizer: optim.Optimizer, cfg: dict[str, Any]) -> None | optim.lr_scheduler.LRScheduler:
    """Build learning rate scheduler from config (optional).

    Args:
        optimizer: PyTorch optimizer
        cfg: config dict with scheduler settings

    Returns:
        scheduler or None if not configured
    """
    scheduler_cfg = cfg.get("scheduler", {})
    if scheduler_cfg is None:
        scheduler_cfg = {}

    scheduler_name = scheduler_cfg.get("name")

    # If scheduler is omitted, default to cosine schedule:
    # lr starts from optimizer lr (default 1e-3) and anneals to 1e-7 over 50 epochs.
    if not scheduler_name:
        scheduler_name = "cosine"
        scheduler_cfg = {
            "t_max": 50,
            "eta_min": 1e-7,
            **scheduler_cfg,
        }

    if scheduler_name == "cosine":
        t_max = int(scheduler_cfg.get("t_max", 50))
        eta_min = float(scheduler_cfg.get("eta_min", 1e-7))
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
    elif scheduler_name == "step":
        step_size = int(scheduler_cfg.get("step_size", 10))
        gamma = float(scheduler_cfg.get("gamma", 0.1))
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name == "constant":
        factor = float(scheduler_cfg.get("factor", 1.0))
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _epoch: factor)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
