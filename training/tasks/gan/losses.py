"""Loss builders for GAN task."""

from __future__ import annotations

from typing import Any

import torch.nn as nn


def build_loss(cfg: dict[str, Any]) -> dict[str, Any]:
    model_cfg = cfg.get("model", {})
    gan_cfg = model_cfg.get("gan", {}) if isinstance(model_cfg, dict) else {}

    return {
        "adv": nn.BCEWithLogitsLoss(),
        "recon": nn.L1Loss(),
        "weight_d": float(gan_cfg.get("weight_d", 1.0)),
        "weight_g": float(gan_cfg.get("weight_g", 1.0)),
        "weight_recon": float(gan_cfg.get("weight_recon", 10.0)),
    }
