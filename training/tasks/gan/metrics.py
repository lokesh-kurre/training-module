"""GAN metrics."""

from __future__ import annotations

from typing import Any

import torch


def compute_metrics(outputs: dict[str, Any], _cfg: dict[str, Any]) -> dict[str, float]:
    d_real_logits = outputs["d_real_logits"]
    d_fake_logits = outputs["d_fake_logits"]
    d_recon_logits = outputs["d_recon_logits"]

    return {
        "d_real_score": float(torch.sigmoid(d_real_logits).mean().item()),
        "d_fake_score": float(torch.sigmoid(d_fake_logits).mean().item()),
        "d_recon_score": float(torch.sigmoid(d_recon_logits).mean().item()),
        "d_acc_real": float((torch.sigmoid(d_real_logits) > 0.5).float().mean().item()),
        "d_acc_fake": float((torch.sigmoid(d_fake_logits) < 0.5).float().mean().item()),
        "g_adv_loss": float(outputs["g_adv_loss"].item()),
        "d_loss": float(outputs["d_loss"].item()),
        "recon_l1": float(outputs["recon_l1"].item()),
    }
