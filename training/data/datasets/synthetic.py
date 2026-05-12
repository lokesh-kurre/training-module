from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import TensorDataset

from training.utils.input_spec import out_size_to_chw, resolve_input_spec


def build_synthetic_dataset(cfg: dict[str, Any], split: str = "train", num_samples: int = 100) -> TensorDataset:
    """Build a synthetic image dataset used for smoke tests and skeleton loops."""
    _ = split
    model_cfg = cfg.get("model", {})
    num_classes = int(model_cfg.get("num_classes", 2))
    out_size, layout = resolve_input_spec(cfg)
    channels, h, w = out_size_to_chw(out_size, layout)
    if layout == "HWC":
        x = torch.randn(num_samples, h, w, channels)
    else:
        x = torch.randn(num_samples, channels, h, w)

    y = torch.randint(0, num_classes, (num_samples,))
    return TensorDataset(x, y)
