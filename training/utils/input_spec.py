from __future__ import annotations

from typing import Any


def resolve_input_spec(cfg: dict[str, Any]) -> tuple[tuple[int, ...], str]:
    """Resolve model/data input size and layout from config.

    Preferred config:
        input:
            size: [C, H, W] for CHW layout, or [H, W, C] for HWC layout
      layout: CHW|HWC

    Backward-compatible fallback:
    dataset.input_size
    """
    input_cfg = cfg.get("input", {}) if isinstance(cfg, dict) else {}

    raw_size = input_cfg.get("size")
    if raw_size is None:
        raw_size = cfg.get("dataset", {}).get("input_size", [224, 224])

    if not isinstance(raw_size, (list, tuple)) or len(raw_size) < 2:
        raise ValueError("input.size must be a list/tuple with at least 2 dims")

    layout = str(input_cfg.get("layout", "CHW")).upper()
    if layout not in {"CHW", "HWC"}:
        raise ValueError(f"input.layout must be 'CHW' or 'HWC', got '{layout}'")

    if len(raw_size) == 2:
        # Backward compatible: infer channels if only spatial size is provided.
        channels = int(cfg.get("model", {}).get("in_channels", 3))
        if channels <= 0:
            raise ValueError("model.in_channels must be a positive integer")
        h, w = int(raw_size[0]), int(raw_size[1])
        out_size = (channels, h, w) if layout == "CHW" else (h, w, channels)
    elif len(raw_size) == 3:
        out_size = tuple(int(v) for v in raw_size)
    else:
        raise ValueError("input.size must have 2 or 3 dims")

    return out_size, layout


def out_size_to_chw(out_size: tuple[int, ...], layout: str) -> tuple[int, int, int]:
    """Convert layout-aware out_size to (C, H, W)."""
    if len(out_size) != 3:
        raise ValueError("out_size must have exactly 3 dims when converting to CHW")
    if layout == "CHW":
        c, h, w = out_size
    elif layout == "HWC":
        h, w, c = out_size
    else:
        raise ValueError(f"Unsupported layout '{layout}'")
    return int(c), int(h), int(w)
