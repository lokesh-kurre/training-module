from __future__ import annotations

from typing import Any


def build_sampler(dataset: Any, split: str, runtime: Any) -> Any:
    """Build split-aware samplers with DDP support."""
    if not getattr(runtime, "is_distributed", False):
        return None

    from torch.utils.data import DistributedSampler

    shuffle = split == "train"
    return DistributedSampler(
        dataset,
        num_replicas=runtime.world_size,
        rank=runtime.rank,
        shuffle=shuffle,
        drop_last=(split == "train"),
    )
