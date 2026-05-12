from __future__ import annotations

from typing import Any

from torch.utils.data import DataLoader, IterableDataset

from training.data.samplers.builder import build_sampler


def build_dataloader(cfg: dict[str, Any], dataset: Any, split: str, runtime: Any) -> DataLoader:
    dataloader_cfg = cfg.get("dataloader", {})
    backend = str(dataloader_cfg.get("backend", "torch")).lower()
    batch_size = int(dataloader_cfg.get("batch_size", 32))
    num_workers = int(dataloader_cfg.get("num_workers", 0))
    pin_memory = bool(dataloader_cfg.get("pin_memory", str(getattr(runtime, "device", "cpu")).startswith("cuda")))
    persistent_workers = bool(dataloader_cfg.get("persistent_workers", num_workers > 0))
    prefetch_factor_raw = dataloader_cfg.get("prefetch_factor", None)

    if backend not in {"torch", "data_gen"}:
        raise ValueError(f"Unsupported dataloader backend '{backend}'. Expected 'torch' or 'data_gen'.")

    if backend == "data_gen":
        if not isinstance(dataset, IterableDataset):
            raise TypeError("data_gen backend requires an IterableDataset")
        if not getattr(dataset, "yields_batched", False):
            raise ValueError("data_gen backend expects dataset batches to be pre-batched")
        return DataLoader(
            dataset,
            batch_size=None,
            num_workers=0,
            pin_memory=pin_memory,
        )

    sampler = build_sampler(dataset, split=split, runtime=runtime)
    shuffle = split == "train" and sampler is None

    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers and num_workers > 0,
    }
    if num_workers > 0 and prefetch_factor_raw is not None:
        loader_kwargs["prefetch_factor"] = int(prefetch_factor_raw)

    return DataLoader(**loader_kwargs)
