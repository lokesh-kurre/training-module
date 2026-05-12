from __future__ import annotations

from typing import Any

from torch.utils.data import ConcatDataset, Dataset

from training.data.datasets.generator_dataset import build_generator_dataset_from_config
from training.data.datasets.synthetic import build_synthetic_dataset
from training.utils.importer import get_obj_by_name


def build_dataset_from_config(
    cfg: dict[str, Any],
    split: str = "train",
    rank: int = 0,
    world_size: int = 1,
    runtime: Any = None,
) -> Dataset:
    _ = runtime
    dataset_cfg = cfg.get("dataset", {})
    dataset_name = dataset_cfg.get("name", "synthetic")

    dataset_mix = dataset_cfg.get("mix", dataset_cfg.get("datasets"))
    if isinstance(dataset_mix, str):
        dataset_mix = [dataset_mix]

    # Backward compatibility: if dataset.names is a list[str], treat it as dataset mix.
    if dataset_mix is None:
        names_value = dataset_cfg.get("names")
        if isinstance(names_value, list) and all(isinstance(item, str) for item in names_value):
            dataset_mix = names_value

    dataset_names = dataset_mix if isinstance(dataset_mix, list) and dataset_mix else [dataset_name]

    dataloader_backend = str(cfg.get("dataloader", {}).get("backend", "torch")).lower()
    dataset_backend = str(dataset_cfg.get("backend", "torch")).lower()

    if dataset_backend == "data_gen" or dataloader_backend == "data_gen":
        return build_generator_dataset_from_config(cfg, split=split, rank=rank, world_size=world_size)

    # Optional configurable torch Dataset class/callable.
    # Expected path points to a class or function that returns a torch Dataset.
    torch_dataset_class = dataset_cfg.get("torch_dataset_class", dataset_cfg.get("class"))
    if isinstance(torch_dataset_class, str) and torch_dataset_class:
        dataset_ctor = get_obj_by_name(torch_dataset_class)
        created = dataset_ctor(
            cfg,
            split=split,
            rank=rank,
            world_size=world_size,
            runtime=runtime,
        )
        if not isinstance(created, Dataset):
            raise TypeError(
                f"Configured dataset class '{torch_dataset_class}' returned {type(created)}, expected torch.utils.data.Dataset"
            )
        return created

    datasets: list[Dataset] = []
    for name in dataset_names:
        _ = name
        datasets.append(build_synthetic_dataset(cfg, split=split, num_samples=100))

    if len(datasets) == 1:
        return datasets[0]

    return ConcatDataset(datasets)
