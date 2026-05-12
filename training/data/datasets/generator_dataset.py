from __future__ import annotations

from functools import partial
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import IterableDataset

from training.data.generator_backend import get_generator
from training.utils.importer import get_obj_by_name
from training.utils.input_spec import resolve_input_spec


def _looks_like_record(value: Any) -> bool:
    if isinstance(value, dict):
        return "filepath" in value and "class" in value
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return not isinstance(value[0], (list, tuple, dict))
    return False


def _to_records(data_ref: Any) -> list[Any]:
    if data_ref is None:
        return []

    try:
        import pandas as pd  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        pd = None
    dataframe_type = (pd.DataFrame,) if pd is not None else tuple()

    if _looks_like_record(data_ref):
        return [data_ref]

    if isinstance(data_ref, list):
        records: list[Any] = []
        for item in data_ref:
            if _looks_like_record(item):
                records.append(item)
            elif isinstance(item, (list, tuple, str)) or (dataframe_type and isinstance(item, dataframe_type)):
                records.extend(_to_records(item))
            else:
                records.append(item)
        return records

    if dataframe_type and isinstance(data_ref, dataframe_type):
        return data_ref.values.tolist()

    if isinstance(data_ref, str):
        path = Path(data_ref)
        if not path.exists():
            raise FileNotFoundError(f"Generator data path does not exist: {data_ref}")
        if path.suffix == ".feather":
            if pd is None:
                raise ModuleNotFoundError("pandas is required to read .feather files")
            return pd.read_feather(path).values.tolist()
        if path.suffix == ".csv":
            if pd is None:
                raise ModuleNotFoundError("pandas is required to read .csv files")
            return pd.read_csv(path).values.tolist()
        raise ValueError("Unsupported data path format for generator dataset. Use .csv or .feather")

    raise TypeError(f"Unsupported generator data reference type: {type(data_ref)}")


def _extract_label(record: Any) -> Any:
    if isinstance(record, dict):
        if "class" not in record:
            raise ValueError("record dict must contain 'class' key")
        return record["class"]
    if isinstance(record, (list, tuple)) and len(record) >= 2:
        return record[1]
    raise ValueError("record must be dict or list/tuple with at least [filepath, class]")


def _apply_label_map(record: Any, label_to_index: dict[str, int]) -> Any:
    label = _extract_label(record)
    key = str(label)
    if key not in label_to_index:
        raise ValueError(f"label '{label}' is not present in training-derived label map")
    mapped = int(label_to_index[key])

    if isinstance(record, dict):
        out = dict(record)
        out["class"] = mapped
        return out

    if isinstance(record, tuple):
        out = list(record)
        out[1] = mapped
        return tuple(out)

    out = list(record)
    out[1] = mapped
    return out


def _get_split_data_ref(dataset_cfg: dict[str, Any], generator_cfg: dict[str, Any], split: str) -> Any:
    data_by_split = generator_cfg.get("data", {})
    split_data = data_by_split.get(split)
    if split_data is None:
        split_data = dataset_cfg.get("splits", {}).get(split)
    if split_data is None:
        split_data = dataset_cfg.get(f"{split}_path")
    return split_data


def _build_label_map_from_train(dataset_cfg: dict[str, Any], generator_cfg: dict[str, Any]) -> dict[str, int]:
    train_ref = _get_split_data_ref(dataset_cfg, generator_cfg, "train")
    train_records = _to_records(train_ref)
    if not train_records:
        raise ValueError("No training records available to derive dynamic label map")

    labels: list[str] = []
    seen: set[str] = set()
    for rec in train_records:
        key = str(_extract_label(rec))
        if key not in seen:
            labels.append(key)
            seen.add(key)

    return {label: idx for idx, label in enumerate(labels)}


class GeneratorIterableDataset(IterableDataset):
    """
    IterableDataset that yields pre-batched tensors from shared-memory workers.

    input_dtype must match the output of read_func. For example:
      - If read_func returns only image: input_dtype = "(3,224,224)f4"
      - If read_func returns (image, label): input_dtype = "(3,224,224)f4,(1,)i8"
    """

    yields_batched = True

    def __init__(
        self,
        records: list[Any],
        read_func: Any,
        input_dtype: str,
        batch_size: int,
        no_of_workers: int,
        no_of_worker_threads: int,
        qsize: int,
        take: int | None,
        infinite: bool,
        shuffle: bool,
    ) -> None:
        super().__init__()
        self.num_records = len(records)
        self.batch_size = max(1, int(batch_size))
        self.take = None if take is None else int(take)
        self.infinite = bool(infinite)
        self._generator = get_generator(
            data=records,
            read_func=read_func,
            input_dtype=input_dtype,
            batch_size=batch_size,
            no_of_workers=no_of_workers,
            no_of_worker_threads=no_of_worker_threads,
            qsize=qsize,
            take=take,
            infinite=infinite,
            shuffle=shuffle,
        )

    def __iter__(self):
        for batch in self._generator():
            if isinstance(batch, tuple):
                yield tuple(torch.from_numpy(np.asarray(item)) for item in batch)
            else:
                yield torch.from_numpy(np.asarray(batch))

    def __len__(self) -> int:
        # Prefer explicit take (number of yielded batches) when configured.
        if self.take is not None:
            return max(0, self.take)
        if self.infinite:
            raise TypeError("GeneratorIterableDataset has infinite length when infinite=True and take is unset")
        return max(1, int(math.ceil(self.num_records / self.batch_size)))

def build_generator_dataset_from_config(
    cfg: dict[str, Any],
    split: str = "train",
    rank: int = 0,
    world_size: int = 1,
) -> GeneratorIterableDataset:
    dataset_cfg = cfg.get("dataset", {})
    generator_cfg = dataset_cfg.get("generator", {})
    if not generator_cfg:
        generator_cfg = cfg.get("dataset_generator", {})

    split_data = _get_split_data_ref(dataset_cfg, generator_cfg, split)

    records = _to_records(split_data)
    if not records:
        raise ValueError(
            f"No generator records configured for split='{split}'. "
            "Set dataset.generator.data.<split> or dataset.splits.<split> (or legacy dataset.<split>_path)."
        )

    label_to_index = _build_label_map_from_train(dataset_cfg, generator_cfg)
    records = [_apply_label_map(record, label_to_index) for record in records]

    if int(world_size) > 1:
        safe_rank = max(0, int(rank))
        safe_world_size = max(1, int(world_size))
        records = records[safe_rank::safe_world_size]
        if not records:
            raise ValueError(
                f"No generator records left after sharding split='{split}' for rank={safe_rank}, world_size={safe_world_size}"
            )

    read_func_path = generator_cfg.get("read_func")
    if not isinstance(read_func_path, str) or not read_func_path:
        raise ValueError("dataset.generator.read_func must be a fully-qualified callable path")
    read_func_raw = get_obj_by_name(read_func_path)
    out_size, layout = resolve_input_spec(cfg)
    s3_client = generator_cfg.get("s3_client")
    read_func = partial(
        read_func_raw,
        out_size=out_size,
        layout=layout,
        split=split,
        label_to_index=label_to_index,
        s3_client=s3_client,
        cfg=cfg,
    )

    # Validate input_dtype matches read_func output
    input_dtype = str(generator_cfg.get("input_dtype", "(224,224,3)f4"))
    # Try to infer if read_func returns tuple (image, label)
    test_sample = records[0]
    test_out = read_func(test_sample)
    if isinstance(test_out, tuple):
        if len(test_out) == 2:
            # Expect input_dtype to be tuple dtype
            if "," not in input_dtype:
                raise ValueError(
                    f"read_func returns (image, label) but input_dtype ('{input_dtype}') is not a tuple dtype. "
                    "Use e.g. '(3,224,224)f4,(1,)i8' for (image, label)."
                )
        else:
            raise ValueError("read_func returns tuple of unexpected length (expected 2: (image, label))")
    return GeneratorIterableDataset(
        records=records,
        read_func=read_func,
        input_dtype=input_dtype,
        batch_size=int(generator_cfg.get("batch_size", cfg.get("dataloader", {}).get("batch_size", 32))),
        no_of_workers=int(generator_cfg.get("no_of_workers", cfg.get("dataloader", {}).get("num_workers", 4))),
        no_of_worker_threads=int(generator_cfg.get("no_of_worker_threads", 8)),
        qsize=int(generator_cfg.get("qsize", 8)),
        take=generator_cfg.get("take"),
        infinite=bool(generator_cfg.get("infinite", split == "train")),
        shuffle=bool(generator_cfg.get("shuffle", split == "train")),
    )
