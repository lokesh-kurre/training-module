from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError as exc:
    raise RuntimeError("PyYAML is required. Install dependencies from requirements.txt") from exc


ConfigDict = dict[str, Any]

FLAT_KEY_MAP: dict[str, str] = {
    "task": "project.task",
    "name": "run.name",
    "experiment": "project.experiment",
    "epochs": "trainer.epochs",
    "device": "trainer.device",
    "mixed_precision": "trainer.mixed_precision",
    "amp": "trainer.mixed_precision",
    "batch": "dataloader.batch_size",
    "batch_size": "dataloader.batch_size",
    "imgsz": "input.size",
    "workers": "dataloader.num_workers",
    "num_workers": "dataloader.num_workers",
    "input_size": "input.size",
    "layout": "input.layout",
    "dataset_names": "dataset.mix",
    "model_name": "model.name",
    "architecture": "model.architecture",
    "num_classes": "model.num_classes",
    "pretrained": "model.pretrained",
    "architectures_supported": "model.architectures_supported",
    "loss": "loss.name",
    "metrics": "task.metrics",
    "model_builder": "trainer.model_builder",
    "dataset_builder": "trainer.dataset_builder",
    "dataloader_builder": "trainer.dataloader_builder",
    "loss_builder": "trainer.loss_builder",
    "optimizer_builder": "trainer.optimizer_builder",
    "scheduler_builder": "trainer.scheduler_builder",
    "metrics_fn": "trainer.metrics_fn",
    "callback_monitor": "callbacks.monitor",
    "callback_mode": "callbacks.mode",
    "lr": "optimizer.lr",
    "lr0": "optimizer.lr",
    "weight_decay": "optimizer.weight_decay",
    "momentum": "optimizer.momentum",
    "distributed": "distributed.enabled",
    "distributed_backend": "distributed.backend",
    "find_unused_parameters": "distributed.find_unused_parameters",
    "broadcast_buffers": "distributed.broadcast_buffers",
    "sync_batchnorm": "distributed.sync_batchnorm",
    "resume": "resume.enabled",
    "resume_path": "resume.checkpoint_path",
    "resume_strict": "resume.strict",
    "weights": "model.weights_path",
    "weights_path": "model.weights_path",
}


def _is_uri(value: str) -> bool:
    return "://" in value


def _resolve_data_ref(value: Any, base_dir: Path) -> Any:
    if value is None:
        return None

    if isinstance(value, list):
        return [_resolve_data_ref(item, base_dir) for item in value]

    if isinstance(value, str):
        if _is_uri(value):
            return value
        p = Path(value)
        if p.is_absolute():
            return str(p)
        return str((base_dir / p).resolve())

    return value


def _is_yaml_path(value: str) -> bool:
    suffix = Path(value).suffix.lower()
    return suffix in {".yaml", ".yml"}


def _dataset_name_from_ref(value: str) -> str:
    if _is_yaml_path(value):
        return Path(value).stem
    return value


def _resolve_dataset_yaml_path(dataset_ref: str, root_dir: Path, cfg_dir: Path) -> Path | None:
    ref = dataset_ref.strip()
    if not ref:
        return None

    if _is_yaml_path(ref):
        path = Path(ref)
        if path.is_absolute():
            return path if path.exists() else None
        candidate = (root_dir / path).resolve()
        return candidate if candidate.exists() else None

    candidate = cfg_dir / "datasets" / f"{ref}.yaml"
    return candidate if candidate.exists() else None


def _last_cli_override(cli_overrides: list[str], key: str) -> Any | None:
    last_value: Any | None = None
    for item in cli_overrides:
        if "=" not in item:
            continue
        k, raw_value = item.split("=", 1)
        if k.strip() == key:
            last_value = yaml.safe_load(raw_value)
    return last_value


def _load_dataset_cfg_from_ref(dataset_ref: str, root_dir: Path, cfg_dir: Path) -> ConfigDict:
    dataset_yaml = _resolve_dataset_yaml_path(dataset_ref, root_dir=root_dir, cfg_dir=cfg_dir)
    if dataset_yaml is None:
        return {}
    dataset_cfg = load_yaml(dataset_yaml)
    return _normalize_yolo_dataset_schema(dataset_cfg, dataset_yaml)


def _load_dataset_cfg_from_refs(dataset_ref: Any, root_dir: Path, cfg_dir: Path) -> ConfigDict:
    """Load one or more dataset refs (name or yaml path) into a merged config."""
    if isinstance(dataset_ref, str):
        return _load_dataset_cfg_from_ref(dataset_ref, root_dir=root_dir, cfg_dir=cfg_dir)

    if not isinstance(dataset_ref, list):
        return {}

    refs = [item for item in dataset_ref if isinstance(item, str) and item.strip()]
    if not refs:
        return {}

    merged: ConfigDict = {}
    for ref in refs:
        cfg_part = _load_dataset_cfg_from_ref(ref, root_dir=root_dir, cfg_dir=cfg_dir)
        if cfg_part:
            merged = deep_merge(merged, cfg_part)

    dataset_section = merged.get("dataset", {})
    if not isinstance(dataset_section, dict):
        dataset_section = {}

    # Preserve all provided refs for downstream builders that support mixed datasets.
    ref_names = [_dataset_name_from_ref(ref) for ref in refs]
    if ref_names:
        dataset_section.setdefault("name", ref_names[0])
        dataset_section["mix"] = ref_names
        dataset_section["yaml"] = refs
        merged["dataset"] = dataset_section

    return merged


def _normalize_yolo_dataset_schema(cfg: ConfigDict, cfg_path: Path) -> ConfigDict:
    """Convert YOLO-style dataset YAML into internal nested config shape.

    Supported input keys:
    - path
    - train
    - val
    - test
    - names
    """
    if not isinstance(cfg, dict):
        return cfg

    if not any(key in cfg for key in ("train", "val", "test")):
        return cfg

    normalized = deepcopy(cfg)
    yaml_dir = cfg_path.parent

    path_root = normalized.get("path", ".")
    if isinstance(path_root, str) and not _is_uri(path_root):
        root_dir = Path(path_root)
        if not root_dir.is_absolute():
            root_dir = (yaml_dir / root_dir).resolve()
    else:
        root_dir = yaml_dir.resolve()

    splits: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        if split in normalized:
            splits[split] = _resolve_data_ref(normalized.get(split), root_dir)

    dataset_cfg = normalized.get("dataset", {})
    if not isinstance(dataset_cfg, dict):
        dataset_cfg = {}
    dataset_cfg.setdefault("name", cfg_path.stem)

    existing_splits = dataset_cfg.get("splits", {})
    if not isinstance(existing_splits, dict):
        existing_splits = {}
    existing_splits.update({k: v for k, v in splits.items() if v is not None})
    dataset_cfg["splits"] = existing_splits

    class_names = normalized.get("names")
    if isinstance(class_names, (dict, list)):
        dataset_cfg["class_names"] = class_names
        class_count = len(class_names)
        model_cfg = normalized.get("model", {})
        if not isinstance(model_cfg, dict):
            model_cfg = {}
        model_cfg.setdefault("num_classes", class_count)
        normalized["model"] = model_cfg

    normalized["dataset"] = dataset_cfg

    for key in ("path", "train", "val", "test", "names"):
        normalized.pop(key, None)

    return normalized


def deep_merge(base: ConfigDict, update: ConfigDict) -> ConfigDict:
    """Recursively merge update into base and return a new dictionary."""
    merged = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_yaml(path: Path) -> ConfigDict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data or {}


def set_by_dot_key(cfg: ConfigDict, key: str, value: Any) -> None:
    """Set a nested key using dot notation, e.g. trainer.epochs=100."""
    cursor = cfg
    parts = key.split(".")
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def get_by_dot_key(cfg: ConfigDict, key: str) -> Any:
    """Get a nested key using dot notation, returning None if any part is missing."""
    cursor: Any = cfg
    for part in key.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return None
        cursor = cursor[part]
    return cursor


def pop_if_present(cfg: ConfigDict, key: str) -> Any:
    """Pop a top-level key if present, else return None."""
    if key in cfg:
        return cfg.pop(key)
    return None


def normalize_config(cfg: ConfigDict) -> ConfigDict:
    """Normalize flat external config keys into the nested internal schema.

    External config files should prefer flat keys for merge friendliness, similar to
    Ultralytics. Internal code continues to consume the nested structure.
    """
    normalized = deepcopy(cfg)

    # Ultralytics-style alias: data=<dataset_name_or_yaml>
    # Process data first so explicit dataset dict values (from later layers/CLI)
    # can still override it under the required precedence model.
    data_value = pop_if_present(normalized, "data")
    if isinstance(data_value, str):
        dataset_name = _dataset_name_from_ref(data_value)
        set_by_dot_key(normalized, "project.dataset", dataset_name)
        set_by_dot_key(normalized, "dataset.name", dataset_name)
        if _is_yaml_path(data_value):
            set_by_dot_key(normalized, "dataset.yaml", data_value)
    elif isinstance(data_value, list):
        refs = [item for item in data_value if isinstance(item, str) and item.strip()]
        if refs:
            dataset_names = [_dataset_name_from_ref(item) for item in refs]
            set_by_dot_key(normalized, "project.dataset", dataset_names[0])
            set_by_dot_key(normalized, "dataset.name", dataset_names[0])
            set_by_dot_key(normalized, "dataset.mix", dataset_names)
            set_by_dot_key(normalized, "dataset.yaml", refs)
    elif isinstance(data_value, dict):
        existing_dataset = normalized.get("dataset", {})
        normalized["dataset"] = deep_merge(existing_dataset, data_value)

    # Ultralytics-style imgsz shorthand.
    imgsz_value = pop_if_present(normalized, "imgsz")
    if imgsz_value is not None:
        if isinstance(imgsz_value, (int, float)):
            s = int(imgsz_value)
            set_by_dot_key(normalized, "input.size", [s, s])
        elif isinstance(imgsz_value, (list, tuple)):
            set_by_dot_key(normalized, "input.size", list(imgsz_value))

    dataset_value = pop_if_present(normalized, "dataset")
    if isinstance(dataset_value, str):
        dataset_name = _dataset_name_from_ref(dataset_value)
        set_by_dot_key(normalized, "project.dataset", dataset_name)
        set_by_dot_key(normalized, "dataset.name", dataset_name)
        if _is_yaml_path(dataset_value):
            set_by_dot_key(normalized, "dataset.yaml", dataset_value)
    elif isinstance(dataset_value, dict):
        existing_dataset = normalized.get("dataset", {})
        normalized["dataset"] = deep_merge(existing_dataset, dataset_value)

    optimizer_value = pop_if_present(normalized, "optimizer")
    if isinstance(optimizer_value, str):
        set_by_dot_key(normalized, "optimizer.name", optimizer_value)
    elif isinstance(optimizer_value, dict):
        existing_optimizer = normalized.get("optimizer", {})
        normalized["optimizer"] = deep_merge(existing_optimizer, optimizer_value)

    scheduler_value = pop_if_present(normalized, "scheduler")
    if isinstance(scheduler_value, str) or scheduler_value is None:
        set_by_dot_key(normalized, "scheduler.name", scheduler_value)
    elif isinstance(scheduler_value, dict):
        existing_scheduler = normalized.get("scheduler", {})
        normalized["scheduler"] = deep_merge(existing_scheduler, scheduler_value)

    task_value = pop_if_present(normalized, "task")
    if isinstance(task_value, str):
        set_by_dot_key(normalized, "project.task", task_value)
        set_by_dot_key(normalized, "task.name", task_value)
    elif isinstance(task_value, dict):
        existing_task = normalized.get("task", {})
        normalized["task"] = deep_merge(existing_task, task_value)
        # Always force task.name to be a string if possible
        if isinstance(task_value.get("name"), str):
            set_by_dot_key(normalized, "project.task", task_value["name"])
            set_by_dot_key(normalized, "task.name", task_value["name"])

    for flat_key, nested_key in FLAT_KEY_MAP.items():
        if flat_key in normalized:
            flat_value = normalized.pop(flat_key)
            # Preserve dict sections (e.g., distributed/resume) instead of treating
            # them as scalar aliases to a single nested key.
            if isinstance(flat_value, dict):
                existing = normalized.get(flat_key, {})
                if not isinstance(existing, dict):
                    existing = {}
                normalized[flat_key] = deep_merge(existing, flat_value)
                continue
            set_by_dot_key(normalized, nested_key, flat_value)

    if "project" in normalized and isinstance(normalized["project"], dict):
        project_task = normalized["project"].get("task")
        project_dataset = normalized["project"].get("dataset")
        if project_task and get_by_dot_key(normalized, "task.name") is None:
            set_by_dot_key(normalized, "task.name", project_task)
        if project_dataset and get_by_dot_key(normalized, "dataset.name") is None:
            set_by_dot_key(normalized, "dataset.name", project_dataset)

    # If task.name is still a dict, but has a 'name' key, force it to string
    task_name = get_by_dot_key(normalized, "task.name")
    if isinstance(task_name, dict) and isinstance(task_name.get("name"), str):
        set_by_dot_key(normalized, "task.name", task_name["name"])

    dataset_cfg = normalized.get("dataset")
    if isinstance(dataset_cfg, dict):
        split_keys = {
            "train": "train_path",
            "val": "val_path",
            "test": "test_path",
        }
        splits = dataset_cfg.get("splits")
        if not isinstance(splits, dict):
            splits = {}

        for split, legacy_key in split_keys.items():
            if split not in splits and legacy_key in dataset_cfg:
                splits[split] = dataset_cfg[legacy_key]

        if splits:
            dataset_cfg["splits"] = splits

        generator_cfg = dataset_cfg.get("generator")
        if isinstance(generator_cfg, dict):
            generator_data = generator_cfg.get("data")
            if not isinstance(generator_data, dict):
                generator_data = {}
            for split in ("train", "val", "test"):
                if split not in generator_data and split in splits:
                    generator_data[split] = splits[split]
            generator_cfg["data"] = generator_data

    return normalized


def load_layered_config(
    root_dir: Path,
    task: str | None,
    dataset: str | None,
    experiment: str | None,
    extra_config_paths: list[Path],
    cli_overrides: list[str],
) -> ConfigDict:
    """Load and merge configs with strict precedence.

    Precedence (later wins):
    1) cfg/default.yaml
    2) cfg/tasks/<task>.yaml
    3) cfg/experiments/<task>/<experiment>.yaml
    4) extra config file paths (--config)
    5) CLI overrides
    """
    cfg_dir = root_dir / "cfg"
    merged: ConfigDict = {}

    ordered_paths = [cfg_dir / "default.yaml"]
    if task:
        ordered_paths.append(cfg_dir / "tasks" / f"{task}.yaml")
    if experiment:
        if task:
            ordered_paths.append(cfg_dir / "experiments" / task / f"{experiment}.yaml")
        else:
            ordered_paths.append(cfg_dir / "experiments" / f"{experiment}.yaml")

    for cfg_path in ordered_paths:
        if not cfg_path.exists():
            continue
        file_cfg = load_yaml(cfg_path)
        file_cfg = _normalize_yolo_dataset_schema(file_cfg, cfg_path)

        # Resolve data=<dataset|yaml> within this same layer.
        layer_data_ref = file_cfg.get("data")
        if isinstance(layer_data_ref, (str, list)):
            dataset_cfg = _load_dataset_cfg_from_refs(layer_data_ref, root_dir=root_dir, cfg_dir=cfg_dir)
            if dataset_cfg:
                file_cfg = deep_merge(file_cfg, dataset_cfg)

        merged = deep_merge(merged, file_cfg)

    # User-provided config files are applied after default/task/experiment layers,
    # but before CLI overrides.
    for cfg_path in extra_config_paths:
        resolved = cfg_path if cfg_path.is_absolute() else (root_dir / cfg_path)
        if not resolved.exists():
            raise FileNotFoundError(f"Config file not found: {resolved}")
        file_cfg = load_yaml(resolved)
        file_cfg = _normalize_yolo_dataset_schema(file_cfg, resolved)

        layer_data_ref = file_cfg.get("data")
        if isinstance(layer_data_ref, (str, list)):
            dataset_cfg = _load_dataset_cfg_from_refs(layer_data_ref, root_dir=root_dir, cfg_dir=cfg_dir)
            if dataset_cfg:
                file_cfg = deep_merge(file_cfg, dataset_cfg)

        merged = deep_merge(merged, file_cfg)

    effective_cli_overrides = list(cli_overrides)
    if dataset:
        # CLI dataset arg remains highest precedence.
        effective_cli_overrides.append(f"data={dataset}")

    def _apply_overrides(target: ConfigDict, overrides: list[str]) -> None:
        for item in overrides:
            if "=" not in item:
                raise ValueError(f"Invalid override '{item}'. Use key=value format.")
            key, raw_value = item.split("=", 1)
            value = yaml.safe_load(raw_value)
            set_by_dot_key(target, key, value)

    _apply_overrides(merged, effective_cli_overrides)

    # If CLI selects a dataset ref, load dataset YAML in CLI stage,
    # then re-apply CLI overrides so CLI stays highest.
    cli_data_ref = _last_cli_override(effective_cli_overrides, "data")
    if isinstance(cli_data_ref, (str, list)):
        cli_dataset_cfg = _load_dataset_cfg_from_refs(cli_data_ref, root_dir=root_dir, cfg_dir=cfg_dir)
        if cli_dataset_cfg:
            merged = deep_merge(merged, cli_dataset_cfg)
            _apply_overrides(merged, effective_cli_overrides)

    return normalize_config(merged)


def dump_yaml(cfg: ConfigDict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(cfg, sort_keys=False)
    path.write_text(text, encoding="utf-8")
