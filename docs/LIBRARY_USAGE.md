# Using training-module as a Library

This document shows the current, code-accurate public usage of the package, with direct links to the source function signatures.

## Installation

```bash
pip install -e /path/to/training-module
# or from repo root
make install
```

Build wheel:

```bash
make build
pip install dist/training_module-*.whl
```

## Canonical Imports

```python
from pathlib import Path

from dotenv import load_dotenv

from training.config import load_layered_config
from training.engine import Trainer
from training.tasks.registry import get_task_class, available_tasks
from training.utils.importer import get_obj_by_name, call_func_by_name, construct_class_by_name
```

## Function Signature Cross-Reference

- Config loader:
  - `load_layered_config(root_dir: Path, task: str | None, dataset: str | None, experiment: str | None, extra_config_paths: list[Path], cli_overrides: list[str]) -> dict[str, Any]`
  - Source: [training/config.py](../training/config.py#L347)
- Task registry:
  - `get_task_class(name: str) -> Type[Any]`
  - `available_tasks() -> list[str]`
  - Source: [training/tasks/registry.py](../training/tasks/registry.py#L15)
- Dynamic import helpers:
  - `get_obj_by_name(name: str) -> Any`
  - `call_func_by_name(*args, func_name: str = None, **kwargs) -> Any`
  - `construct_class_by_name(*args, class_name: str = None, **kwargs) -> Any`
  - Source: [training/utils/importer.py](../training/utils/importer.py#L58)
- Data builders:
  - `build_dataset_from_config(cfg: dict[str, Any], split: str = "train", rank: int = 0, world_size: int = 1, runtime: Any = None) -> Dataset`
  - `build_dataloader(cfg: dict[str, Any], dataset: Any, split: str, runtime: Any) -> DataLoader`
  - Source: [training/data/datasets/factory.py](../training/data/datasets/factory.py#L11), [training/data/loaders/builder.py](../training/data/loaders/builder.py#L10)
- IO helpers:
  - `read_file(filepath: str | None = None, verbose: bool = False, s3_client: Any = None, as_stream: bool = False, **kwargs) -> bytes | Any | None`
  - `read_image(filepath: str | None = None, as_pillow: bool = False, cv2_imdecode_mode: int = -1, verbose: bool = False, s3_client: Any = None, **kwargs) -> Any | None`
  - `list_files(directory: str | None = None, /, recursive: bool = False, count: int = 1000, verbose: bool = False, s3_client: Any = None, **kwargs) -> Iterator[str]`
  - Source: [training/utils/io.py](../training/utils/io.py#L66)

## Minimal Library Example

```python
from pathlib import Path

from dotenv import load_dotenv

from training.config import load_layered_config
from training.engine import Trainer

load_dotenv(override=False)

project_root = Path(__file__).resolve().parent

cfg = load_layered_config(
    root_dir=project_root,
    task="classification",
    dataset="imagenet",
    experiment="sample_generator_mobilenet_v1",
    extra_config_paths=[],
    cli_overrides=["trainer.epochs=2", "batch=8"],
)

trainer = Trainer(cfg=cfg, project_root=project_root)
trainer.fit(run_name="library_example")
```

## Using Task Classes Directly

```python
from training.tasks.registry import get_task_class

TaskClass = get_task_class("classification")
task = TaskClass(cfg)
```

To inspect registered tasks:

```python
from training.tasks.registry import available_tasks

print(available_tasks())
```

## Dynamic Builder Utilities

Use fully-qualified Python names for runtime resolution:

```python
obj = get_obj_by_name("training.tasks.classification.model.build_model")
metric_value = call_func_by_name([1, 2, 3], func_name="builtins.sum")
counter = construct_class_by_name(class_name="collections.Counter")
```

## S3/CEPH IO Usage

Preferred in worker-style pipelines:

```python
from training.utils.io import read_image

# Reuse caller-managed client when available.
sample = read_image("s3://bucket/key.jpg", s3_client=my_s3_client)
```

If no client is passed, IO uses a thread/process-local cached client internally.

## Notes

- Always call `load_dotenv(override=False)` early in entrypoints when using env-backed storage credentials.
- For CLI-compatible behavior in library mode, keep config precedence aligned with [training/config.py](../training/config.py#L347).
