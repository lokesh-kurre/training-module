# training-module

General-purpose training module skeleton for future ML tasks.

## Goal

Build a pluggable and configurable training framework where:

1. generic and reusable code lives in `training/`
2. experiment and task parameters live in `cfg/`
3. new tasks (classification, detection, etc.) can be added without changing core engine code

Layout direction is inspired by Ultralytics and pluggable ideas from StyleGAN3.

## Recommended Structure

```text
.
├── cfg/
│   ├── default.yaml
│   ├── datasets/
│   │   ├── dataset_a.yaml
│   │   └── dataset_b.yaml
│   ├── tasks/
│   │   ├── classification.yaml
│   │   └── detection.yaml
│   └── experiments/
│       └── classification/
│           ├── ddp_smoke.yaml
│           └── sample_generator_mobilenet_v1.yaml
├── data/
├── pretrained_weights/
├── notebooks/
├── run/
├── scripts/
├── training/
│   ├── __init__.py
│   ├── callbacks/
│   ├── data/
│   │   ├── datasets/
│   │   ├── samplers/
│   │   ├── loaders/
│   │   └── transforms/
│   ├── engine/
│   ├── nets/
│   ├── tasks/
│   │   ├── __init__.py
│   │   ├── classification/
│   │   │   ├── model.py
│   │   │   ├── losses.py
│   │   │   └── metrics.py
│   ├── registry.py
│   ├── config.py
│   └── utils/
├── train.py
├── requirements.txt
└── .env
```

## Decision: Where Experiment `.py` Files Should Live

Keep this boundary:

1. `cfg/` should contain configuration only (`.yaml`)
2. all Python code should stay in `training/`

Why:

1. clear separation of concerns
2. easier testing and import paths
3. easier packaging and deployment
4. less risk of hidden logic in config folders

If experiment-specific helper code is needed, place it under task modules, for example:

- `training/tasks/classification/...`

## Config Layering Strategy

Use deterministic merge order:

1. `cfg/default.yaml`
2. `cfg/tasks/<task>.yaml`
3. `cfg/experiments/<task>/<exp>.yaml`
4. CLI overrides (`key=value`)

This gives reproducible runs while keeping each config file small and focused.
Dataset YAML is resolved from top-level `data` (or CLI `--dataset`) and merged in the same loading flow while preserving CLI highest precedence.

## Pluggable Architecture Plan

Implement a registry-driven architecture:

1. register models, losses, metrics, callbacks, optimizers, schedulers by name
2. select components from config names (not hardcoded imports)
3. keep train loop generic in `training/engine/`

Suggested core contracts:

1. `build_model(cfg)`
2. `build_dataset(cfg, split)`
3. `build_loss(cfg)`
4. `build_metrics(cfg)`
5. `Trainer.fit()` and `Trainer.validate()`

## Run/Reproducibility Conventions

Each run in `run/<timestamp_or_name>/` should store:

1. merged final config (`config_resolved.yaml`)
2. logs
3. checkpoints
4. metrics (`metrics.json`/`csv`)
5. environment snapshot (`pip freeze` or equivalent)

Checkpointing and metrics persistence are callback-driven.
Runtime loop state passed to hooks is represented by `TrainState`.

This is critical for a skeleton that will be reused long term.

## StyleGAN3-inspired Features (Optional)

From StyleGAN3 references:

1. import/util helpers: use a controlled dynamic import utility for plugins
2. persistence helpers: provide robust checkpoint serialization and load utilities

Avoid overusing pickle for cross-version portability. Prefer state-dict checkpoints by default.

## Documentation

The following documentation covers different aspects of the framework:

### Architecture & Design
- **[VIBE_CODER_ARCHITECTURE.md](docs/VIBE_CODER_ARCHITECTURE.md)** - Core design principles, vibe-coder alignment, anti-patterns, and extensibility patterns
- **[PROJECT_STRUCTURE_AND_FLOW.md](docs/PROJECT_STRUCTURE_AND_FLOW.md)** - Directory structure, config flow, entry points, S3 patterns

### Usage & Integration
- **[LIBRARY_USAGE.md](docs/LIBRARY_USAGE.md)** - Using training-module as an installed library, custom builders, extending with models/datasets/losses/callbacks
- **[LOGGING_AND_PROGRESS.md](docs/LOGGING_AND_PROGRESS.md)** - Logging architecture, progress bars (console vs file), troubleshooting

### Testing
- **[tests/README.md](tests/README.md)** - Test infrastructure, mock data organization, pytest fixtures, writing tests

## Implementation Roadmap

### Phase 1 (Foundation)

1. create folder structure above
2. add config loader with layered merge
3. add minimal registry implementation
4. add baseline `train.py` CLI

### Phase 2 (Task Enablement)

1. implement one task end-to-end (`classification`)
2. add one dataset adapter
3. add metrics and checkpoint callbacks

### Phase 3 (Pluggable Expansion)

1. add second task (`detection`)
2. verify no engine changes required for new task
3. add plugin discovery/tests

### Phase 4 (Operational Hardening)

1. DDP support in engine
2. structured logging and experiment tracking hooks (tensorboard/mlflow/dvc)
3. resume/restart reliability tests

## Minimal Rules for This Skeleton

1. no task logic inside engine core
2. no Python code inside `cfg/`
3. all runnable experiments must be reproducible from saved resolved config
4. each new task must be addable by config + registration, not by editing trainer internals

## Train CLI

Install dependencies:

```bash
pip install -r requirements.txt
```

Install training backbones (CPU-only, recommended for local dev):

```bash
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu -r requirements-train.txt
```

Quick commands via Makefile:

```bash
make help        # Show all available commands
make install     # Install in editable mode
make install-dev # Install with dev tools (lint, test, build)
make train       # Run training with sample args
make dry-run     # Validate config without training
make format      # Auto-format code
make lint        # Lint code
make test        # Run tests
```

Dry-run with layered config:

```bash
python train.py --task classification --dataset imagenet --experiment sample_generator_mobilenet_v1 --dry-run
```

Run with overrides:

```bash
python train.py --task classification --dataset imagenet --experiment sample_generator_mobilenet_v1 --set trainer.epochs=25 --set optimizer.lr=0.0003
```

Run with explicit run name:

```bash
python train.py --task classification --dataset imagenet --experiment sample_generator_mobilenet_v1 --run-name cls_sample_v1
```

Distributed training via torchrun:

```bash
torchrun --nproc_per_node=2 train.py --task classification --dataset imagenet --experiment sample_generator_mobilenet_v1 --distributed --distributed-backend gloo
```

DDP smoke test (2 ranks on CPU/gloo):

```bash
bash scripts/smoke_ddp.sh
```

Select a popular backbone from CLI:

```bash
python train.py --task classification --dataset imagenet --experiment sample_generator_mobilenet_v1 --set model.architecture=resnet50 --set model.num_classes=1000
```

For minimal environments without torchvision, use the built-in smoke-test backbone:

```bash
python train.py --task classification --dataset imagenet --experiment sample_generator_mobilenet_v1 --set model.architecture=tiny_cnn --set epochs=1
```

Supported architecture names:

```text
tiny_cnn, mobilenet_v1, mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large, resnet18, resnet50
```

## Using training-module as a Library

The `training-module` is designed to be both a CLI tool and a reusable Python library. Install it as a wheel in your own projects:

```bash
# Build wheel
make build

# Install in another project
pip install dist/training_module-*.whl
```

Then import and use it in your code:

```python
from pathlib import Path

from training.config import load_layered_config
from training.engine import Trainer
from training.tasks.registry import get_task_class

cfg = load_layered_config(
  root_dir=Path("."),
  task="classification",
  dataset="imagenet",
  experiment="sample_generator_mobilenet_v1",
  extra_config_paths=[],
  cli_overrides=[],
)
trainer = Trainer(cfg=cfg, project_root=Path("."))
trainer.fit()
```

### Extending with Custom Objects

The framework uses a **pluggable builder pattern** to allow you to register custom models, datasets, losses, and callbacks without modifying core code.

Pass custom modules to task builders:

```python
from training.tasks.registry import get_task_class
import my_project.models as custom_models
import my_project.losses as custom_losses

TaskClass = get_task_class("classification")
task = TaskClass(
    cfg=cfg,
    model_module=custom_models,   # Your custom models
    loss_module=custom_losses,    # Your custom losses
)
```

Reference custom objects in YAML config:

```yaml
# cfg/experiments/classification/custom.yaml
model:
  arch: MyCustomResNet
  num_classes: 1000

loss: MyCustomFocalLoss
loss_alpha: 0.25
```

The builder automatically resolves string names to your custom objects at runtime.

**For detailed examples, see [LIBRARY_USAGE.md](docs/LIBRARY_USAGE.md)** which covers:

- Installation and setup as a wheel
- Custom training scripts (library mode)
- Custom models, datasets, losses, and callbacks
- Builder pattern examples
- Project structure recommendations

## Optional Shared-Memory Data Backend

The default pipeline uses standard torch DataLoader workers.

You can optionally switch to the shared-memory generator backend (for reduced IPC overhead) by enabling:

```bash
python train.py \
  --task classification \
  --dataset dataset_a \
  --experiment baseline \
  --set dataloader.backend=data_gen \
  --set dataset.backend=data_gen \
  --set dataset_generator.read_func=your_package.your_module.read_func \
  --set dataset_generator.data.train=path/to/train.csv \
  --set dataset_generator.data.val=path/to/val.csv
```

Notes:

1. The configured read function must return values matching dataset_generator.input_dtype.
2. The data_gen backend expects pre-batched IterableDataset output and bypasses DataLoader samplers.
3. If not enabled, the standard torch backend behavior is unchanged.

## API Signature Cross-Reference

- `load_layered_config(root_dir: Path, task: str | None, dataset: str | None, experiment: str | None, extra_config_paths: list[Path], cli_overrides: list[str])`
  - Source: [training/config.py](training/config.py#L347)
- `Trainer(cfg: dict[str, Any], project_root: Path)`
  - Source: [training/engine/trainer.py](training/engine/trainer.py#L28)
- `get_task_class(name: str)` / `available_tasks()`
  - Source: [training/tasks/registry.py](training/tasks/registry.py#L15)
- `get_obj_by_name(name: str)`, `call_func_by_name(*args, func_name=..., **kwargs)`, `construct_class_by_name(*args, class_name=..., **kwargs)`
  - Source: [training/utils/importer.py](training/utils/importer.py#L58)
