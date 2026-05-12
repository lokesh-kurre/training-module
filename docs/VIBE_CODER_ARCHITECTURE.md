# Architecture and Design Principles

This document outlines how `training-module` follows **vibe-coder** principles for building scalable, reproducible, and pluggable deep learning training frameworks.

## Core Principles

### 1. Strict Config/Code Separation ✅

**Principle**: Configuration directories MUST remain declarative only.

**Implementation**:
- ✅ `cfg/` contains ONLY `.yaml` files
- ✅ No `.py` files in `cfg/`
- ✅ No executable logic in `cfg/`
- ✅ All code lives in `training/`

**Example**:
```
cfg/
├── default.yaml              # YAML only
├── tasks/
│   └── classification.yaml   # YAML only
└── experiments/
    └── classification/
        └── baseline.yaml     # YAML only

training/
├── tasks/classification/     # Code here
│   ├── model.py
│   └── losses.py
```

**Violation Example (WRONG)**:
```
cfg/classification/helper.py  # ❌ NO Python files in cfg/
```

### 2. Tasks as Primary Scaling Abstraction ✅

**Principle**: Framework scales by TASKS, not experiments.

**Implementation**:
- Each task in `training/tasks/<task>/` owns:
  - `build_model()`: Model construction
  - `build_losses()`: Loss functions
  - `build_metrics()`: Metric computation
  - `training_step()`: Forward + loss computation
  - `validation_step()`: Validation logic
  - `predict_step()`: Inference logic

**Code**:
```python
# training/tasks/base.py
class BaseTask(ABC):
    @abstractmethod
    def training_step(self, batch) -> TaskStepOutput:
        """Task owns all forward/loss/metric computation."""
        raise NotImplementedError
    
    @abstractmethod
    def validation_step(self, batch) -> TaskStepOutput:
        raise NotImplementedError
    
    @abstractmethod
    def predict_step(self, batch) -> Any:
        raise NotImplementedError
```

**Auto-Discovery** (No Manual Registration):
```python
# training/tasks/__init__.py
# Automatically imports all task packages
for module in pkgutil.iter_modules(__path__):
    importlib.import_module(f".{module.name}", __package__)
```

**Usage**:
```bash
# New task added automatically via directory
training/tasks/detection/
├── __init__.py
├── model.py
└── losses.py

# No manual edits to training/tasks/__init__.py required ✅
```

### 3. Generic Engine (No Task-Specific Logic) ✅

**Principle**: Engine handles generic concerns only.

**Engine Responsibilities**:
- ✅ Epoch/batch loops
- ✅ Gradient accumulation
- ✅ AMP (Mixed Precision)
- ✅ DDP coordination
- ✅ Callback execution
- ✅ Checkpointing
- ✅ Logging and progress tracking

**NOT the Engine's Responsibility**:
- ❌ Task-specific losses
- ❌ Task-specific metrics
- ❌ Dataset-specific preprocessing
- ❌ Augmentation logic
- ❌ Model architecture
- ❌ Inference pipelines

**Code Pattern**:
```python
# training/engine/trainer.py
def _train_epoch(self, state, task, train_loader, ...):
    for batch in train_loader:
        # Generic engine logic
        batch = self._move_batch_to_device(batch, device)
        optimizer.zero_grad()
        
        # DELEGATE to task (not hardcoded)
        step = task.training_step(batch)  # ✅ Task-agnostic
        
        # Generic engine logic
        step.loss.backward()
        optimizer.step()
        callbacks.on_batch_end(state)
```

**NOT like this**:
```python
# ❌ BAD: Task-specific branching
if task.name == "classification":
    loss = cross_entropy(...)
elif task.name == "detection":
    loss = yolo_loss(...)
```

### 4. Everything Pluggable via Registries ✅

**Principle**: All components support factory/registry loading.

**What's Pluggable**:
- ✅ Tasks (auto-discovered)
- ✅ Models (via builder pattern)
- ✅ Datasets (via config)
- ✅ Losses (via config)
- ✅ Metrics (via config)
- ✅ Optimizers (via config)
- ✅ Schedulers (via config)
- ✅ Callbacks (via config)

**Registry Pattern**:
```python
# training/registry.py
def register(kind: str, name: str):
    """Decorator-based registration."""
    def decorator(fn):
        _REGISTRIES[kind][name] = fn
        return fn
    return decorator

def get(kind: str, name: str):
    """Factory lookup."""
    return _REGISTRIES[kind][name]
```

**Usage Example**:
```python
# training/tasks/classification/__init__.py
@register("task", "classification")
class ClassificationTask(BaseTask):
    def build_model(self):
        return build_backbone(self.cfg)

# Automatic registration on import ✅
```

**Config-Driven Construction**:
```yaml
# cfg/experiments/classification/baseline.yaml
model:
  architecture: resnet50  # String reference
  num_classes: 1000

loss: cross_entropy  # String reference
optimizer:
  name: adam           # String reference
  lr: 0.001
```

**Resolution at Runtime**:
```python
# String names resolved via builder utilities (fully-qualified names)
model_builder = get_obj_by_name("training.tasks.classification.model.build_model")
model = model_builder(cfg)
```

### 5. Clean Ownership Boundaries ✅

**Principle**: Clear separation of who owns what.

**Ownership Model**:
| Component | Owner | Responsibility |
|-----------|-------|-----------------|
| Forward pass | Task | Compute model output |
| Loss computation | Task | Compute loss from outputs |
| Metrics | Task | Compute per-batch metrics |
| Backward pass | Engine | Compute gradients |
| Optimizer step | Engine | Update parameters |
| Checkpointing | Engine | Save/load state |
| Callbacks | Callbacks | Side effects (logging, EMA, etc.) |
| Config merging | Config layer | Load and layer configs |
| Device management | Engine runtime | Distribute across devices |

**Example: No Overlap**:
```python
# ✅ CORRECT: Clear boundaries
# Task owns training_step
step = task.training_step(batch)  # Returns loss, metrics

# Engine owns backward
step.loss.backward()

# Engine owns optimizer
optimizer.step()

# Callbacks own side effects
callbacks.on_batch_end(state)
```

### 6. Minimal Hidden Behavior ✅

**Principle**: Explicitly visible code, not implicit magic.

**No Silent Mutations**:
```python
# ❌ BAD: Hidden side effects
def train(cfg):
    cfg.lr *= world_size  # Silent mutation!

# ✅ GOOD: Explicit
def train(cfg):
    world_size = get_world_size()
    scaled_lr = cfg.lr * world_size  # Visible
    optimizer = Adam(model.parameters(), lr=scaled_lr)
```

**Explicit Config Layering**:
```python
# training/config.py
def load_layered_config(
    root_dir: Path,
    task: str | None,
    dataset: str | None,
    experiment: str | None,
    extra_config_paths: list[Path],
    cli_overrides: list[str],
) -> dict[str, Any]:
    ...
```

**Visible Precedence** (Later overrides earlier):
```
cfg/default.yaml
  ↓ overrides
cfg/tasks/<task>.yaml
  ↓ overrides
cfg/experiments/<task>/<experiment>.yaml
  ↓ overrides
CLI --set key=value
```

### 7. Long-Term Extensibility ✅

**Principle**: Designed for 5+ year growth.

**New Task Addition (No Core Changes)**:
```bash
# Add new task
mkdir -p training/tasks/detection
cat > training/tasks/detection/__init__.py << 'EOF'
from training.tasks.base import BaseTask
from training.registry import register

@register("task", "detection")
class DetectionTask(BaseTask):
    def build_model(self): ...
    def training_step(self, batch): ...
    # ... implement interface
EOF

# Create config
cat > cfg/tasks/detection.yaml << 'EOF'
task: detection
model: yolov8
# ...
EOF

# Trainer works with new task WITHOUT code changes ✅
python train.py --task detection
```

**New Model Support (No Trainer Changes)**:
```python
# training/nets/my_custom_model.py
import torch.nn as nn

class MyCustomModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # implementation

# Config references it
# cfg/experiments/classification/custom.yaml
model:
  architecture: MyCustomModel
  num_classes: 1000

# Trainer resolves via builder (no hardcoding) ✅
```

## Run Directory Convention ✅

**Structure**:
```
run/<run_name>/
├── config_resolved.yaml        # Merged config for reproducibility
├── log.txt                     # Training logs
├── checkpoints/
│   ├── best.pt                # Best checkpoint
│   ├── last.pt                # Latest checkpoint
│   └── epoch_050.pt           # Epoch-based checkpoints
└── metrics/
    └── metrics.json           # Aggregated metrics
```

**Reproducibility**:
```bash
# Exact reproduction: just use resolved config
python train.py --config run/exp_001/config_resolved.yaml
```

## Testing Infrastructure ✅

**Test Data Organization**:
```
tests/
├── conftest.py              # Pytest fixtures
├── data/                    # Mock datasets (excluded from wheel)
│   ├── train_split.csv
│   └── val_split.csv
├── configs/                 # Test configs (excluded from wheel)
│   ├── test_classification.yaml
│   └── test_dataset.yaml
└── fixtures/                # Reusable test utilities
```

**Test Config Convention**:
- Named `test_*.yaml`
- Minimal size (1-2 epoch)
- CPU-only
- No progress bars (cleaner output)
- Fast (< 1s per epoch)

**No Test Files in Production**:
```bash
# wheel build excludes test data
pip install dist/training-module-*.whl
# ✅ tests/ directory not included
```

## Distributed Training (DDP) ✅

**Concerns Isolated**:
```python
# training/engine/runtime.py
def setup_distributed():
    """Setup DDP with all initialization."""
    
# training/engine/trainer.py
def _is_rank0(self, runtime):
    """Query rank for selective logging."""
    return runtime.is_main_process
```

**Transparent to Engine Core**:
```python
# Trainer handles single-GPU and DDP identically
def _train_epoch(self, state, task, ...):
    for batch in train_loader:
        # Same code path whether DDP or single-GPU ✅
        step = task.training_step(batch)
        step.loss.backward()  # DDP handles sync automatically
        optimizer.step()
```

## Configuration Philosophy

### Flattened Top-Level Keys (Ultralytics Style)

```yaml
# cfg/default.yaml - flat keys for user-facing settings
task: classification
data: imagenet
model:
  architecture: resnet50
batch: 64
epochs: 100
device: cuda
imgsz: 224

# Nested sections for engine internals
trainer:
  log_level: info
optimizer:
  name: adam
  lr: 0.001
scheduler:
  name: cosine
  t_max: 100
```

**Why Flattened**:
- ✅ Simple for users (no deep nesting)
- ✅ Flat keys in CLI overrides (`--set batch=32`)
- ✅ Matches Ultralytics precedent

### Full Config Resolution

Every builder receives fully merged config:

```python
# Not just minimal sub-config
task.build_model(cfg)  # ✅ Entire cfg

# Allows adding new keys to default.yaml
# and using them anywhere (no modification to builders)
```

## Anti-Patterns to Avoid

❌ **God Trainer** (5000+ line trainer.py)
- Solution: Delegate to task via `task.training_step()`

❌ **Hidden Runtime Mutation**
```python
cfg.lr *= world_size  # Silent!
```
- Solution: Explicit scaling, logged

❌ **Executable Config Directories**
```python
# ❌ Wrong
cfg/spoofbuster/helper.py

# ✅ Right
training/tasks/spoofbuster/
```

❌ **Dataset-Specific Engine Logic**
```python
if dataset == "casia":  # ❌
    special_processing()
```
- Solution: Delegate to dataset builder

❌ **Implicit Magic (Import Side Effects)**
```python
# ❌ BAD: Silent registration on import
import all_losses  # magic happens

# ✅ GOOD: Explicit registration decorator
@register("loss", "focal")
class FocalLoss: ...
```

## Summary

## Signature References

- Config layering source: [training/config.py](../training/config.py#L347)
- Task registry source: [training/tasks/registry.py](../training/tasks/registry.py#L15)
- Dynamic importer source: [training/utils/importer.py](../training/utils/importer.py#L58)
- Trainer loop source: [training/engine/trainer.py](../training/engine/trainer.py#L28)

This framework achieves vibe-coder principles through:

1. **Strict separation**: Config files have no code
2. **Task-first architecture**: Tasks own their logic, engine stays generic
3. **Pluggability**: Registries and builders enable unlimited extension
4. **Clean boundaries**: Clear ownership prevents hidden coupling
5. **No magic**: Explicit config layering and setup
6. **Designed for scale**: Adding tasks/models requires no core changes

The result: maintainable, reproducible, scalable framework suitable for 5+ years of research and production use.
