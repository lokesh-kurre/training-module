# Test Infrastructure

This directory contains test utilities, mock data, and test configurations for `training-module`.

## Directory Structure

```
tests/
├── conftest.py              # Pytest configuration and shared fixtures
├── data/                    # Mock datasets for fast local testing
│   ├── train_split.csv      # Test training data manifest
│   ├── val_split.csv        # Test validation data manifest
│   └── test_dataset/        # Test image files (generated or mocked)
├── configs/                 # Minimal test configurations
│   ├── test_classification.yaml  # Fast smoke test config
│   └── test_dataset.yaml    # Mock dataset definition
└── fixtures/                # Reusable test fixtures and utilities
```

## Design Principles

Based on vibe-coder architecture practices:

### 1. Test Data Isolation

- Test data lives **exclusively in `tests/data/`**, NOT in main `cfg/` directories
- Test configs live in `tests/configs/`, not in production `cfg/`
- Keeps production configuration clean
- Easy to exclude test files from wheel distributions

### 2. Minimal Mock Data

- Test data is **intentionally small** for fast iteration
- Use CSV manifests (`train_split.csv`, `val_split.csv`) to reference mock images
- Images can be:
  - Tiny PIL-generated images (32x32 RGB)
  - Cached fixtures
  - Symbolic/empty placeholders
- Goal: < 1 second training epoch for CI/CD validation

### 3. Pytest Fixtures (conftest.py)

Shared fixtures available to all tests:

- `test_data_dir`: Path to `tests/data/`
- `test_config_dir`: Path to `tests/configs/`
- `tmp_run_dir`: Temporary run directory for test outputs
- `mock_config`: Minimal mock config dict for unit tests

Usage in tests:

```python
def test_training_step(mock_config):
    trainer = Trainer(cfg=mock_config, project_root=Path("."))
    # ... test logic
```

### 4. Test Configuration Conventions

Test configs follow naming: `test_*.yaml`

- `test_classification.yaml`: Minimal classification task
  - 1 epoch, batch_size=4, device=cpu
  - Progress bar disabled (quieter output)
  - Timing disabled (faster)
  - `show_progress_bar: false` (keeps logs clean)

## Running Tests

### Run all tests

```bash
pytest tests/ -v
```

### Run specific test file

```bash
pytest tests/test_trainer.py -v
```

### Run with coverage

```bash
pytest tests/ --cov=training --cov-report=html
```

### Run smoke test (fast integration test)

```bash
python train.py \
    --task classification \
    --dataset test_dataset \
    --config tests/configs/test_classification.yaml \
    --dry-run
```

## Writing Tests

### Unit Test Template

```python
import pytest
from pathlib import Path
from training.config import load_layered_config
from training.engine import Trainer


def test_config_loading(test_config_dir):
    """Test config layering with test configs."""
    cfg = load_layered_config(
        root_dir=Path("."),
        task="classification",
        dataset="imagenet",
        experiment="sample_generator_mobilenet_v1",
        extra_config_paths=[],
        cli_overrides=[],
    )
    assert cfg is not None
    assert cfg.get("task", {}).get("name") == "classification"


def test_training_step(mock_config):
    """Test trainer initialization."""
    trainer = Trainer(cfg=mock_config, project_root=Path("."))
    assert trainer.cfg is not None
    assert trainer.project_root is not None


def test_with_temp_run(tmp_run_dir):
    """Test with temporary run directory."""
    config_file = tmp_run_dir / "config.yaml"
    config_file.write_text("test: value")
    assert config_file.exists()
```

## CI/CD Integration

Add to `.github/workflows/tests.yml` (or similar):

```yaml
- name: Run tests
  run: |
    pip install -e ".[dev]"
    pytest tests/ -v --tb=short

- name: Smoke test
  run: |
    python train.py \
        --task classification \
        --dataset test_dataset \
        --config tests/configs/test_classification.yaml \
        --dry-run
```

## Best Practices

1. **Keep test data small**
   - Use 32x32 RGB images or PIL placeholders
   - Aim for < 1MB test dataset

2. **Disable expensive features in test configs**
   - `show_progress_bar: false`
   - `timing_enabled: false`
   - `log_heartbeat: false`
   - `num_workers: 0` (avoid multiprocessing complexity)

3. **Mock external dependencies**
   - Use fixtures for S3 clients if needed
   - Avoid real network calls
   - Use pytest-mock for patching

4. **Test config, not just code**
   - Verify config layering works correctly
   - Verify YAML merging
   - Verify CLI override precedence

5. **Keep test fixtures focused**
   - One responsibility per fixture
   - Compose fixtures for complex setups
   - Cleanup after tests (use tmp_path)

## Extending Test Infrastructure

### Adding a New Task Test

1. Create `tests/configs/test_<task>.yaml` with minimal config
2. Add mock data references in `tests/data/`
3. Create `tests/test_<task>.py` with task-specific tests

### Adding Test Utilities

1. Create modules in `tests/fixtures/`
2. Import fixtures in `conftest.py` if shared globally
3. Or import locally in test files if task-specific

### Mocking External Services

Use `pytest-mock` for S3, external APIs:

```python
def test_with_mock_s3(mocker):
    """Test with mocked S3 client."""
    mock_client = mocker.MagicMock()
    mocker.patch("training.utils.io._resolve_s3_client", return_value=mock_client)
    # ... test logic
```

## Performance Tips

- **Parallelize tests**: `pytest -n auto` (requires pytest-xdist)
- **Cache test data**: Use fixtures with session scope for expensive setups
- **Profile slow tests**: `pytest --durations=10` shows slowest tests
- **Skip slow tests locally**: Use `@pytest.mark.slow` decorator

## Documentation References

- [Pytest Official Docs](https://docs.pytest.org/)
- [Pytest Fixtures](https://docs.pytest.org/en/stable/example/fixtures.html)
- [Pytest Plugins](https://docs.pytest.org/en/latest/how-to-use-plugins.html)

## Related Files

- `.gitignore`: Excludes `tests/` from production distributions
- `pyproject.toml`: Contains `pytest` in dev dependencies
- `Makefile`: `make test` target runs pytest
