# Training Module: Project Structure and Runtime Flow

This document explains how the repository is organized and how config/runtime flow through the system.

## Design Goals

- Keep configuration declarative under `cfg/`.
- Keep executable logic under `training/`.
- Keep trainer task-agnostic; task owns forward/loss/metrics.
- Make new tasks pluggable with minimal edits.
- Keep runtime behavior reproducible and explicit.

## Directory Overview

- `cfg/`
- declarative configuration only (`.yaml`, `.json`)
- layering base + task + experiment + CLI overrides

- `cfg/default.yaml`
- flattened, Ultralytics-style defaults
- canonical runtime knobs and placeholders

- `cfg/tasks/`
- task-specific defaults (`classification.yaml`, etc.)

- `cfg/experiments/<task>/`
- experiment-level overrides

- `cfg/datasets/`
- dataset YAML definitions (`path`, `train/val/test`, `names`)
- referenced by top-level `data:` key

- `cfg/schema/`
- JSON schema files for config and dataset YAML validation/documentation

- `training/engine/`
- generic runtime loop, checkpoints, callbacks orchestration

- `training/tasks/`
- task implementations and task contract
- auto-discovery imports task packages dynamically

- `training/data/`
- dataset and dataloader builders
- torch backend and generator backend support

- `training/nets/`
- model/backbone construction
- local-vs-remote pretrained handling centralized in `backbone.py`

- `training/utils/`
- utility helpers (`io`, checkpoint, logger, env, optimizer, summary)

- `run/`
- per-run outputs: resolved config, logs, checkpoints, metrics

- `pretrained_weights/`
- local pretrained artifacts checked before remote download

## Entry Points

- `train.py`
- primary training/validation/test CLI
- loads `.env` at startup via `load_dotenv(override=False)` from python-dotenv
- then resolves layered config and launches trainer

- `main.py`
- minimal alternate entrypoint, also loads `.env`

## Runtime Env Flow

1. process starts in `train.py` or `main.py`
2. `load_dotenv(override=False)` loads `.env` (if present) into `os.environ`
3. child processes inherit environment variables automatically
4. IO helpers in `training.utils.io` resolve S3 clients per call and cache auto-created clients per thread/process when caller client is not provided

Important: do not defer `.env` loading to deep utility code if subprocesses need the values.

## Config Merge Precedence

Later layers override earlier layers.

1. `cfg/default.yaml`
2. `cfg/tasks/<task>.yaml`
3. `cfg/experiments/<task>/<experiment>.yaml`
4. CLI `--set key=value`

Dataset YAML (`cfg/datasets/*.yaml`) is resolved from top-level `data` reference and merged within layer processing; CLI `data=...` remains highest precedence.

## Task and Trainer Boundary

- trainer owns loop, optimization, callbacks, distributed runtime
- task owns `build_model`, `build_losses`, `build_metrics`, and step logic
- avoid task-specific branching inside trainer

## S3/Ceph I/O Pattern

Preferred pattern for high-throughput datagen/workers:

1. initialize one client per worker/thread context in your caller code
2. pass it explicitly as `s3_client=` into `read_file/read_image/list_files`
3. avoid per-sample client creation

Fallback behavior: `io.py` can create cached clients automatically when `s3_client` is not passed.

### IO Signature Cross-Reference

- `list_files(directory: str | None = None, /, recursive: bool = False, count: int = 1000, verbose: bool = False, s3_client: Any = None, **kwargs) -> Iterator[str]`
- `read_file(filepath: str | None = None, verbose: bool = False, s3_client: Any = None, as_stream: bool = False, **kwargs) -> bytes | Any | None`
- `read_image(filepath: str | None = None, as_pillow: bool = False, cv2_imdecode_mode: int = -1, verbose: bool = False, s3_client: Any = None, **kwargs) -> Any | None`
- Source: [training/utils/io.py](../training/utils/io.py#L66)

## Run Naming

Run name priority:

1. CLI `--run-name`
2. config `run.name` (or top-level `name`)
3. timestamp fallback (`YYYYMMDD_HHMMSS`)

## Quick Start

1. copy `sample.env` to `.env` and set credentials/endpoints
2. choose task/data/experiment config
3. run training with optional overrides

Example:

`python train.py --mode train --task classification --dataset imagenet --experiment baseline`
