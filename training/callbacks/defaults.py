from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from training.callbacks.base import Callback
from training.engine.state import TrainState
from training.utils.checkpoint import CheckpointManager


class CheckpointCallback(Callback):
    """Save model, optimizer, scheduler, and metrics to checkpoint directory.

    Checkpoints are the single source of truth for training state.
    Use training.utils.checkpoint.load_checkpoint to resume from saved state.
    """

    def __init__(self, monitor: str = "val_loss", mode: str = "min", save_freq: int = 1):
        self.monitor = monitor
        self.mode = mode
        self.save_freq = save_freq
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.best_epoch = None

    def _is_improved(self, value: float) -> bool:
        if self.mode == "min":
            return value < self.best_value
        return value > self.best_value

    def on_epoch_end(self, state: TrainState) -> None:
        runtime = state.runtime
        if runtime is not None and not getattr(runtime, "is_main_process", True):
            return
        if state.run_dir is None:
            return

        run_dir = Path(state.run_dir)
        ckpt_dir = run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save checkpoint based on save frequency
        if state.epoch % self.save_freq == 0:
            epoch_path = ckpt_dir / f"epoch_{state.epoch:03d}.pt"
            CheckpointManager.save_checkpoint(
                epoch_path,
                model=state.model,
                optimizer=state.optimizer,
                scheduler=state.scheduler,
                scaler=state.scaler,
                config=state.cfg,
                epoch=state.epoch,
                step=state.step,
                global_step=state.global_step,
                metrics=state.metrics,
            )

        # Save best checkpoint if metric improves
        value = float(state.metrics.get(self.monitor, float("inf")))
        if self._is_improved(value):
            self.best_value = value
            self.best_epoch = state.epoch
            best_path = ckpt_dir / "best.pt"
            CheckpointManager.save_checkpoint(
                best_path,
                model=state.model,
                optimizer=state.optimizer,
                scheduler=state.scheduler,
                scaler=state.scaler,
                config=state.cfg,
                epoch=state.epoch,
                step=state.step,
                global_step=state.global_step,
                metrics=state.metrics,
            )


class MetricsCallback(Callback):
    def __init__(self):
        self.history: dict[str, list[float]] = {}

    def on_epoch_end(self, state: TrainState) -> None:
        for key, value in state.metrics.items():
            self.history.setdefault(key, []).append(float(value))

    def on_train_end(self, state: TrainState) -> None:
        runtime = state.runtime
        if runtime is not None and not getattr(runtime, "is_main_process", True):
            return
        if state.run_dir is None:
            return

        run_dir = Path(state.run_dir)
        metrics_path = run_dir / "metrics.json"
        metrics_path.write_text(json.dumps(self.history, indent=2), encoding="utf-8")
