from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from training.utils.logger import get_logger


LOGGER = get_logger("training.utils.checkpoint")


class CheckpointManager:
    """Unified checkpoint manager for saving and loading training state."""

    @staticmethod
    def find_latest_checkpoint(run_dir: str | Path) -> Path | None:
        """Find latest checkpoint from run directory with multi-pattern support."""
        run_path = Path(run_dir)
        candidates = sorted(run_path.glob("ckpt_epoch_*.pth"))
        if not candidates:
            candidates = sorted(run_path.glob("ckpt_epoch_*.pkl"))
        if not candidates:
            candidates = sorted(run_path.glob("checkpoints/epoch_*.pt"))
        if not candidates:
            best = run_path / "checkpoints" / "best.pt"
            if best.exists():
                return best
        return candidates[-1] if candidates else None

    @staticmethod
    def save_checkpoint(
        path: Path | str,
        model: Any,
        optimizer: Any = None,
        scheduler: Any = None,
        scaler: Any = None,
        config: Any = None,
        epoch: int = 0,
        step: int = 0,
        global_step: int = 0,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        """Save checkpoint with unified schema (used by CheckpointCallback)."""
        checkpoint = {
            "model": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "config": config,
            "epoch": epoch,
            "step": step,
            "global_step": global_step,
            "metrics": metrics or {},
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)

    @staticmethod
    def load_checkpoint(
        run_dir: str | Path,
        model: Any,
        optimizer: Any = None,
        scheduler: Any = None,
        device: str | torch.device = "cpu",
        checkpoint_path: str | Path | None = None,
        strict: bool = False,
        rank: int = 0,
    ) -> tuple[list[str], list[str], int, dict[str, Any]]:
        """Load model and optional optimizer/scheduler state from checkpoint."""
        checkpoint_ref = Path(checkpoint_path) if checkpoint_path else None
        path = (
            checkpoint_ref
            if checkpoint_ref and checkpoint_ref.is_file()
            else CheckpointManager.find_latest_checkpoint(run_dir)
        )

        if path is None or not path.is_file():
            if rank == 0:
                LOGGER.info("No checkpoint found to resume from.")
            return [], [], 0, {}

        if rank == 0:
            LOGGER.info("Loading checkpoint: %s", path)

        checkpoint = torch.load(path, map_location=device)
        state_dict = checkpoint.get("model", checkpoint.get("model_state", checkpoint))
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)

        optimizer_state = checkpoint.get("optimizer", checkpoint.get("optimizer_state"))
        scheduler_state = checkpoint.get("scheduler", checkpoint.get("scheduler_state"))

        if optimizer is not None and optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        if scheduler is not None and scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)

        epoch = int(checkpoint.get("epoch", 0))
        extra = checkpoint.get("extra", {})
        return list(missing_keys), list(unexpected_keys), epoch + 1, extra


# Backward compatibility: module-level functions delegate to CheckpointManager
def find_latest_checkpoint(run_dir: str | Path) -> Path | None:
    """Find latest checkpoint. Delegates to CheckpointManager."""
    return CheckpointManager.find_latest_checkpoint(run_dir)


def load_checkpoint(
    run_dir: str | Path,
    model: Any,
    optimizer: Any = None,
    scheduler: Any = None,
    device: str | torch.device = "cpu",
    checkpoint_path: str | Path | None = None,
    strict: bool = False,
    rank: int = 0,
) -> tuple[list[str], list[str], int, dict[str, Any]]:
    """Load model and optional optimizer/scheduler state from checkpoint. Delegates to CheckpointManager."""
    return CheckpointManager.load_checkpoint(
        run_dir, model, optimizer, scheduler, device, checkpoint_path, strict, rank
    )
