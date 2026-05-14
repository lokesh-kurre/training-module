from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import torch

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
    def __init__(
        self,
        write_json: bool = True,
        write_csv: bool = True,
        json_filename: str = "metrics.json",
        csv_filename: str = "metrics.csv",
    ):
        self.history: dict[str, list[float]] = {}
        self.write_json = bool(write_json)
        self.write_csv = bool(write_csv)
        self.json_filename = str(json_filename)
        self.csv_filename = str(csv_filename)

    def _is_main_process(self, state: TrainState) -> bool:
        runtime = state.runtime
        return runtime is None or getattr(runtime, "is_main_process", True)

    def _write_json(self, run_dir: Path) -> None:
        if not self.write_json:
            return
        metrics_path = run_dir / self.json_filename
        metrics_path.write_text(json.dumps(self.history, indent=2), encoding="utf-8")

    def _write_csv(self, run_dir: Path) -> None:
        if not self.write_csv:
            return

        metrics_path = run_dir / self.csv_filename
        metric_names = sorted(self.history.keys())
        max_epochs = max((len(values) for values in self.history.values()), default=0)

        with metrics_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["epoch", *metric_names])
            for epoch_idx in range(max_epochs):
                row: list[float | int | str] = [epoch_idx + 1]
                for name in metric_names:
                    values = self.history.get(name, [])
                    row.append(values[epoch_idx] if epoch_idx < len(values) else "")
                writer.writerow(row)

    def _persist_metrics(self, state: TrainState) -> None:
        if not self._is_main_process(state):
            return
        if state.run_dir is None:
            return

        run_dir = Path(state.run_dir)
        self._write_json(run_dir)
        self._write_csv(run_dir)

    def on_epoch_end(self, state: TrainState) -> None:
        for key, value in state.metrics.items():
            self.history.setdefault(key, []).append(float(value))
        self._persist_metrics(state)

    def on_train_end(self, state: TrainState) -> None:
        # Keep final flush for robustness in case training exits after a final epoch callback.
        self._persist_metrics(state)


class GANSampleGridCallback(Callback):
    """Persist generator sample grids with optional headers for style transfer.
    
    Supports layouts like:
    - Simple grid: generated samples
    - Style transfer grid: first row=content, first col=style, rest=generated outputs
    
    Auto-detects grayscale vs color and applies proper colormap.
    """

    def __init__(
        self,
        save_freq: int = 1,
        num_samples: int = 16,
        nrow: int = 4,
        normalize: bool = True,
        with_headers: bool = False,
        row_header_prefix: str = "Style",
        col_header_prefix: str = "Content",
    ):
        self.save_freq = max(1, int(save_freq))
        self.num_samples = max(1, int(num_samples))
        self.nrow = max(1, int(nrow))
        self.normalize = bool(normalize)
        self.with_headers = bool(with_headers)
        self.row_header_prefix = str(row_header_prefix)
        self.col_header_prefix = str(col_header_prefix)

    def _is_grayscale(self, images: torch.Tensor) -> bool:
        """Detect if image is grayscale (C=1) or color (C=3)."""
        if images.ndim == 4:
            return images.shape[1] == 1
        return False

    def _save_grid_with_pil(self, samples: torch.Tensor, out_path: Path, normalize: bool) -> None:
        """Fallback grid saving using PIL with proper grayscale/color handling."""
        import numpy as np
        from PIL import Image

        samples_np = samples.detach().cpu().numpy()
        batch_size = samples_np.shape[0]
        is_gray = samples_np.shape[1] == 1

        # Convert to uint8
        if normalize:
            samples_np = ((np.clip(samples_np, -1, 1) + 1.0) * 127.5).astype(np.uint8)
        else:
            samples_np = np.clip(samples_np, 0, 255).astype(np.uint8)

        # Build grid
        tile_h, tile_w = samples_np.shape[2], samples_np.shape[3]
        ncol = self.nrow
        nrow = (batch_size + ncol - 1) // ncol
        grid_h = tile_h * nrow
        grid_w = tile_w * ncol

        if is_gray:
            grid_img = Image.new("L", (grid_w, grid_h), 255)
        else:
            grid_img = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))

        for idx, sample in enumerate(samples_np):
            row = idx // ncol
            col = idx % ncol
            x = col * tile_w
            y = row * tile_h

            if is_gray:
                sample_img = Image.fromarray(sample[0], mode="L")
            else:
                sample_img = Image.fromarray(sample.transpose(1, 2, 0), mode="RGB")
            grid_img.paste(sample_img, (x, y))

        grid_img.save(out_path)

    def _to_plot_array(self, image: torch.Tensor) -> tuple[Any, str]:
        """Convert CHW image to 2D array and cmap for matplotlib plotting.

        Rule requested by user:
        - grayscale image -> cmap='gray'
        - color image -> cmap='viridis'
        """
        if image.ndim != 3:
            raise ValueError("Expected CHW tensor image")

        img = image.detach().cpu().float()
        if self.normalize:
            img = (img.clamp(-1.0, 1.0) + 1.0) * 0.5
        else:
            img = img.clamp(0.0, 1.0)

        if img.shape[0] == 1:
            return img[0].numpy(), "gray"

        # For multi-channel inputs, plot luminance-like map with viridis colormap.
        return img[:3].mean(dim=0).numpy(), "viridis"

    def _save_grid_with_matplotlib(
        self,
        samples: torch.Tensor,
        out_path: Path,
        content_images: torch.Tensor | None = None,
        style_images: torch.Tensor | None = None,
    ) -> None:
        import matplotlib.pyplot as plt

        batch = samples.detach().cpu()
        if batch.ndim != 4:
            raise ValueError("Expected NCHW samples")

        inner_cols = self.nrow
        inner_rows = int(math.ceil(batch.shape[0] / max(1, inner_cols)))

        use_headers = (
            self.with_headers
            and content_images is not None
            and style_images is not None
            and getattr(content_images, "ndim", 0) == 4
            and getattr(style_images, "ndim", 0) == 4
        )

        grid_rows = inner_rows + 1 if use_headers else inner_rows
        grid_cols = inner_cols + 1 if use_headers else inner_cols

        fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(2.2 * grid_cols, 2.2 * grid_rows), squeeze=False)

        for r in range(grid_rows):
            for c in range(grid_cols):
                ax = axes[r][c]
                ax.set_xticks([])
                ax.set_yticks([])

                if use_headers and r == 0 and c == 0:
                    ax.axis("off")
                    continue

                if use_headers and r == 0 and c > 0:
                    idx = (c - 1) % max(1, content_images.shape[0])
                    arr, cmap = self._to_plot_array(content_images[idx].detach().cpu())
                    ax.imshow(arr, cmap=cmap)
                    ax.set_title(f"{self.col_header_prefix} {c}", fontsize=9)
                    continue

                if use_headers and c == 0 and r > 0:
                    idx = (r - 1) % max(1, style_images.shape[0])
                    arr, cmap = self._to_plot_array(style_images[idx].detach().cpu())
                    ax.imshow(arr, cmap=cmap)
                    ax.set_ylabel(f"{self.row_header_prefix} {r}", fontsize=9)
                    continue

                rr = r - 1 if use_headers else r
                cc = c - 1 if use_headers else c
                sample_idx = rr * inner_cols + cc
                if sample_idx >= batch.shape[0]:
                    ax.axis("off")
                    continue

                arr, cmap = self._to_plot_array(batch[sample_idx])
                ax.imshow(arr, cmap=cmap)

        fig.tight_layout()
        fig.savefig(out_path, dpi=140)
        plt.close(fig)

    def on_epoch_end(self, state: TrainState) -> None:
        runtime = state.runtime
        if runtime is not None and not getattr(runtime, "is_main_process", True):
            return
        if state.run_dir is None:
            return
        if state.epoch % self.save_freq != 0:
            return

        model = state.model.module if hasattr(state.model, "module") else state.model
        generate = getattr(model, "generate", None)
        if not callable(generate):
            return

        device = next(model.parameters()).device
        with torch.no_grad():
            samples = generate(self.num_samples, device=device)

        if not torch.is_tensor(samples) or samples.ndim != 4:
            return

        samples_dir = Path(state.run_dir) / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        out_path = samples_dir / f"epoch_{state.epoch:03d}.png"

        content_images = None
        style_images = None
        if self.with_headers and isinstance(state.batch, (list, tuple)) and len(state.batch) >= 2:
            if torch.is_tensor(state.batch[0]) and getattr(state.batch[0], "ndim", 0) == 4:
                content_images = state.batch[0].detach().cpu()
            if torch.is_tensor(state.batch[1]) and getattr(state.batch[1], "ndim", 0) == 4:
                style_images = state.batch[1].detach().cpu()

        try:
            self._save_grid_with_matplotlib(samples, out_path, content_images=content_images, style_images=style_images)
        except Exception:
            self._save_grid_with_pil(samples, out_path, self.normalize)
