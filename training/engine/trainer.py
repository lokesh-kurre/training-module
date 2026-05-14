from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

import training.tasks  # noqa: F401 - ensures built-in tasks register themselves

from training.callbacks import CallbackManager, build_callbacks
from training.config import dump_yaml
from training.engine.runtime import (
    build_distributed_runtime,
    reduce_scalar,
    setup_distributed,
    teardown_distributed,
    wrap_model_for_runtime,
)
from training.engine.state import TrainState
from training.tasks.base import TaskStepOutput
from training.tasks.registry import get_task_class
from training.utils.importer import get_obj_by_name
from training.utils.input_spec import out_size_to_chw, resolve_input_spec
from training.utils.checkpoint import load_checkpoint
from training.utils.logger import configure_run_error_log, get_logger
from training.utils.summary import print_module_summary, summarize_training_data


@dataclass
class Trainer:
    cfg: dict[str, Any]
    project_root: Path
    logger: Any = field(init=False)

    def __post_init__(self) -> None:
        self.logger = get_logger("training.engine.trainer")

    def _first_accuracy_key(self, metrics: dict[str, float]) -> str | None:
        for key in metrics:
            if "acc" in key.lower():
                return key
        return None

    def _format_progress_bar(self, current: int, total: int | None, width: int = 28) -> str:
        if total is None or total <= 0:
            return "[" + ("#" * (width // 2)) + ("-" * (width - (width // 2))) + "]"
        ratio = max(0.0, min(1.0, float(current) / float(total)))
        filled = int(width * ratio)
        return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"

    def _log_level(self) -> str:
        level = self.cfg.get("run", {}).get("log_level", self.cfg.get("trainer", {}).get("log_level", "info"))
        return str(level).strip().lower()

    def _should_log(self, level: str) -> bool:
        order = {"debug": 10, "info": 20}
        configured = order.get(self._log_level(), 20)
        current = order.get(level, 20)
        return current >= configured

    def _append_log(self, run_dir: Path, level: str, message: str, *, to_console: bool = False) -> None:
        normalized_level = str(level).strip().lower()
        if not self._should_log(normalized_level):
            return

        message_text = str(message)
        # Never persist carriage-return progress updates to files.
        if "\r" in message_text:
            message_text = message_text.split("\r")[-1]
        if "\r" in str(message) or (message_text.startswith("Epoch ") and "[" in message_text and "]" in message_text):
            return

        # Keep train log file concise: only debug/info go to log.txt.
        if normalized_level in {"debug", "info"}:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            line = f"[{timestamp}] [{normalized_level.upper()}] {message_text}"
            try:
                (run_dir / "log.txt").open("a", encoding="utf-8").write(line + "\n")
            except Exception:
                pass

        if to_console and self._is_rank0(getattr(self, "_runtime_for_logging", None) or type("_R", (), {"is_main_process": True})()):
            print(message_text)
        if self._is_rank0(getattr(self, "_runtime_for_logging", None) or type("_R", (), {"is_main_process": True})()):
            if normalized_level == "debug":
                self.logger.debug(message_text)
            elif normalized_level == "warning":
                self.logger.warning(message_text)
            elif normalized_level in {"error", "exception", "critical"}:
                self.logger.error(message_text)
            else:
                self.logger.info(message_text)

    def resolve_run_dir(self, run_name: str | None = None) -> Path:
        root = self.project_root / self.cfg.get("run", {}).get("root_dir", "run")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        configured_name = self.cfg.get("run", {}).get("name")
        resolved_name = run_name or configured_name or stamp
        directory = root / str(resolved_name)
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def save_resolved_config(self, run_dir: Path) -> None:
        if self.cfg.get("run", {}).get("save_resolved_config", True):
            dump_yaml(self.cfg, run_dir / "config_resolved.yaml")

    def _is_rank0(self, runtime: Any) -> bool:
        return bool(getattr(runtime, "is_main_process", True))

    def _seed_everything(self, runtime: Any) -> None:
        seed_value = self.cfg.get("seed", None)
        if seed_value is None:
            return
        try:
            base_seed = int(seed_value)
        except Exception:
            return

        rank = int(getattr(runtime, "rank", 0))
        final_seed = base_seed + rank
        random.seed(final_seed)
        np.random.seed(final_seed)

        try:
            import torch

            torch.manual_seed(final_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(final_seed)
        except Exception:
            return

    def _move_batch_to_device(self, batch: Any, device: Any) -> Any:
        """Move a batch of tensors to the target device recursively."""
        import torch

        if torch.is_tensor(batch):
            return batch.to(device)
        if isinstance(batch, dict):
            return {key: self._move_batch_to_device(value, device) for key, value in batch.items()}
        if isinstance(batch, (list, tuple)):
            converted = [self._move_batch_to_device(item, device) for item in batch]
            return type(batch)(converted)
        return batch

    def _get_device(self) -> Any:
        """Resolve device from config. Returns torch.device."""
        import torch

        device_str = self.cfg.get("trainer", {}).get("device", "cpu")
        if device_str == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _train_epoch(
        self,
        state: TrainState,
        task: Any,
        train_loader: Any,
        optimizer: Any,
        device: Any,
        runtime: Any,
        callbacks: CallbackManager,
        amp_enabled: bool = False,
        scaler: Any = None,
        optimizers_dict: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        """Train one epoch and return metrics.
        
        Supports both single-optimizer (classification) and multi-optimizer (GAN) tasks.
        """
        import torch

        task.model.train()
        
        # Branch: single-optimizer or multi-optimizer (GAN) path
        if optimizers_dict is not None:
            return self._train_epoch_multi_optimizer(state, task, train_loader, device, runtime, callbacks, optimizers_dict, amp_enabled, scaler)
        else:
            return self._train_epoch_single_optimizer(state, task, train_loader, optimizer, device, runtime, callbacks, amp_enabled, scaler)
    
    def _train_epoch_single_optimizer(
        self,
        state: TrainState,
        task: Any,
        train_loader: Any,
        optimizer: Any,
        device: Any,
        runtime: Any,
        callbacks: CallbackManager,
        amp_enabled: bool = False,
        scaler: Any = None,
    ) -> dict[str, float]:
        """Train one epoch with single optimizer (classification, etc)."""
        import torch

        task.model.train()
        totals: dict[str, float] = {}
        total_loss = 0.0
        num_batches = 0
        trainer_cfg = self.cfg.get("trainer", {})
        progress_bar_enabled = bool(trainer_cfg.get("show_progress_bar", True))
        heartbeat_enabled = bool(trainer_cfg.get("log_heartbeat", True))
        heartbeat_minutes = float(trainer_cfg.get("heartbeat_minutes", 15.0))
        heartbeat_seconds = max(0.0, heartbeat_minutes * 60.0)
        max_steps_raw = trainer_cfg.get("steps_per_epoch", trainer_cfg.get("max_steps_per_epoch"))
        max_steps: int | None = None
        if max_steps_raw is not None:
            max_steps = int(max_steps_raw)
            if max_steps <= 0:
                max_steps = None
        timing_enabled = bool(trainer_cfg.get("timing_enabled", trainer_cfg.get("profile_timing", True)))
        timing_log_level = str(trainer_cfg.get("timing_log_level", "info")).strip().lower()
        batch_metric_logging = bool(trainer_cfg.get("log_batch_metrics", False))
        progress_bar_enabled = bool(trainer_cfg.get("show_progress_bar", True))
        heartbeat_enabled = bool(trainer_cfg.get("log_heartbeat", True))
        heartbeat_minutes = float(trainer_cfg.get("heartbeat_minutes", 15.0))
        heartbeat_seconds = max(0.0, heartbeat_minutes * 60.0)
        timing = {
            "data_wait": 0.0,
            "move_to_device": 0.0,
            "zero_grad": 0.0,
            "forward_loss": 0.0,
            "backward": 0.0,
            "optimizer_step": 0.0,
            "callbacks": 0.0,
        }
        epoch_wall_start = time.perf_counter()
        try:
            train_loader_len = len(train_loader)
        except TypeError:
            train_loader_len = None
        if max_steps is None and train_loader_len is not None:
            # If user did not provide steps_per_epoch, use inferred steps from sample_count/batch_size.
            max_steps = int(train_loader_len)

        loop_total = max_steps if max_steps is not None else train_loader_len

        train_iter = iter(train_loader)
        batch_idx = 0
        last_heartbeat = time.perf_counter()
        running_acc_key: str | None = None

        while True:
            if max_steps is not None and batch_idx >= max_steps:
                break

            t0 = time.perf_counter()
            try:
                batch = next(train_iter)
            except StopIteration:
                break
            timing["data_wait"] += time.perf_counter() - t0

            t1 = time.perf_counter()
            batch = self._move_batch_to_device(batch, device)
            timing["move_to_device"] += time.perf_counter() - t1

            t2 = time.perf_counter()
            optimizer.zero_grad()
            timing["zero_grad"] += time.perf_counter() - t2

            t3 = time.perf_counter()
            if amp_enabled:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    step = task.training_step(batch)
            else:
                step = task.training_step(batch)
            timing["forward_loss"] += time.perf_counter() - t3

            t4 = time.perf_counter()
            if amp_enabled and scaler is not None:
                scaler.scale(step.loss).backward()
            else:
                step.loss.backward()
            timing["backward"] += time.perf_counter() - t4

            t5 = time.perf_counter()
            if amp_enabled and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            timing["optimizer_step"] += time.perf_counter() - t5

            batch_idx += 1

            state.batch = batch
            state.outputs = {"loss": float(step.loss.item()), **{k: float(v) for k, v in step.metrics.items()}}
            state.step = batch_idx
            state.global_step += 1
            t6 = time.perf_counter()
            callbacks.on_batch_end(state)
            timing["callbacks"] += time.perf_counter() - t6

            total_loss += float(step.loss.item())
            for metric_name, metric_value in step.metrics.items():
                totals[metric_name] = totals.get(metric_name, 0.0) + float(metric_value)
            num_batches += 1

            if running_acc_key is None:
                running_acc_key = self._first_accuracy_key(totals)

            running_loss = total_loss / max(1, num_batches)
            running_acc = None
            if running_acc_key is not None:
                running_acc = totals[running_acc_key] / max(1, num_batches)

            lr_now = None
            if optimizer is not None and getattr(optimizer, "param_groups", None):
                try:
                    lr_now = float(optimizer.param_groups[0].get("lr"))
                except Exception:
                    lr_now = None

            if self._is_rank0(runtime):
                if progress_bar_enabled:
                    step_text = f"{batch_idx}/{loop_total}" if loop_total is not None else f"{batch_idx}/?"
                    bar = self._format_progress_bar(batch_idx, loop_total)
                    line = f"\rEpoch {state.epoch + 1} {bar} {step_text} loss={running_loss:.4f}"
                    if running_acc is not None:
                        line += f" acc={running_acc:.4f}"
                    if lr_now is not None:
                        line += f" lr={lr_now:.6f}"
                    print(line, end="", flush=True)

                if batch_metric_logging:
                    if loop_total is not None:
                        if batch_idx % max(1, loop_total // 5) == 0 or batch_idx == loop_total:
                            msg = f"  Batch {batch_idx}/{loop_total}: loss={step.loss.item():.4f}"
                            print(msg)
                            self._append_log(state.run_dir, "debug", msg)
                    elif batch_idx % 10 == 0:
                        msg = f"  Batch {batch_idx}: loss={step.loss.item():.4f}"
                        print(msg)
                        self._append_log(state.run_dir, "debug", msg)

                if heartbeat_enabled and heartbeat_seconds > 0.0:
                    now = time.perf_counter()
                    if now - last_heartbeat >= heartbeat_seconds:
                        if progress_bar_enabled:
                            print("")
                        progress = f"{batch_idx}/{loop_total}" if loop_total is not None else str(batch_idx)
                        hb_msg = f"Heartbeat: epoch={state.epoch + 1}, step={progress}, running_loss={running_loss:.4f}"
                        if running_acc is not None:
                            hb_msg += f", running_acc={running_acc:.4f}"
                        if lr_now is not None:
                            hb_msg += f", lr={lr_now:.6f}"
                        print(hb_msg)
                        self._append_log(state.run_dir, "info", hb_msg)
                        last_heartbeat = now

        if self._is_rank0(runtime) and progress_bar_enabled and num_batches > 0:
            print("")

        if timing_enabled and self._is_rank0(runtime) and num_batches > 0:
            epoch_wall_seconds = max(1e-9, time.perf_counter() - epoch_wall_start)
            stage_total = sum(timing.values())
            debug_batch_size = int(self.cfg.get("dataloader", {}).get("batch_size", 1))
            approx_samples_per_sec = (num_batches * max(1, debug_batch_size)) / epoch_wall_seconds
            print("\n  Timing Breakdown (train epoch):")
            print(f"    steps: {num_batches}, batch_size: {debug_batch_size}, approx throughput: {approx_samples_per_sec:.2f} samples/s")
            print(f"    wall time: {epoch_wall_seconds:.3f}s")
            self._append_log(state.run_dir, timing_log_level, "Timing Breakdown (train epoch):")
            self._append_log(
                state.run_dir,
                timing_log_level,
                f"steps={num_batches}, batch_size={debug_batch_size}, approx_throughput={approx_samples_per_sec:.2f} samples/s, wall_time={epoch_wall_seconds:.3f}s",
            )
            for key, value in timing.items():
                pct = (100.0 * value / stage_total) if stage_total > 0 else 0.0
                avg_ms = (1000.0 * value / num_batches)
                print(f"    {key:>14}: {value:7.3f}s  ({pct:5.1f}%)  avg/batch={avg_ms:7.2f} ms")
                self._append_log(
                    state.run_dir,
                    timing_log_level,
                    f"timing.{key}: total={value:.3f}s pct={pct:.1f}% avg_batch_ms={avg_ms:.2f}",
                )

        total_loss = reduce_scalar(total_loss, runtime)
        totals = {name: reduce_scalar(value, runtime) for name, value in totals.items()}
        num_batches = max(1.0, reduce_scalar(float(num_batches), runtime))

        result = {"train_loss": total_loss / num_batches}
        for metric_name, total_value in totals.items():
            result[f"train_{metric_name}"] = total_value / num_batches
        return result
    
    def _train_epoch_multi_optimizer(
        self,
        state: TrainState,
        task: Any,
        train_loader: Any,
        device: Any,
        runtime: Any,
        callbacks: CallbackManager,
        optimizers_dict: dict[str, Any],
        amp_enabled: bool = False,
        scaler: Any = None,
    ) -> dict[str, float]:
        """Train one epoch with multiple optimizers (GANs with alternating D/G updates)."""
        import torch

        task.model.train()
        totals: dict[str, float] = {}
        total_loss = 0.0
        num_batches = 0
        trainer_cfg = self.cfg.get("trainer", {})
        progress_bar_enabled = bool(trainer_cfg.get("show_progress_bar", True))
        heartbeat_enabled = bool(trainer_cfg.get("log_heartbeat", True))
        heartbeat_minutes = float(trainer_cfg.get("heartbeat_minutes", 15.0))
        heartbeat_seconds = max(0.0, heartbeat_minutes * 60.0)
        max_steps_raw = trainer_cfg.get("steps_per_epoch", trainer_cfg.get("max_steps_per_epoch"))
        max_steps: int | None = None
        if max_steps_raw is not None:
            max_steps = int(max_steps_raw)
            if max_steps <= 0:
                max_steps = None

        try:
            train_loader_len = len(train_loader)
        except TypeError:
            train_loader_len = None
        if max_steps is None and train_loader_len is not None:
            max_steps = int(train_loader_len)

        loop_total = max_steps if max_steps is not None else train_loader_len
        train_iter = iter(train_loader)
        batch_idx = 0
        last_heartbeat = time.perf_counter()
        running_acc_key: str | None = None
        d_step_fn = getattr(task, "training_step_discriminator", None)
        g_step_fn = getattr(task, "training_step_generator", None)

        while True:
            if max_steps is not None and batch_idx >= max_steps:
                break

            try:
                batch = next(train_iter)
            except StopIteration:
                break

            batch = self._move_batch_to_device(batch, device)

            opt_d = optimizers_dict.get("discriminator")
            opt_g = optimizers_dict.get("generator")

            # Discriminator step
            if callable(d_step_fn):
                if amp_enabled:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        step_d = d_step_fn(batch)
                else:
                    step_d = d_step_fn(batch)
            else:
                if amp_enabled:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        step_d = task.training_step(batch)
                else:
                    step_d = task.training_step(batch)

            if opt_d is not None:
                opt_d.zero_grad()
                if amp_enabled and scaler is not None:
                    scaler.scale(step_d.loss).backward()
                    scaler.step(opt_d)
                else:
                    step_d.loss.backward()
                    opt_d.step()

            # Generator step
            if callable(g_step_fn):
                if amp_enabled:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        step_g = g_step_fn(batch)
                else:
                    step_g = g_step_fn(batch)
            else:
                if amp_enabled:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        step_g = task.training_step(batch)
                else:
                    step_g = task.training_step(batch)

            # Ensure discriminator grads do not accumulate during generator backward.
            if opt_d is not None:
                opt_d.zero_grad()

            if opt_g is not None:
                opt_g.zero_grad()
                if amp_enabled and scaler is not None:
                    scaler.scale(step_g.loss).backward()
                    scaler.step(opt_g)
                else:
                    step_g.loss.backward()
                    opt_g.step()

            if amp_enabled and scaler is not None:
                scaler.update()

            step = step_g
            merged_metrics = dict(step_d.metrics)
            merged_metrics.update(step_g.metrics)
            merged_metrics["d_step_loss"] = float(step_d.loss.item())
            merged_metrics["g_step_loss"] = float(step_g.loss.item())
            merged_loss = float(step_d.loss.item()) + float(step_g.loss.item())

            batch_idx += 1
            state.batch = batch
            state.outputs = {"loss": merged_loss, **{k: float(v) for k, v in merged_metrics.items()}}
            state.step = batch_idx
            state.global_step += 1
            callbacks.on_batch_end(state)

            total_loss += merged_loss
            for metric_name, metric_value in merged_metrics.items():
                totals[metric_name] = totals.get(metric_name, 0.0) + float(metric_value)
            num_batches += 1

            if running_acc_key is None:
                running_acc_key = self._first_accuracy_key(totals)

            running_loss = total_loss / max(1, num_batches)
            running_acc = None
            if running_acc_key is not None:
                running_acc = totals[running_acc_key] / max(1, num_batches)

            lr_now = None
            opt_for_lr = opt_g if opt_g is not None else opt_d
            if opt_for_lr is not None and getattr(opt_for_lr, "param_groups", None):
                try:
                    lr_now = float(opt_for_lr.param_groups[0].get("lr"))
                except Exception:
                    lr_now = None

            if self._is_rank0(runtime):
                if progress_bar_enabled:
                    step_text = f"{batch_idx}/{loop_total}" if loop_total is not None else f"{batch_idx}/?"
                    bar = self._format_progress_bar(batch_idx, loop_total)
                    line = f"\rEpoch {state.epoch + 1} {bar} {step_text} loss={running_loss:.4f}"
                    if running_acc is not None:
                        line += f" acc={running_acc:.4f}"
                    if lr_now is not None:
                        line += f" lr={lr_now:.6f}"
                    print(line, end="", flush=True)

                if heartbeat_enabled and heartbeat_seconds > 0.0:
                    now = time.perf_counter()
                    if now - last_heartbeat >= heartbeat_seconds:
                        if progress_bar_enabled:
                            print("")
                        progress = f"{batch_idx}/{loop_total}" if loop_total is not None else str(batch_idx)
                        hb_msg = f"Heartbeat: epoch={state.epoch + 1}, step={progress}, running_loss={running_loss:.4f}"
                        if running_acc is not None:
                            hb_msg += f", running_acc={running_acc:.4f}"
                        if lr_now is not None:
                            hb_msg += f", lr={lr_now:.6f}"
                        print(hb_msg)
                        self._append_log(state.run_dir, "info", hb_msg)
                        last_heartbeat = now

        if self._is_rank0(runtime) and progress_bar_enabled and num_batches > 0:
            print("")

        total_loss = reduce_scalar(total_loss, runtime)
        totals = {name: reduce_scalar(value, runtime) for name, value in totals.items()}
        num_batches = max(1.0, reduce_scalar(float(num_batches), runtime))

        result = {"train_loss": total_loss / num_batches}
        for metric_name, total_value in totals.items():
            result[f"train_{metric_name}"] = total_value / num_batches
        return result

    def _validate_epoch(self, task: Any, val_loader: Any, device: Any, runtime: Any) -> dict[str, float]:
        """Validate one epoch and return metrics."""
        import torch

        task.model.eval()
        total_loss = 0.0
        totals: dict[str, float] = {}
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = self._move_batch_to_device(batch, device)
                step: TaskStepOutput = task.validation_step(batch)

                total_loss += float(step.loss.item())
                for metric_name, metric_value in step.metrics.items():
                    totals[metric_name] = totals.get(metric_name, 0.0) + float(metric_value)
                num_batches += 1

        total_loss = reduce_scalar(total_loss, runtime)
        totals = {name: reduce_scalar(value, runtime) for name, value in totals.items()}
        num_batches = max(1.0, reduce_scalar(float(num_batches), runtime))

        result = {"val_loss": total_loss / num_batches}
        for metric_name, total_value in totals.items():
            result[f"val_{metric_name}"] = total_value / num_batches
        return result

    def _build_task_and_loaders(self, device: Any, runtime: Any) -> tuple[Any, Any, Any, Any, Any, Any]:
        optimizer_builder = get_obj_by_name(
            self.cfg.get("trainer", {}).get("optimizer_builder", "training.utils.optimizer.build_optimizer")
        )
        scheduler_builder = get_obj_by_name(
            self.cfg.get("trainer", {}).get("scheduler_builder", "training.utils.optimizer.build_scheduler")
        )
        # Always use string for task_name
        task_name = self.cfg.get("task", {}).get("name")
        if not isinstance(task_name, str):
            task_name = self.cfg.get("project", {}).get("task")
        task_name = str(task_name)
        task_cls = get_task_class(task_name)

        task = task_cls(self.cfg)
        task.model = task.model.to(device)
        task.model = wrap_model_for_runtime(task.model, runtime)

        optimizer = optimizer_builder(task.model, self.cfg)
        scheduler = scheduler_builder(optimizer, self.cfg)

        dataset_builder = get_obj_by_name(
            self.cfg.get("trainer", {}).get("dataset_builder", "training.data.datasets.factory.build_dataset_from_config")
        )
        dataloader_builder = get_obj_by_name(
            self.cfg.get("trainer", {}).get("dataloader_builder", "training.data.loaders.builder.build_dataloader")
        )
        runtime_rank = int(getattr(runtime, "rank", 0))
        runtime_world_size = int(getattr(runtime, "world_size", 1))

        try:
            train_dataset = dataset_builder(
                self.cfg,
                split="train",
                rank=runtime_rank,
                world_size=runtime_world_size,
                runtime=runtime,
            )
            val_dataset = dataset_builder(
                self.cfg,
                split="val",
                rank=runtime_rank,
                world_size=runtime_world_size,
                runtime=runtime,
            )
            test_dataset = None
            try:
                test_dataset = dataset_builder(
                    self.cfg,
                    split="test",
                    rank=runtime_rank,
                    world_size=runtime_world_size,
                    runtime=runtime,
                )
            except ValueError as exc:
                if "split='test'" not in str(exc):
                    raise
        except TypeError:
            # Backward compatibility for custom builders with legacy signature.
            train_dataset = dataset_builder(self.cfg, split="train")
            val_dataset = dataset_builder(self.cfg, split="val")
            try:
                test_dataset = dataset_builder(self.cfg, split="test")
            except ValueError as exc:
                if "split='test'" in str(exc):
                    test_dataset = None
                else:
                    raise

        train_loader = dataloader_builder(self.cfg, train_dataset, "train", runtime)
        val_loader = dataloader_builder(self.cfg, val_dataset, "val", runtime)
        test_loader = dataloader_builder(self.cfg, test_dataset, "test", runtime) if test_dataset is not None else None
        return task, optimizer, scheduler, train_loader, val_loader, test_loader

    def _resume_options(self) -> tuple[bool, str | None, bool, bool, bool]:
        resume_cfg = self.cfg.get("resume", {})
        if not isinstance(resume_cfg, dict):
            resume_cfg = {}

        model_cfg = self.cfg.get("model", {})
        if not isinstance(model_cfg, dict):
            model_cfg = {}

        checkpoint_path = resume_cfg.get("checkpoint_path")
        if checkpoint_path is None:
            checkpoint_path = model_cfg.get("weights_path")
        if checkpoint_path is None:
            checkpoint_path = model_cfg.get("weight_path")
        if checkpoint_path is None:
            checkpoint_path = model_cfg.get("wt_filepath")
        if checkpoint_path is None:
            checkpoint_path = model_cfg.get("weight_filepath")

        enabled = bool(resume_cfg.get("enabled", False))
        if checkpoint_path:
            enabled = True

        strict = bool(resume_cfg.get("strict", False))
        load_optimizer = bool(resume_cfg.get("load_optimizer", True))
        load_scheduler = bool(resume_cfg.get("load_scheduler", True))

        checkpoint_path_str = str(checkpoint_path) if checkpoint_path else None
        return enabled, checkpoint_path_str, strict, load_optimizer, load_scheduler

    def _run_eval_mode(self, split: str, run_name: str | None = None, dry_run: bool = False) -> Path | None:
        run_dir = self.resolve_run_dir(run_name=run_name)
        configure_run_error_log(run_dir)
        self.save_resolved_config(run_dir)

        if dry_run:
            self.logger.info("[dry-run] Resolved config saved to: %s", run_dir / "config_resolved.yaml")
            return run_dir

        task_name = str(self.cfg.get("project", {}).get("task"))
        runtime = build_distributed_runtime(self.cfg)
        setup_distributed(runtime)
        self._seed_everything(runtime)
        device = self._get_device()

        if self._is_rank0(runtime):
            print("\n" + "=" * 60)
            print(f"Starting {task_name.upper()} {split.upper()}")
            print(f"Run directory: {run_dir}")
            print(f"Device: {device}")
            print(f"Distributed: {runtime.is_distributed} (rank={runtime.rank}, world_size={runtime.world_size})")
            print("=" * 60 + "\n")

        try:
            task, optimizer, scheduler, _train_loader, val_loader, test_loader = self._build_task_and_loaders(device, runtime)
            _ = optimizer, scheduler

            loader = val_loader if split == "val" else test_loader
            if loader is None:
                raise ValueError(f"No '{split}' split configured for evaluation")
            metrics = self._validate_epoch(task, loader, device, runtime)

            if self._is_rank0(runtime):
                metric_str = ", ".join(f"{name}={value:.4f}" for name, value in metrics.items()) or "no metrics"
                print(f"{split.upper()} metrics: {metric_str}")
                output_path = run_dir / f"metrics_{split}.json"
                output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
                print(f"Saved {split} metrics to: {output_path}")

            return run_dir
        finally:
            teardown_distributed(runtime)

    def validate(self, run_name: str | None = None, dry_run: bool = False) -> Path | None:
        return self._run_eval_mode(split="val", run_name=run_name, dry_run=dry_run)

    def test(self, run_name: str | None = None, dry_run: bool = False) -> Path | None:
        return self._run_eval_mode(split="test", run_name=run_name, dry_run=dry_run)

    def fit(self, run_name: str | None = None, dry_run: bool = False) -> Path | None:
        run_dir = self.resolve_run_dir(run_name=run_name)
        configure_run_error_log(run_dir)
        self.save_resolved_config(run_dir)

        if dry_run:
            self.logger.info("[dry-run] Resolved config saved to: %s", run_dir / "config_resolved.yaml")
            return run_dir

        task_name = self.cfg.get("task", {}).get("name")
        if not isinstance(task_name, str):
            task_name = self.cfg.get("project", {}).get("task")
        task_name = str(task_name) if task_name is not None else "unknown"
        runtime = build_distributed_runtime(self.cfg)
        self._runtime_for_logging = runtime
        setup_distributed(runtime)
        self._seed_everything(runtime)
        device = self._get_device()
        num_epochs = int(self.cfg.get("trainer", {}).get("epochs", 5))
        mixed_precision = bool(self.cfg.get("trainer", {}).get("mixed_precision", False))
        amp_enabled = bool(mixed_precision and str(device).startswith("cuda"))

        scaler = None
        if amp_enabled:
            import torch

            scaler = torch.amp.GradScaler("cuda")

        if self._is_rank0(runtime):
            print("\n" + "=" * 60)
            print(f"Starting {task_name.upper()} Training")
            print(f"Run directory: {run_dir}")
            print(f"Device: {device}")
            print(f"Distributed: {runtime.is_distributed} (rank={runtime.rank}, world_size={runtime.world_size})")
            print("=" * 60 + "\n")
            self._append_log(run_dir, "info", f"Starting {task_name.upper()} Training")
            self._append_log(run_dir, "info", f"Run directory: {run_dir}")
            self._append_log(run_dir, "info", f"Device: {device}")
            self._append_log(run_dir, "info", f"Mixed precision (AMP): {amp_enabled}")
            self._append_log(
                run_dir,
                "info",
                f"Distributed: {runtime.is_distributed} (rank={runtime.rank}, world_size={runtime.world_size})",
            )
            self._append_log(run_dir, "info", f"Configured log_level={self._log_level()}")

        try:
            task, optimizer, scheduler, train_loader, val_loader, _test_loader = self._build_task_and_loaders(device, runtime)

            # Check if task uses multi-optimizer (GAN, etc.)
            optimizers_dict = None
            if hasattr(task, "get_optimizers") and callable(task.get_optimizers):
                optimizers_dict = task.get_optimizers()
                if isinstance(optimizers_dict, dict) and self._is_rank0(runtime):
                    opt_names = ", ".join(sorted(str(k) for k in optimizers_dict.keys()))
                    msg = f"Multi-optimizer mode enabled: {opt_names}"
                    self._append_log(run_dir, "info", msg)

            # In multi-optimizer mode, rebind primary optimizer/scheduler to the
            # optimizer that is actually stepped in GAN loop (generator preferred).
            if isinstance(optimizers_dict, dict) and optimizers_dict:
                schedule_optimizer = optimizers_dict.get("generator")
                if schedule_optimizer is None:
                    schedule_optimizer = next(iter(optimizers_dict.values()))
                optimizer = schedule_optimizer

                scheduler_builder = get_obj_by_name(
                    self.cfg.get("trainer", {}).get("scheduler_builder", "training.utils.optimizer.build_scheduler")
                )
                scheduler = scheduler_builder(schedule_optimizer, self.cfg)

            start_epoch = 0
            resume_enabled, resume_path, resume_strict, resume_load_opt, resume_load_sched = self._resume_options()
            if resume_enabled:
                model_ref = task.model.module if hasattr(task.model, "module") else task.model
                missing_keys, unexpected_keys, start_epoch, _extra = load_checkpoint(
                    run_dir=run_dir,
                    model=model_ref,
                    optimizer=optimizer if resume_load_opt else None,
                    scheduler=scheduler if resume_load_sched else None,
                    device=device,
                    checkpoint_path=resume_path,
                    strict=resume_strict,
                    rank=int(getattr(runtime, "rank", 0)),
                )

                if self._is_rank0(runtime):
                    source_text = resume_path if resume_path else "latest checkpoint in run dir"
                    self._append_log(run_dir, "info", f"Resume enabled: source={source_text}, start_epoch={start_epoch}")
                    if missing_keys:
                        self.logger.warning("Resume missing keys: %s", missing_keys)
                    if unexpected_keys:
                        self.logger.warning("Resume unexpected keys: %s", unexpected_keys)

            if self._is_rank0(runtime):
                log_file = str(run_dir / "log.txt")

                try:
                    summarize_training_data(
                        train_loader.dataset,
                        self.cfg,
                        log_file=log_file,
                    )
                except Exception as exc:
                    self.logger.warning("Training data summary failed and was ignored: %s", exc)

                try:
                    import torch

                    out_size, layout = resolve_input_spec(self.cfg)
                    in_channels, h, w = out_size_to_chw(out_size, layout)
                    model_ref = task.model.module if hasattr(task.model, "module") else task.model
                    dummy_input = torch.randn(1, in_channels, h, w, device=device)
                    print_module_summary(model_ref, [dummy_input], log_file=log_file)

                    # GANSystem.forward uses encoder+generator path; discriminator is
                    # not invoked there, so include a dedicated discriminator summary.
                    if hasattr(model_ref, "discriminator"):
                        self._append_log(run_dir, "info", "Discriminator summary:")
                        flat_dummy_input = dummy_input.view(dummy_input.shape[0], -1)
                        print_module_summary(model_ref.discriminator, [flat_dummy_input], log_file=log_file)
                except Exception as exc:
                    self.logger.warning("Model summary failed and was ignored: %s", exc)

            callbacks = CallbackManager(build_callbacks(self.cfg))
            state = TrainState(
                model=task.model,
                optimizer=optimizer,
                scheduler=scheduler,
                run_dir=run_dir,
                cfg=self.cfg,
                runtime=runtime,
            )
            callbacks.on_train_start(state)

            for epoch in range(start_epoch, num_epochs):
                if self._is_rank0(runtime):
                    print(f"\nEpoch {epoch + 1}/{num_epochs}")
                    self._append_log(run_dir, "info", f"Epoch {epoch + 1}/{num_epochs} started")
                state.epoch = epoch
                state.step = 0
                callbacks.on_epoch_start(state)

                if runtime.is_distributed and hasattr(train_loader.sampler, "set_epoch"):
                    train_loader.sampler.set_epoch(epoch)

                # Train
                train_metrics = self._train_epoch(
                    state,
                    task,
                    train_loader,
                    optimizer,
                    device,
                    runtime,
                    callbacks,
                    amp_enabled=amp_enabled,
                    scaler=scaler,
                    optimizers_dict=optimizers_dict,
                )

                # Validate
                val_metrics = self._validate_epoch(task, val_loader, device, runtime)
                state.metrics = {**train_metrics, **val_metrics}

                # Log
                train_metric_str = ", ".join(
                    f"{name}={value:.4f}" for name, value in train_metrics.items() if name != "train_loss"
                ) or "no metrics"
                val_metric_str = ", ".join(
                    f"{name}={value:.4f}" for name, value in val_metrics.items() if name != "val_loss"
                ) or "no metrics"
                if self._is_rank0(runtime):
                    print(f"  Train Loss: {train_metrics['train_loss']:.4f} | {train_metric_str}")
                    print(f"  Val Loss: {val_metrics['val_loss']:.4f} | {val_metric_str}")

                lr_before_step = None
                if optimizer is not None and getattr(optimizer, "param_groups", None):
                    try:
                        lr_before_step = float(optimizer.param_groups[0].get("lr"))
                    except Exception:
                        lr_before_step = None

                # Scheduler step
                if scheduler:
                    scheduler.step()
                callbacks.on_epoch_end(state)

                lr_after_step = None
                if optimizer is not None and getattr(optimizer, "param_groups", None):
                    try:
                        lr_after_step = float(optimizer.param_groups[0].get("lr"))
                    except Exception:
                        lr_after_step = None
                train_metric_str = ", ".join(
                    f"{name}={value:.4f}" for name, value in train_metrics.items() if name != "train_loss"
                ) or "no metrics"
                val_metric_str = ", ".join(
                    f"{name}={value:.4f}" for name, value in val_metrics.items() if name != "val_loss"
                ) or "no metrics"

                epoch_summary = f"Epoch {epoch + 1}/{num_epochs} completed | "
                if lr_before_step is not None:
                    epoch_summary += f"lr_before_step={lr_before_step:.8f} | "
                if lr_after_step is not None:
                    epoch_summary += f"lr_after_step={lr_after_step:.8f} | "
                epoch_summary += (
                    f"train_loss={train_metrics.get('train_loss', float('nan')):.4f} | "
                    f"{train_metric_str} | "
                    f"val_loss={val_metrics.get('val_loss', float('nan')):.4f} | "
                    f"{val_metric_str}"
                )
                self._append_log(run_dir, "info", epoch_summary)

            callbacks.on_train_end(state)
            if self._is_rank0(runtime):
                print(f"\nTraining complete. Metrics saved to: {run_dir / 'metrics.json'}")
                print(f"Best checkpoint: {run_dir / 'checkpoints' / 'best.pt'}")
                self._append_log(run_dir, "info", f"Training complete. Metrics saved to: {run_dir / 'metrics.json'}")
                self._append_log(run_dir, "info", f"Best checkpoint: {run_dir / 'checkpoints' / 'best.pt'}")

            return run_dir
        finally:
            teardown_distributed(runtime)
