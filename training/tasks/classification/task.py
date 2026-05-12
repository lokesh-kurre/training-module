from __future__ import annotations

from typing import Any

from training.tasks.base import BaseTask, TaskStepOutput
from training.tasks.registry import register_task
from training.utils.importer import get_obj_by_name
from training.utils.input_spec import resolve_input_spec


@register_task("classification")
class ClassificationTask(BaseTask):
    """Classification task with explicit model/loss/metrics ownership."""

    def build_model(self) -> Any:
        model_builder = get_obj_by_name(
            self.cfg.get("trainer", {}).get(
                "model_builder",
                "training.tasks.classification.model.build_model",
            )
        )
        return model_builder(self.cfg)

    def build_losses(self) -> Any:
        loss_builder = get_obj_by_name(
            self.cfg.get("trainer", {}).get(
                "loss_builder",
                "training.tasks.classification.losses.build_loss",
            )
        )
        return loss_builder(self.cfg)

    def build_metrics(self) -> Any:
        metrics_fn = self.cfg.get("trainer", {}).get(
            "metrics_fn",
            "training.tasks.classification.metrics.compute_metrics",
        )
        if isinstance(metrics_fn, str):
            return get_obj_by_name(metrics_fn)
        return metrics_fn

    def _step(self, batch: Any, training: bool) -> TaskStepOutput:
        images, targets = batch
        _out_size, layout = resolve_input_spec(self.cfg)
        if layout == "HWC" and hasattr(images, "ndim") and images.ndim == 4:
            images = images.permute(0, 3, 1, 2).contiguous()
        logits = self.model(images)
        loss = self.loss_fn(logits, targets)
        metrics = self.metrics_fn(logits.detach(), targets, self.cfg)
        return TaskStepOutput(loss=loss, outputs={"logits": logits, "targets": targets}, metrics=metrics)

    def training_step(self, batch: Any) -> TaskStepOutput:
        return self._step(batch, training=True)

    def validation_step(self, batch: Any) -> TaskStepOutput:
        return self._step(batch, training=False)

    def predict_step(self, batch: Any) -> Any:
        images, _targets = batch
        return self.model(images)
