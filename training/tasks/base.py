from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskStepOutput:
    """Container for task step results.

    The task owns model forward, loss computation, and metrics computation.
    The engine owns backward, optimizer, scheduler, checkpointing, and DDP.
    """

    loss: Any
    outputs: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)


class BaseTask(ABC):
    """Task interface for pluggable training problems."""

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.model = self.build_model()
        self.loss_fn = self.build_losses()
        self.metrics_fn = self.build_metrics()

    @abstractmethod
    def build_model(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def build_losses(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def build_metrics(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def training_step(self, batch: Any) -> TaskStepOutput:
        raise NotImplementedError

    @abstractmethod
    def validation_step(self, batch: Any) -> TaskStepOutput:
        raise NotImplementedError

    @abstractmethod
    def predict_step(self, batch: Any) -> Any:
        raise NotImplementedError
