from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TrainState:
    epoch: int = 0
    step: int = 0
    global_step: int = 0

    model: Any = None
    optimizer: Any = None
    scheduler: Any = None
    scaler: Any = None

    batch: Any = None
    outputs: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)

    run_dir: Path | None = None
    cfg: dict[str, Any] | None = None
    runtime: Any = None
