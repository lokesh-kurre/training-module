from __future__ import annotations

from typing import Iterable

from training.engine.state import TrainState


class Callback:
    def on_train_start(self, state: TrainState) -> None:
        return None

    def on_train_end(self, state: TrainState) -> None:
        return None

    def on_epoch_start(self, state: TrainState) -> None:
        return None

    def on_epoch_end(self, state: TrainState) -> None:
        return None

    def on_batch_end(self, state: TrainState) -> None:
        return None


class CallbackManager:
    def __init__(self, callbacks: Iterable[Callback]):
        self.callbacks = list(callbacks)

    def on_train_start(self, state: TrainState) -> None:
        for callback in self.callbacks:
            callback.on_train_start(state)

    def on_train_end(self, state: TrainState) -> None:
        for callback in self.callbacks:
            callback.on_train_end(state)

    def on_epoch_start(self, state: TrainState) -> None:
        for callback in self.callbacks:
            callback.on_epoch_start(state)

    def on_epoch_end(self, state: TrainState) -> None:
        for callback in self.callbacks:
            callback.on_epoch_end(state)

    def on_batch_end(self, state: TrainState) -> None:
        for callback in self.callbacks:
            callback.on_batch_end(state)
