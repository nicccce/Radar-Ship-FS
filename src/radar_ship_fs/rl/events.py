"""Immutable training events and a small observer dispatcher."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from radar_ship_fs.rl.trainer import UpdateMetrics
from radar_ship_fs.selection.types import TrainingMetrics


@dataclass(frozen=True)
class StepCompleted:
    metrics: TrainingMetrics


@dataclass(frozen=True)
class UpdateCompleted:
    step: int
    metrics: UpdateMetrics


@dataclass(frozen=True)
class CheckpointSaved:
    step: int
    path: Path


class TrainingObserver(Protocol):
    def on_step(self, event: StepCompleted) -> None: ...

    def on_update(self, event: UpdateCompleted) -> None: ...

    def on_checkpoint(self, event: CheckpointSaved) -> None: ...


class EventBus:
    def __init__(self, observers: Sequence[TrainingObserver] = ()) -> None:
        self._observers = tuple(observers)

    def emit_step(self, event: StepCompleted) -> None:
        for observer in self._observers:
            callback = getattr(observer, "on_step", None)
            if callback is not None:
                callback(event)

    def emit_update(self, event: UpdateCompleted) -> None:
        for observer in self._observers:
            callback = getattr(observer, "on_update", None)
            if callback is not None:
                callback(event)

    def emit_checkpoint(self, event: CheckpointSaved) -> None:
        for observer in self._observers:
            callback = getattr(observer, "on_checkpoint", None)
            if callback is not None:
                callback(event)
