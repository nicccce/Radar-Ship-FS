"""Immutable result and observability records for stable RL selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from harness.contract import SubsetSelection


@dataclass(frozen=True)
class TrainingMetrics:
    step: int
    subset: tuple[int, ...]
    subset_size: int
    accuracy: float
    best_accuracy: float
    epsilon: float
    proposed_select_count: int
    transition_applied: bool
    reward_min: float
    reward_mean: float
    reward_max: float
    replay_size: int
    update_performed: bool
    loss: float | None
    td_error_mean: float | None
    td_error_max: float | None
    q_mean: float | None
    q_std: float | None
    q_max: float | None
    target_q_mean: float | None
    gradient_norm: float | None
    target_synced: bool
    advisor_override_count: int
    elapsed_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return {
            key: list(value) if isinstance(value, tuple) else value for key, value in self.__dict__.items()
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TrainingMetrics":
        payload = dict(value)
        payload["subset"] = tuple(int(item) for item in payload["subset"])
        return cls(**payload)


@dataclass(frozen=True)
class StableTrainingResult:
    selection: SubsetSelection
    metrics: tuple[TrainingMetrics, ...]
    initial_subset: tuple[int, ...]
    initial_accuracy: float | None
    learner_updates: int
    rejected_transitions: int
