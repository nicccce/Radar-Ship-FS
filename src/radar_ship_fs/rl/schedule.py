"""Deterministic training schedules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LinearEpsilonSchedule:
    start: float
    end: float
    total_steps: int
    decay_fraction: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.end <= self.start <= 1.0:
            raise ValueError("epsilon values must satisfy 0 <= end <= start <= 1")
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if not 0.0 < self.decay_fraction <= 1.0:
            raise ValueError("decay_fraction must be in (0, 1]")

    @property
    def decay_steps(self) -> int:
        return max(1, int(round(self.total_steps * self.decay_fraction)))

    def value(self, step: int) -> float:
        if step <= 0:
            return float(self.start)
        progress = min(1.0, float(step) / self.decay_steps)
        return float(self.start + progress * (self.end - self.start))

    def state_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "total_steps": self.total_steps,
            "decay_fraction": self.decay_fraction,
        }

    def load_state_dict(self, state: dict) -> None:
        expected = self.state_dict()
        restored = {
            "start": float(state["start"]),
            "end": float(state["end"]),
            "total_steps": int(state["total_steps"]),
            "decay_fraction": float(state["decay_fraction"]),
        }
        if restored != expected:
            raise ValueError("epsilon schedule checkpoint does not match the run configuration")
