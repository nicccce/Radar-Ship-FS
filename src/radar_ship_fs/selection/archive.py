"""Candidate archive separated from environment dynamics and DQN optimization."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class SelectionArchive:
    """Track the historical selection rule without coupling it to the trainer.

    The stable engine deliberately retains the legacy rule in this phase: rank by
    validation accuracy, reject over-budget candidates, and break exact ties in
    favour of fewer features.
    """

    best_subset: tuple[int, ...]
    best_accuracy: float
    feature_budget: int | None = None

    def consider(self, subset: tuple[int, ...], accuracy: float) -> bool:
        if self.feature_budget is not None and len(subset) > self.feature_budget:
            return False
        better = accuracy > self.best_accuracy and not math.isclose(
            accuracy, self.best_accuracy, rel_tol=0.0, abs_tol=1e-12
        )
        tied_smaller = math.isclose(accuracy, self.best_accuracy, rel_tol=0.0, abs_tol=1e-12) and len(
            subset
        ) < len(self.best_subset)
        if better or tied_smaller:
            self.best_subset = tuple(subset)
            self.best_accuracy = float(accuracy)
            return True
        return False

    def state_dict(self) -> dict:
        return {
            "best_subset": tuple(self.best_subset),
            "best_accuracy": float(self.best_accuracy),
            "feature_budget": self.feature_budget,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "SelectionArchive":
        return cls(
            best_subset=tuple(int(value) for value in state["best_subset"]),
            best_accuracy=float(state["best_accuracy"]),
            feature_budget=state.get("feature_budget"),
        )
