"""Vector-valued adapters preserving the existing reward mathematics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from engine.reward_overall import OverallReward
from reward.overall import overall_reward
from reward.personalize import per_agent_reward_vector

if TYPE_CHECKING:
    from harness.contract import SelectionContext


class UniformRewardVector:
    """Broadcast the current MARLFS scalar reward to every feature agent."""

    def __init__(self) -> None:
        self._reward = OverallReward()

    def evaluate(self, subset: tuple[int, ...], context: "SelectionContext") -> np.ndarray:
        value = self._reward.reward(subset, context)
        return np.full(context.n_features, value, dtype=np.float32)

    def state_dict(self) -> dict:
        # Caches are derived and intentionally omitted from checkpoints.
        return {}

    def load_state_dict(self, state: dict) -> None:
        if state:
            raise ValueError("uniform reward checkpoint state must be empty")
        self._reward = OverallReward()


class PersonalizedRewardVector:
    """Compute the existing Full-IRFS reward vector once per environment step."""

    def __init__(self) -> None:
        self._counts: np.ndarray | None = None
        self._last_subset: tuple[int, ...] | None = None

    def evaluate(self, subset: tuple[int, ...], context: "SelectionContext") -> np.ndarray:
        if context.config.per_agent_credit == "symmetric":
            return np.full(context.n_features, overall_reward(subset, context), dtype=np.float32)
        counts = None
        if context.config.reward_scheme == "frequency":
            if self._counts is None:
                self._counts = np.zeros(context.n_features, dtype=float)
            if subset != self._last_subset:
                self._counts[list(subset)] += 1.0
                self._last_subset = subset
            counts = self._counts
        return np.asarray(per_agent_reward_vector(subset, context, selection_counts=counts), dtype=np.float32)

    def state_dict(self) -> dict:
        return {
            "counts": None if self._counts is None else self._counts.copy(),
            "last_subset": self._last_subset,
        }

    def load_state_dict(self, state: dict) -> None:
        counts = state.get("counts")
        self._counts = None if counts is None else np.asarray(counts, dtype=float).copy()
        last = state.get("last_subset")
        self._last_subset = None if last is None else tuple(int(value) for value in last)


def build_reward_vector(name: str):
    if name == "uniform":
        return UniformRewardVector()
    if name == "personalized":
        return PersonalizedRewardVector()
    raise ValueError(f"unknown stable reward {name!r}")
