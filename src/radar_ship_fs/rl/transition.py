"""One environment transition shared by the complete feature-agent population."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class JointTransition:
    subset: tuple[int, ...]
    actions: np.ndarray
    rewards: np.ndarray
    next_subset: tuple[int, ...]
    applied: bool
    done: bool

    def __post_init__(self) -> None:
        actions = np.asarray(self.actions, dtype=np.int64).copy()
        rewards = np.asarray(self.rewards, dtype=np.float32).copy()
        if actions.ndim != 1 or rewards.ndim != 1 or actions.shape != rewards.shape:
            raise ValueError("actions and rewards must be equal-length one-dimensional arrays")
        if not np.all((actions == 0) | (actions == 1)):
            raise ValueError("actions must contain only DESELECT=0 and SELECT=1")
        actions.setflags(write=False)
        rewards.setflags(write=False)
        object.__setattr__(self, "subset", tuple(int(value) for value in self.subset))
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "rewards", rewards)
        object.__setattr__(self, "next_subset", tuple(int(value) for value in self.next_subset))


@dataclass(frozen=True)
class EnvironmentStep:
    transition: JointTransition
    accuracy: float
    advisor_override_count: int
    proposed_select_count: int
