"""Serializable advisor implementations for stable training sessions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np

from trainers.classify import classify_agents
from trainers.dt_importance import DTImportanceTrainer
from trainers.relevance import RelevanceTrainer

if TYPE_CHECKING:
    from harness.contract import SelectionContext


class StableAdvisor:
    """Explicit advisor object without lambda checkpoint coupling."""

    def __init__(self, mode: str, *, switch_step: int, withdraw_step: int) -> None:
        if mode not in {"relevance", "dt_importance", "hybrid"}:
            raise ValueError(f"unsupported stable advisor mode {mode!r}")
        self.mode = mode
        self.switch_step = int(switch_step)
        self.withdraw_step = int(withdraw_step)
        self.relevance = RelevanceTrainer()
        self.dt_importance = DTImportanceTrainer()

    def _active_mode(self, step: int) -> str | None:
        if self.mode != "hybrid":
            return self.mode
        if step < self.switch_step:
            return "relevance"
        if step < self.withdraw_step:
            return "dt_importance"
        return None

    def advise(
        self,
        step: int,
        prior_actions: Sequence[int],
        current_actions: Sequence[int],
        context: "SelectionContext",
    ) -> Mapping[int, int]:
        classification = classify_agents(prior_actions, current_actions)
        active = self._active_mode(step)
        if active == "relevance":
            return self.relevance.advise(classification, context)
        if active == "dt_importance":
            return self.dt_importance.advise(classification, context)
        return {}

    def state_dict(self) -> dict:
        relevance = self.relevance._relevance
        return {
            "mode": self.mode,
            "switch_step": self.switch_step,
            "withdraw_step": self.withdraw_step,
            "relevance": None if relevance is None else relevance.copy(),
        }

    def load_state_dict(self, state: dict) -> None:
        identity = (
            state["mode"],
            int(state["switch_step"]),
            int(state["withdraw_step"]),
        )
        expected = (self.mode, self.switch_step, self.withdraw_step)
        if identity != expected:
            raise ValueError("advisor checkpoint does not match the configured advisor")
        relevance = state["relevance"]
        self.relevance._relevance = None if relevance is None else np.asarray(relevance, dtype=float).copy()


def build_stable_advisor(mode: str | None, context: "SelectionContext") -> StableAdvisor | None:
    if mode in {None, "none"}:
        return None
    return StableAdvisor(
        mode,
        switch_step=context.config.hybrid_switch_step,
        withdraw_step=context.config.hybrid_withdraw_step,
    )
