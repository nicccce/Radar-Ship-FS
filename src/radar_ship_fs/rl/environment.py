"""Feature-subset environment, isolated from replay and DQN optimization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from engine.policy import ACTION_DESELECT, ACTION_SELECT
from radar_ship_fs.rl.transition import EnvironmentStep, JointTransition

if TYPE_CHECKING:
    from engine.seam import ActionAdvisor
    from harness.contract import SelectionContext
    from radar_ship_fs.feedback.protocols import RewardVector


class SubsetEnvironment:
    """Own the only stable-v1 path that can commit a new feature subset."""

    def __init__(
        self,
        context: "SelectionContext",
        reward: "RewardVector",
        advisor: "ActionAdvisor | None" = None,
    ) -> None:
        self.context = context
        self.reward = reward
        self.advisor = advisor

    def initial_subset(self) -> tuple[int, ...]:
        n_features = self.context.n_features
        size = max(1, n_features // 2)
        budget = self.context.config.feature_budget
        if budget is not None:
            size = min(size, int(budget))
        chosen = self.context.rng.numpy.choice(n_features, size=size, replace=False)
        return tuple(sorted(int(value) for value in chosen))

    def score(self, subset: tuple[int, ...]) -> float:
        return float(self.context.probe.probe(subset, self.context.split.validation).accuracy)

    def step(
        self,
        *,
        step: int,
        committed: tuple[int, ...],
        proposed_actions: np.ndarray,
        done: bool,
    ) -> EnvironmentStep:
        actions = np.asarray(proposed_actions, dtype=np.int64).copy()
        if actions.shape != (self.context.n_features,):
            raise ValueError("proposed_actions must contain one action per feature")
        if not np.all((actions == ACTION_DESELECT) | (actions == ACTION_SELECT)):
            raise ValueError("proposed_actions contains an invalid action")
        proposed_select_count = int(np.count_nonzero(actions == ACTION_SELECT))

        override_count = 0
        if self.advisor is not None:
            committed_set = set(committed)
            prior = np.array(
                [
                    ACTION_SELECT if feature in committed_set else ACTION_DESELECT
                    for feature in range(self.context.n_features)
                ],
                dtype=np.int64,
            )
            overrides = self.advisor.advise(step, prior, actions.copy(), self.context)
            for feature, action in overrides.items():
                feature = int(feature)
                if not 0 <= feature < self.context.n_features:
                    raise ValueError(f"advisor returned out-of-range feature {feature}")
                if action not in {ACTION_DESELECT, ACTION_SELECT}:
                    raise ValueError(f"advisor returned invalid action {action!r}")
                if actions[feature] != action:
                    override_count += 1
                actions[feature] = int(action)

        voted = tuple(int(feature) for feature, action in enumerate(actions) if action == ACTION_SELECT)
        applied = 0 < len(voted) < self.context.n_features
        next_subset = voted if applied else committed
        accuracy = self.score(next_subset)
        rewards = self.reward.evaluate(next_subset, self.context)
        if rewards.shape != (self.context.n_features,):
            raise ValueError("reward vector must contain one value per feature")

        transition = JointTransition(
            subset=committed,
            actions=actions,
            rewards=rewards,
            next_subset=next_subset,
            applied=applied,
            done=bool(done),
        )
        return EnvironmentStep(
            transition=transition,
            accuracy=accuracy,
            advisor_override_count=override_count,
            proposed_select_count=proposed_select_count,
        )

    def state_dict(self) -> dict:
        return {
            "reward": self.reward.state_dict(),
            "advisor": None if self.advisor is None else self.advisor.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.reward.load_state_dict(state["reward"])
        advisor_state = state["advisor"]
        if advisor_state is None:
            if self.advisor is not None:
                raise ValueError("checkpoint has no advisor but the run config enables one")
            return
        if self.advisor is None:
            raise ValueError("checkpoint has advisor state but the run config disables it")
        self.advisor.load_state_dict(advisor_state)
