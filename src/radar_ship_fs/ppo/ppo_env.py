"""Pointer-style feature-selection environments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ExperimentConfig
from .evaluator import SubsetEvaluator
from .ppo_graph import FeatureGraph


@dataclass(frozen=True)
class Observation:
    selected: np.ndarray
    progress: float
    valid_actions: np.ndarray


@dataclass(frozen=True)
class EpisodeSummary:
    selected: np.ndarray
    cv_accuracy: float
    fold_accuracies: tuple[float, ...]
    redundancy: float
    objective: float
    total_reward: float


class FeatureSelectionEnv:
    """Construct a subset from scratch or locally reconstruct a strong warm start."""

    def __init__(
        self,
        graph: FeatureGraph,
        evaluator: SubsetEvaluator,
        config: ExperimentConfig,
        *,
        baseline_objective: float,
        initial_mask: np.ndarray | None = None,
    ) -> None:
        self.ppo_graph = graph
        self.evaluator = evaluator
        self.config = config
        self.baseline_objective = float(baseline_objective)
        self.stop_action = graph.n_features

        if config.search_mode == "swap":
            if initial_mask is None:
                raise ValueError("swap search requires an initial mask")
            initial_mask = np.asarray(initial_mask, dtype=bool)
            if int(initial_mask.sum()) != config.feature_budget:
                raise ValueError("swap initial mask must match feature_budget")
            self.initial_mask = initial_mask.copy()
        else:
            self.initial_mask = np.zeros(graph.n_features, dtype=bool)

        self.selected = self.initial_mask.copy()
        self.swaps_completed = 0
        self.removed_feature: int | None = None
        self.total_reward = 0.0

    def reset(self) -> Observation:
        self.selected = self.initial_mask.copy()
        self.swaps_completed = 0
        self.removed_feature = None
        self.total_reward = 0.0
        return self.observation()

    def _scratch_valid_actions(self) -> np.ndarray:
        selected_count = int(self.selected.sum())
        valid = np.zeros(self.ppo_graph.n_features + 1, dtype=bool)
        if selected_count < self.config.feature_budget:
            valid[: self.ppo_graph.n_features] = ~self.selected
        valid[self.stop_action] = selected_count >= self.config.min_features
        return valid

    def _swap_candidates(
        self,
        eligible: np.ndarray,
        *,
        largest: bool,
    ) -> np.ndarray:
        indices = np.flatnonzero(eligible)
        scores = self.ppo_graph.static_node_features[indices, :2].mean(axis=1)
        order = np.argsort(scores, kind="stable")
        if largest:
            order = order[::-1]
        return indices[order[: self.config.swap_candidate_pool]]

    def _swap_valid_actions(self) -> np.ndarray:
        selected_count = int(self.selected.sum())
        valid = np.zeros(self.ppo_graph.n_features + 1, dtype=bool)
        if selected_count == self.config.feature_budget:
            candidates = self._swap_candidates(self.selected, largest=False)
            valid[candidates] = True
            valid[self.stop_action] = self.swaps_completed > 0
        else:
            eligible = ~self.selected
            if self.removed_feature is not None:
                eligible[self.removed_feature] = False
            candidates = self._swap_candidates(eligible, largest=True)
            valid[candidates] = True
        return valid

    def _valid_actions(self) -> np.ndarray:
        if self.config.search_mode == "swap":
            return self._swap_valid_actions()
        return self._scratch_valid_actions()

    def observation(self) -> Observation:
        if self.config.search_mode == "swap":
            half_step = 0.5 if self.removed_feature is not None else 0.0
            progress = (self.swaps_completed + half_step) / self.config.max_swaps
        else:
            progress = float(self.selected.sum() / self.config.feature_budget)
        return Observation(
            selected=self.selected.copy(),
            progress=float(progress),
            valid_actions=self._valid_actions(),
        )

    def _proxy(self, mask: np.ndarray) -> float:
        indices = np.flatnonzero(mask)
        if indices.size == 0:
            return 0.0
        quality = self.ppo_graph.static_node_features[:, :2].mean(axis=1)
        relevance = float(quality[indices].sum() / self.config.feature_budget)
        coverage = float(indices.size / self.config.feature_budget)
        return relevance - 0.25 * coverage * self.ppo_graph.redundancy(mask)

    def objective(self, accuracy: float, mask: np.ndarray) -> tuple[float, float]:
        redundancy = self.ppo_graph.redundancy(mask)
        sparsity = 1.0 - float(mask.sum()) / self.config.feature_budget
        value = accuracy - self.config.correlation_penalty * redundancy
        value += self.config.sparsity_bonus * sparsity
        return float(value), redundancy

    def _take_feature_action(self, action: int) -> bool:
        if self.config.search_mode == "scratch":
            self.selected[action] = True
            return int(self.selected.sum()) == self.config.feature_budget

        if int(self.selected.sum()) == self.config.feature_budget:
            self.selected[action] = False
            self.removed_feature = action
            return False

        self.selected[action] = True
        self.removed_feature = None
        self.swaps_completed += 1
        return self.swaps_completed == self.config.max_swaps

    def step(self, action: int) -> tuple[float, bool, EpisodeSummary | None]:
        valid = self._valid_actions()
        if action < 0 or action >= valid.size or not valid[action]:
            raise ValueError(
                f"invalid action {action}; valid={np.flatnonzero(valid).tolist()}"
            )

        before = self._proxy(self.selected)
        done = action == self.stop_action
        if not done:
            done = self._take_feature_action(action)
        after = self._proxy(self.selected)
        reward = self.config.shaping_scale * (after - before)

        summary = None
        if done:
            cv = self.evaluator.score(self.selected)
            objective, redundancy = self.objective(cv.mean_accuracy, self.selected)
            reward += self.config.terminal_reward_scale * (
                objective - self.baseline_objective
            )
            self.total_reward += reward
            summary = EpisodeSummary(
                selected=self.selected.copy(),
                cv_accuracy=cv.mean_accuracy,
                fold_accuracies=cv.fold_accuracies,
                redundancy=redundancy,
                objective=objective,
                total_reward=self.total_reward,
            )
        else:
            self.total_reward += reward
        return float(reward), done, summary
