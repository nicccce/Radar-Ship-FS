"""Pointer-style feature-selection environments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ExperimentConfig
from .evaluator import SubsetEvaluator
from .feature_ids import feature_id_score, normalized_feature_ids
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
    feature_id_score: float
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
        self.feature_ids = np.asarray(graph.feature_ids, dtype=np.int64).copy()
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
        self._support_draw = -1
        self._support_rng = np.random.default_rng(
            np.random.SeedSequence([int(config.seed), 0x53555050, 0])
        )
        self._support_cache: dict[tuple[bytes, bool], np.ndarray] = {}

    def reset(self) -> Observation:
        self.selected = self.initial_mask.copy()
        self.swaps_completed = 0
        self.removed_feature = None
        self.total_reward = 0.0
        self._support_draw += 1
        self._support_rng = np.random.default_rng(
            np.random.SeedSequence([int(self.config.seed), 0x53555050, self._support_draw])
        )
        self._support_cache.clear()
        return self.observation()

    def _scratch_valid_actions(self) -> np.ndarray:
        selected_count = int(self.selected.sum())
        valid = np.zeros(self.ppo_graph.n_features + 1, dtype=bool)
        if selected_count < self.config.feature_budget:
            valid[: self.ppo_graph.n_features] = ~self.selected
        valid[self.stop_action] = selected_count >= self.config.min_features
        return valid

    def _quality_order(self, eligible: np.ndarray, *, largest: bool) -> np.ndarray:
        indices = np.flatnonzero(eligible)
        scores = self.ppo_graph.static_node_features[indices, :2].mean(axis=1)
        if self.config.feature_id_reward_weight > 0.0:
            identifier_preference = normalized_feature_ids(self.feature_ids)[indices]
            scores = scores + self.config.feature_id_reward_weight * identifier_preference
        order = np.argsort(scores, kind="stable")
        if largest:
            order = order[::-1]
        return indices[order]

    def swap_action_support_probability(
        self,
        eligible: np.ndarray,
        action: int,
        *,
        largest: bool,
    ) -> float:
        """Return the action's marginal inclusion probability over support draws."""
        eligible = np.asarray(eligible, dtype=bool)
        if action < 0 or action >= eligible.size or not eligible[action]:
            return 0.0
        ordered = self._quality_order(eligible, largest=largest)
        prior_count = min(self.config.swap_candidate_pool, len(ordered))
        if action in set(ordered[:prior_count].tolist()):
            return 1.0
        remainder = len(ordered) - prior_count
        quota = min(self.config.swap_exploration_pool, remainder)
        return float(quota / remainder) if remainder else 0.0

    def _swap_candidates(
        self,
        eligible: np.ndarray,
        *,
        largest: bool,
    ) -> np.ndarray:
        eligible = np.asarray(eligible, dtype=bool)
        cache_key = (np.packbits(eligible).tobytes(), largest)
        if cache_key in self._support_cache:
            return self._support_cache[cache_key].copy()

        ordered = self._quality_order(eligible, largest=largest)
        prior_count = min(self.config.swap_candidate_pool, len(ordered))
        prior = ordered[:prior_count]
        remainder = ordered[prior_count:]
        quota = min(self.config.swap_exploration_pool, len(remainder))
        if quota:
            exploration = self._support_rng.choice(remainder, size=quota, replace=False)
            candidates = np.concatenate([prior, np.sort(exploration)])
        else:
            candidates = prior
        self._support_cache[cache_key] = candidates.copy()
        return candidates

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
        value = relevance - 0.25 * coverage * self.ppo_graph.redundancy(mask)
        if self.config.feature_id_reward_weight > 0.0:
            value += self.config.feature_id_reward_weight * feature_id_score(mask, self.feature_ids)
        return float(value)

    def objective(self, accuracy: float, mask: np.ndarray) -> tuple[float, float]:
        redundancy = self.ppo_graph.redundancy(mask)
        sparsity = 1.0 - float(mask.sum()) / self.config.feature_budget
        value = accuracy - self.config.correlation_penalty * redundancy
        value += self.config.sparsity_bonus * sparsity
        if self.config.feature_id_reward_weight > 0.0:
            value += self.config.feature_id_reward_weight * feature_id_score(mask, self.feature_ids)
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
            raise ValueError(f"invalid action {action}; valid={np.flatnonzero(valid).tolist()}")

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
            reward += self.config.terminal_reward_scale * (objective - self.baseline_objective)
            self.total_reward += reward
            summary = EpisodeSummary(
                selected=self.selected.copy(),
                cv_accuracy=cv.mean_accuracy,
                fold_accuracies=cv.fold_accuracies,
                redundancy=redundancy,
                feature_id_score=feature_id_score(self.selected, self.feature_ids),
                objective=objective,
                total_reward=self.total_reward,
            )
        else:
            self.total_reward += reward
        return float(reward), done, summary
