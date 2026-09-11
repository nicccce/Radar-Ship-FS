"""PPO stable random feature-ID reward tests."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from radar_ship_fs.ppo.config import ExperimentConfig
from radar_ship_fs.ppo.evaluator import CVResult
from radar_ship_fs.ppo.experiment import _candidate, _is_archive_better
from radar_ship_fs.ppo.feature_ids import feature_id_score, normalized_feature_ids, random_feature_ids
from radar_ship_fs.ppo.ppo_env import FeatureSelectionEnv
from radar_ship_fs.ppo.ppo_graph import build_feature_graph


class _FlatGraph:
    def __init__(self, n_features: int, feature_id_seed: int) -> None:
        self.n_features = n_features
        self.static_node_features = np.zeros((n_features, 2), dtype=np.float32)
        self.feature_ids = random_feature_ids(n_features, feature_id_seed)

    def redundancy(self, mask: np.ndarray) -> float:  # noqa: ARG002
        return 0.0


class _ConstantEvaluator:
    def score(self, mask: np.ndarray) -> CVResult:  # noqa: ARG002
        return CVResult(0.8, (0.8,))


def _singleton_mask(n_features: int, feature: int) -> np.ndarray:
    mask = np.zeros(n_features, dtype=bool)
    mask[feature] = True
    return mask


def test_random_feature_ids_are_a_fixed_seeded_permutation() -> None:
    first = random_feature_ids(65, seed=17)
    repeated = random_feature_ids(65, seed=17)
    different = random_feature_ids(65, seed=18)

    assert np.array_equal(np.sort(first), np.arange(1, 66))
    assert np.array_equal(first, repeated)
    assert not np.array_equal(first, different)


def test_random_feature_id_is_visible_to_the_graph_actor() -> None:
    rng = np.random.default_rng(7)
    X = rng.normal(size=(40, 4))
    y = np.tile(np.array([0, 1]), 20)
    graph = build_feature_graph(
        X,
        y,
        np.zeros(4, dtype=int),
        threshold=0.8,
        seed=11,
        feature_id_seed=17,
    )

    assert np.array_equal(graph.feature_ids, random_feature_ids(4, seed=17))
    assert graph.static_node_features[:, -1] == pytest.approx(normalized_feature_ids(graph.feature_ids))


def test_same_accuracy_candidate_prefers_larger_random_feature_id() -> None:
    graph = _FlatGraph(8, feature_id_seed=23)
    config = ExperimentConfig(
        feature_budget=1,
        min_features=1,
        correlation_penalty=0.0,
        sparsity_bonus=0.0,
        feature_id_seed=23,
        feature_id_reward_weight=0.001,
    )
    identifiers = random_feature_ids(graph.n_features, config.feature_id_seed)
    low_mask = _singleton_mask(graph.n_features, int(np.argmin(identifiers)))
    high_mask = _singleton_mask(graph.n_features, int(np.argmax(identifiers)))
    cv = CVResult(0.8, (0.8,))

    low = _candidate(low_mask, cv, graph, config, "low_id")
    high = _candidate(high_mask, cv, graph, config, "high_id")

    assert high.feature_id_score > low.feature_id_score
    assert high.objective > low.objective
    assert _is_archive_better(high, low, minimum_cv_gain=0.003, accuracy_tolerance=0.001)

    slightly_worse = _candidate(high_mask, CVResult(0.798, (0.798,)), graph, config, "too_far")
    assert not _is_archive_better(
        slightly_worse,
        low,
        minimum_cv_gain=0.003,
        accuracy_tolerance=0.001,
    )


def test_swap_candidate_pool_prefers_high_ids_for_add_and_low_ids_for_remove() -> None:
    graph = _FlatGraph(8, feature_id_seed=29)
    config = ExperimentConfig(
        feature_budget=4,
        min_features=1,
        search_mode="swap",
        swap_candidate_pool=1,
        feature_id_seed=29,
        feature_id_reward_weight=1.0,
    )
    initial_mask = np.zeros(graph.n_features, dtype=bool)
    initial_mask[: config.feature_budget] = True
    env = FeatureSelectionEnv(
        graph,
        _ConstantEvaluator(),
        config,
        baseline_objective=0.0,
        initial_mask=initial_mask,
    )
    eligible = np.ones(graph.n_features, dtype=bool)

    assert env._swap_candidates(eligible, largest=True).tolist() == [int(np.argmax(graph.feature_ids))]
    assert env._swap_candidates(eligible, largest=False).tolist() == [int(np.argmin(graph.feature_ids))]


def test_feature_id_bonus_is_present_in_shaping_and_terminal_reward() -> None:
    graph = _FlatGraph(6, feature_id_seed=31)
    config = ExperimentConfig(
        feature_budget=1,
        min_features=1,
        search_mode="scratch",
        correlation_penalty=0.0,
        sparsity_bonus=0.0,
        feature_id_seed=31,
        feature_id_reward_weight=0.1,
        shaping_scale=1.0,
        terminal_reward_scale=1.0,
    )
    identifiers = random_feature_ids(graph.n_features, config.feature_id_seed)
    low_feature = int(np.argmin(identifiers))
    high_feature = int(np.argmax(identifiers))

    def run(feature: int):
        env = FeatureSelectionEnv(
            graph,
            _ConstantEvaluator(),
            config,
            baseline_objective=0.8,
        )
        env.reset()
        return env.step(feature)

    low_reward, low_done, low_summary = run(low_feature)
    high_reward, high_done, high_summary = run(high_feature)

    assert low_done and high_done
    assert low_summary is not None and high_summary is not None
    assert high_summary.feature_id_score == pytest.approx(
        feature_id_score(_singleton_mask(graph.n_features, high_feature), identifiers)
    )
    assert high_summary.objective > low_summary.objective
    assert high_reward > low_reward


def test_negative_feature_id_settings_are_rejected() -> None:
    with pytest.raises(ValueError, match="feature_id_reward_weight"):
        replace(ExperimentConfig(), feature_id_reward_weight=-0.1).validate()
    with pytest.raises(ValueError, match="archive_accuracy_tolerance"):
        replace(ExperimentConfig(), archive_accuracy_tolerance=-0.1).validate()
