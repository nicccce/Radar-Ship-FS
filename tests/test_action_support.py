"""Candidate-support invariants for quality-prior plus global exploration."""

from dataclasses import replace

import numpy as np
import pytest

from radar_ship_fs.ppo.config import ExperimentConfig
from radar_ship_fs.ppo.ppo_env import FeatureSelectionEnv
from radar_ship_fs.ppo.ppo_graph import FeatureGraph


def _graph(d: int) -> FeatureGraph:
    identity = np.eye(d, dtype=np.float32)
    quality = np.linspace(0.0, 1.0, d, dtype=np.float32)
    return FeatureGraph(
        identity,
        identity,
        identity,
        identity,
        np.column_stack([quality, quality]),
        quality,
        quality,
        np.arange(d),
        0.8,
        0,
        0,
    )


def _env(*, exploration: int, seed: int = 42) -> FeatureSelectionEnv:
    graph = _graph(14)
    config = ExperimentConfig(
        seed=seed,
        feature_budget=6,
        min_features=4,
        search_mode="swap",
        max_swaps=2,
        swap_candidate_pool=4,
        swap_exploration_pool=exploration,
        feature_id_reward_weight=0.0,
    )
    initial = np.zeros(graph.n_features, dtype=bool)
    initial[: config.feature_budget] = True
    return FeatureSelectionEnv(graph, None, config, baseline_objective=0.0, initial_mask=initial)


def test_zero_exploration_preserves_old_pool_exactly() -> None:
    env = _env(exploration=0)
    observation = env.reset()
    removes = np.flatnonzero(observation.valid_actions[: env.ppo_graph.n_features])
    assert removes.tolist() == [0, 1, 2, 3]
    env._take_feature_action(int(removes[0]))
    adds = np.flatnonzero(env.observation().valid_actions[: env.ppo_graph.n_features])
    assert adds.tolist() == [10, 11, 12, 13]


def test_exploration_quota_is_stable_within_state_and_reproducible() -> None:
    first = _env(exploration=1, seed=77)
    second = _env(exploration=1, seed=77)
    first_masks = []
    second_masks = []
    for _ in range(8):
        a = first.reset().valid_actions
        assert np.array_equal(a, first.observation().valid_actions)
        first_masks.append(a)
        second_masks.append(second.reset().valid_actions)
    assert all(np.array_equal(a, b) for a, b in zip(first_masks, second_masks))
    assert len({np.packbits(mask).tobytes() for mask in first_masks}) > 1
    assert all(mask[:14].sum() == 5 for mask in first_masks)


def test_every_legal_swap_has_strictly_positive_marginal_support() -> None:
    env = _env(exploration=2)
    start = env.initial_mask.copy()
    for remove in np.flatnonzero(start):
        assert env.swap_action_support_probability(start, int(remove), largest=False) > 0.0
        after_remove = start.copy()
        after_remove[remove] = False
        eligible_add = ~after_remove
        eligible_add[remove] = False
        for add in np.flatnonzero(eligible_add):
            assert env.swap_action_support_probability(
                eligible_add, int(add), largest=True
            ) > 0.0


def test_old_pool_assigns_zero_support_outside_quality_prior() -> None:
    env = _env(exploration=0)
    eligible = env.initial_mask.copy()
    probabilities = [
        env.swap_action_support_probability(eligible, action, largest=False)
        for action in np.flatnonzero(eligible)
    ]
    assert probabilities.count(1.0) == 4
    assert probabilities.count(0.0) == 2


def test_negative_exploration_quota_is_rejected() -> None:
    with pytest.raises(ValueError, match="swap_exploration_pool"):
        replace(ExperimentConfig(), swap_exploration_pool=-1).validate()
