"""Tests for the stage-2 inner-CV DT reward and sparse tie-break."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from config import load_config
from cv_probe import CrossValidatedDecisionTreeProbe
from data.splitter import Partition, Split
from engine.explore import ReinforcedEngine
from harness.contract import SelectionContext
from rng import SeededRng


def _partition() -> Partition:
    generator = np.random.default_rng(7)
    y = np.asarray([-1, 1] * 25)
    X = generator.normal(size=(50, 4))
    X[:, 0] = y + generator.normal(scale=0.2, size=50)
    return Partition(
        X=X,
        y=y,
        indices=np.arange(50),
        feature_names=[f"feature_{index}" for index in range(4)],
    )


def test_cross_validated_probe_reports_fixed_fold_scores_and_rejects_test() -> None:
    development = _partition()
    probe = CrossValidatedDecisionTreeProbe(
        development,
        load_config(),
        SeededRng.from_seed(42),
        n_splits=5,
    )

    result = probe.probe((0, 1), development)
    fold_scores = probe.fold_accuracies((0, 1), development)
    folds = probe.fold_indices()

    assert len(fold_scores) == 5
    assert result.accuracy == pytest.approx(np.mean(fold_scores))
    assert probe.probe((0, 1), development).tree is result.tree
    assert len(folds) == 5
    held_out_rows = [row for fold in folds for row in fold["held_out"]]
    assert sorted(held_out_rows) == list(range(50))
    for fold in folds:
        assert set(fold["fit"]).isdisjoint(fold["held_out"])

    test = development._replace(indices=np.arange(100, 150))
    with pytest.raises(ValueError, match="only its bound development partition"):
        probe.probe((0, 1), test)


def test_inner_cv_split_view_keeps_test_private() -> None:
    development = _partition()
    test = development._replace(indices=np.arange(100, 150))
    ordinary = Split(development, development, test)

    inner = ordinary.replace_development_for_inner_cv(development)

    assert inner.train is development
    assert inner.validation is development
    with pytest.raises(AttributeError):
        _ = inner.test


def test_equal_accuracy_prefers_fewer_features() -> None:
    assert ReinforcedEngine._is_better_candidate(0.8, (0,), 0.8, (0, 1))
    assert not ReinforcedEngine._is_better_candidate(0.8, (0, 1, 2), 0.8, (0, 1))
    assert ReinforcedEngine._is_better_candidate(0.81, (0, 1, 2), 0.8, (0, 1))


def test_budget_filter_rejects_even_more_accurate_oversized_candidate() -> None:
    assert not ReinforcedEngine._is_better_candidate(
        0.99,
        (0, 1, 2),
        0.80,
        (0, 1),
        feature_budget=2,
    )


class _AccuracyProbe:
    def probe(self, subset, partition):  # noqa: ARG002
        scores = {(0, 1): 0.80, (0, 1, 2): 0.99}
        return SimpleNamespace(accuracy=scores[tuple(subset)])


class _OversizedVoteEngine(ReinforcedEngine):
    def _initial_subset(self, context):  # noqa: ARG002
        return (0, 1)

    def _vote(self, *args, **kwargs):  # noqa: ARG002
        return [], (0, 1, 2)


def test_initial_feasible_candidate_survives_when_all_steps_are_oversized(
    monkeypatch,
) -> None:
    config = load_config(
        {
            "feature_budget": 2,
            "over_budget_penalty_weight": 0.1,
            "exploration_step_budget": 1,
        }
    )
    split = SimpleNamespace(
        train=SimpleNamespace(X=np.zeros((4, 4))),
        validation=SimpleNamespace(),
    )
    context = SelectionContext(
        split=split,
        probe=_AccuracyProbe(),
        config=config,
        rng=SeededRng.from_seed(7),
    )
    monkeypatch.setattr("engine.explore.build_agents", lambda context, encoder: [])

    observed = []
    selection = _OversizedVoteEngine().select(
        context, on_initial=lambda subset, accuracy: observed.append((subset, accuracy))
    )

    assert observed == [((0, 1), 0.80)]
    assert selection.selected == (0, 1)
    assert selection.per_step[0].subset == (0, 1, 2)
    assert selection.per_step[0].accuracy == pytest.approx(0.99)
