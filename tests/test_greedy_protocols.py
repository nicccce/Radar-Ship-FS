"""The three MI/greedy labels must continue to denote different algorithms."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from methods.forward_greedy import ForwardGreedySelector
from methods.mi_greedy import MIOrderedImprovementSelector
from methods.relevance_topk import RelevanceTopKSelector


class _Rng:
    numpy = np.random.default_rng(7)


class _Probe:
    def __init__(self, scores):
        self.scores = {tuple(sorted(key)): value for key, value in scores.items()}
        self.calls = []

    def probe(self, subset, validation):  # noqa: ARG002
        key = tuple(sorted(int(value) for value in subset))
        self.calls.append(key)
        return SimpleNamespace(accuracy=self.scores.get(key, 0.0))


def _context(probe):
    X = np.arange(40, dtype=float).reshape(10, 4)
    split = SimpleNamespace(
        train=SimpleNamespace(X=X, y=np.asarray([0, 1] * 5)),
        validation=object(),
    )
    return SimpleNamespace(
        split=split,
        probe=probe,
        config=SimpleNamespace(feature_budget=2),
        rng=_Rng(),
        n_features=4,
    )


def test_mi_topk_is_fixed_k_without_subset_scoring(monkeypatch) -> None:
    monkeypatch.setattr(
        "methods.relevance_topk.mutual_info_classif",
        lambda X, y, random_state: np.asarray([0.4, 0.3, 0.2, 0.1]),  # noqa: ARG005
    )
    probe = _Probe({})
    selected = RelevanceTopKSelector().select(_context(probe)).selected

    assert selected == (0, 1)
    assert probe.calls == []


def test_mi_ordered_accept_checks_one_fixed_ranking(monkeypatch) -> None:
    monkeypatch.setattr(
        "methods.mi_greedy.mutual_info_classif",
        lambda X, y, random_state: np.asarray([0.4, 0.3, 0.2, 0.1]),  # noqa: ARG005
    )
    probe = _Probe({(0,): 0.60, (0, 1): 0.59, (0, 2): 0.61})
    selected = MIOrderedImprovementSelector().select(_context(probe)).selected

    assert selected == (0, 2)
    assert probe.calls == [(0,), (0, 1), (0, 2)]


def test_forward_greedy_enumerates_all_remaining_features_each_step() -> None:
    probe = _Probe(
        {
            (0,): 0.60,
            (1,): 0.70,
            (2,): 0.65,
            (3,): 0.50,
            (0, 1): 0.69,
            (1, 2): 0.80,
            (1, 3): 0.60,
        }
    )
    selected = ForwardGreedySelector().select(_context(probe)).selected

    assert selected == (1, 2)
    assert probe.calls[:4] == [(0,), (1,), (2,), (3,)]
    assert set(probe.calls[4:]) == {(0, 1), (1, 2), (1, 3)}
