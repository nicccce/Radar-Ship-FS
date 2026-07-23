"""Tests for validation-free basic baselines and the shared final LR scorer."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification

from harness.lr_final import score_selected_features_with_lr
from run_basic_baselines import _kbest_mutual_information


def test_kbest_mutual_information_is_fixed_size_and_seed_reproducible() -> None:
    X, y = make_classification(
        n_samples=200,
        n_features=8,
        n_informative=3,
        n_redundant=0,
        random_state=7,
    )
    first, first_scores = _kbest_mutual_information(X, y, k=4, seed=42)
    second, second_scores = _kbest_mutual_information(X, y, k=4, seed=42)

    assert len(first) == 4
    assert first == tuple(sorted(set(first))) == second
    np.testing.assert_array_equal(first_scores, second_scores)

    with pytest.raises(ValueError, match="k must be"):
        _kbest_mutual_information(X, y, k=0, seed=42)


def test_final_lr_scorer_reports_binary_metrics_for_selected_features() -> None:
    X, y_zero_one = make_classification(
        n_samples=300,
        n_features=10,
        n_informative=6,
        class_sep=1.5,
        random_state=11,
    )
    y = np.where(y_zero_one == 1, 1, -1)
    X_train, X_test = X[:220], X[220:]
    y_train, y_test = y[:220], y[220:]

    metrics = score_selected_features_with_lr(
        X_train,
        y_train,
        X_test,
        y_test,
        subset=range(10),
        random_state=42,
    )

    assert metrics.test_accuracy > 0.8
    assert metrics.positive_label == 1
    assert len(metrics.confusion_matrix) == 2
    for value in (
        metrics.train_accuracy,
        metrics.test_accuracy,
        metrics.balanced_accuracy,
        metrics.precision,
        metrics.recall,
        metrics.f1,
        metrics.f1_macro,
        metrics.roc_auc,
    ):
        assert 0.0 <= value <= 1.0
