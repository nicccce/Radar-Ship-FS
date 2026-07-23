"""Tests for held-out Decision-Tree testing of selected feature sets."""

from __future__ import annotations

import numpy as np
import pytest

from data.splitter import Partition, Split
from harness.contract import SelectionContext
from run_stage2_dt_test import _aggregate, _development_partition, _top_k


def test_top_k_is_canonical_and_validated() -> None:
    ranked = np.asarray([3, 1, 2, 0])

    assert _top_k(ranked, 2) == (1, 3)
    with pytest.raises(ValueError, match="k must be"):
        _top_k(ranked, 0)


def test_development_partition_combines_train_and_validation() -> None:
    X = np.arange(60, dtype=np.float32).reshape(12, 5)
    y = np.asarray([-1, 1] * 6)
    names = [f"feature_{index}" for index in range(5)]

    def partition(indices: list[int]) -> Partition:
        index_array = np.asarray(indices)
        return Partition(X[index_array], y[index_array], index_array, names)

    context = SelectionContext(
        split=Split(partition(list(range(6))), partition([6, 7, 8]), partition([9, 10, 11])),
        probe=None,
        config=None,
        rng=None,
    )

    development = _development_partition(context)

    np.testing.assert_array_equal(development.X, X[:9])
    np.testing.assert_array_equal(development.y, y[:9])
    np.testing.assert_array_equal(development.indices, np.arange(9))


def test_aggregate_reports_paired_dt_test_delta_against_fixed() -> None:
    seed_results = []
    for seed, all_score, fixed_score in ((42, 0.8, 0.9), (43, 0.95, 0.9)):
        seed_results.append(
            {
                "seed": seed,
                "methods": [
                    {
                        "name": "all_features_54",
                        "selected_count": 54,
                        "compression_ratio": 0.0,
                        "dt_development_accuracy": 1.0,
                        "dt_test_accuracy": all_score,
                        "dt_fit_and_test_seconds": 0.1,
                        "selected_original_feature_ids": list(range(54)),
                    },
                    {
                        "name": "full_irfs_fixed_selected",
                        "selected_count": 33,
                        "compression_ratio": 1 - 33 / 54,
                        "dt_development_accuracy": 1.0,
                        "dt_test_accuracy": fixed_score,
                        "dt_fit_and_test_seconds": 0.1,
                        "selected_original_feature_ids": list(range(33)),
                    },
                ],
            }
        )

    aggregate, flat_rows = _aggregate(seed_results)
    all_features = next(method for method in aggregate["methods"] if method["name"] == "all_features_54")

    assert all_features["delta_vs_full_irfs_fixed"]["values"] == pytest.approx([-0.1, 0.05])
    assert all_features["win_tie_loss_vs_full_irfs_fixed"] == {
        "win": 1,
        "tie": 0,
        "loss": 1,
    }
    assert all_features["dt_test_accuracy"]["values"] == [0.8, 0.95]
    assert len(flat_rows) == 4
