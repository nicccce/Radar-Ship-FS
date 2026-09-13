from __future__ import annotations

import numpy as np
from sklearn.datasets import dump_svmlight_file

from data.loader import (
    load_radar_ship,
    load_radar_ship_source_test,
    load_radar_ship_source_train,
)
from run_unified_baselines import (
    CountingSubsetScorer,
    build_seed_context,
    load_unified_config,
)


def _radar_pair(tmp_path):
    X_train = np.zeros((12, 75), dtype=np.float32)
    X_train[:, 0] = np.arange(12)
    X_train[:, 1] = np.asarray([0, 1] * 6)
    X_train[:, 2] = X_train[:, 1]
    y_train = np.asarray([-1, 1] * 6)

    X_test = np.zeros((6, 75), dtype=np.float32)
    X_test[:, 0] = np.arange(20, 26)
    X_test[:, 1] = np.asarray([1, 0] * 3)
    X_test[:, 2] = X_test[:, 1]
    X_test[:, 3] = np.arange(30, 36)  # test-only variation remains removed
    y_test = np.asarray([-1, 1] * 3)

    dump_svmlight_file(
        X_train,
        y_train,
        str(tmp_path / "sim_ship_cr_unit.train.svm"),
        zero_based=False,
    )
    dump_svmlight_file(
        X_test,
        y_test,
        str(tmp_path / "sim_ship_cr_unit.test.svm"),
        zero_based=False,
    )
    return X_train, y_train, X_test, y_test


def test_train_only_loader_matches_full_loader_cleaning(tmp_path) -> None:
    _radar_pair(tmp_path)
    X_train, y_train, names, metadata = load_radar_ship_source_train(
        str(tmp_path),
        "unit",
    )

    assert X_train.shape == (12, 2)
    assert names == ["feature_1", "feature_2"]
    assert metadata["source_files"]["test"]["sha256"] is None
    assert metadata["source_test_rows"] is None

    X_test, y_test, test_metadata = load_radar_ship_source_test(
        str(tmp_path),
        "unit",
        metadata,
    )
    X_all, y_all, full_names, full_metadata = load_radar_ship(str(tmp_path), "unit")

    np.testing.assert_array_equal(X_all, np.vstack((X_train, X_test)))
    np.testing.assert_array_equal(y_all, np.concatenate((y_train, y_test)))
    np.testing.assert_array_equal(y_train, np.asarray([-1, 1] * 6))
    np.testing.assert_array_equal(y_test, np.asarray([-1, 1] * 3))
    assert full_names == names
    assert full_metadata["final_feature_ids"] == metadata["final_feature_ids"]
    assert test_metadata["sha256"] == full_metadata["source_files"]["test"]["sha256"]


def test_seed_context_reproduces_shared_split_and_folds() -> None:
    X = np.arange(120, dtype=float).reshape(40, 3)
    y = np.asarray([-1, 1] * 20)

    first = build_seed_context(
        X,
        y,
        seed=42,
        validation_fraction=0.25,
        n_splits=2,
    )
    second = build_seed_context(
        X,
        y,
        seed=42,
        validation_fraction=0.25,
        n_splits=2,
    )

    np.testing.assert_array_equal(first.development_original_rows, second.development_original_rows)
    assert first.cv_tree_random_state == second.cv_tree_random_state
    assert first.mi_random_state == second.mi_random_state
    assert sorted(first.development_original_rows.tolist()) == list(range(40))
    held_out = sorted(row for fold in first.fold_original_rows for row in fold["held_out"])
    assert held_out == list(range(40))


def test_counting_scorer_counts_unique_subsets_cache_hits_and_fits() -> None:
    X = np.column_stack(
        (
            np.asarray([0, 1] * 10),
            np.arange(20),
            np.asarray([1, 1, 0, 0] * 5),
        )
    ).astype(float)
    y = np.asarray([-1, 1] * 10)
    context = build_seed_context(
        X,
        y,
        seed=7,
        validation_fraction=0.25,
        n_splits=2,
    )
    scorer = CountingSubsetScorer(context, n_jobs=1)

    scores = scorer.score_many([(0,), (1,), (0,)])

    assert scores[0] == scores[2]
    assert scorer.counters()["candidate_subset_requests"] == 3
    assert scorer.counters()["unique_scored_subsets"] == 2
    assert scorer.counters()["scorer_cache_hits"] == 1
    assert scorer.counters()["classifier_fit_count"] == 4


def test_frozen_unified_configs_validate() -> None:
    full = load_unified_config("configs/v16n/unified_strong_baselines.toml")
    smoke = load_unified_config("configs/v16n/unified_strong_baselines_smoke.toml")

    assert full.selection["fixed_k"] == [8, 16, 32]
    assert full.dataset["seeds"] == [42, 43, 44, 45, 46]
    assert smoke.selection["fixed_k"] == [4, 8]
