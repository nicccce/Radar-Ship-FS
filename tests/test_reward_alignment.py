from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler

import run_reward_alignment as reward_alignment


def test_candidate_bank_contains_frozen_strata_and_complete_single_swap() -> None:
    forward = tuple(range(8))
    mi_topk = tuple(range(2, 10))
    rows = reward_alignment.build_candidate_bank(
        forward,
        mi_topk,
        n_features=20,
        k=8,
        seed=42,
        global_random_count=6,
        perturbation_count=4,
    )
    assert all(len(row["clean_indices_0based"]) == 8 for row in rows)
    assert len({tuple(row["clean_indices_0based"]) for row in rows}) == len(rows)
    assert sum("forward_checkpoint" in row["strata"] for row in rows) == 1
    assert sum("mi_topk" in row["strata"] for row in rows) == 1
    assert sum("complete_forward_single_swap" in row["strata"] for row in rows) == 8 * 12
    assert sum("global_random" in row["strata"] for row in rows) == 6
    for distance in (2, 4, 8):
        label = f"forward_perturbation_distance_{distance}"
        assert sum(label in row["strata"] for row in rows) == 4
        assert all(distance in row["swap_distances_from_forward"] for row in rows if label in row["strata"])


def test_repeat_aggregations_use_fixed_three_repeat_rule() -> None:
    result = reward_alignment.aggregate_repeat_scores([0.7, 0.8, 0.9])
    assert result["single_5fold"] == 0.7
    assert np.isclose(result["repeated_mean"], 0.8)
    assert np.isclose(result["mean_minus_0_5_sd"], 0.75)


def test_lr_scaler_is_fitted_separately_on_each_scoring_training_fold(monkeypatch) -> None:
    rng = np.random.default_rng(7)
    raw = rng.normal(size=(30, 3))
    labels = np.tile([0, 1, 2], 10)
    repeats = []
    expected_means = []
    for random_state in (11, 12, 13):
        folds = []
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        for fit, held_out in splitter.split(raw, labels):
            folds.append({"fit": fit.tolist(), "held_out": held_out.tolist()})
            expected_means.append(raw[fit][:, [0, 2]].mean(axis=0))
        repeats.append(folds)

    observed_means = []

    class RecordingScaler(SklearnStandardScaler):
        def fit(self, X, y=None, sample_weight=None):  # noqa: N803
            observed_means.append(np.asarray(X).mean(axis=0))
            return super().fit(X, y, sample_weight=sample_weight)

    monkeypatch.setattr(reward_alignment, "StandardScaler", RecordingScaler)
    result = reward_alignment.score_candidate_all_signals(
        raw,
        labels,
        [1, 2, 3],
        [0, 2],
        repeats,
        seed=42,
        tree_random_state=19,
    )
    assert result["standard_scaler_fit_scope"] == "each_scoring_training_fold"
    assert result["classifier_fit_count"] == 30
    assert len(observed_means) == 15
    assert np.allclose(observed_means, expected_means)


def test_protocol_selection_is_lr_bacc_only_and_lexicographic() -> None:
    rows = []
    enrichments = {
        "single_5fold": {16: 2.1, 32: 1.8},
        "repeated_mean": {16: 2.2, 32: 2.3},
        "mean_minus_0_5_sd": {16: 2.0, 32: 2.0},
    }
    for aggregation, by_k in enrichments.items():
        for k, enrichment in by_k.items():
            for seed in reward_alignment.SEEDS:
                rows.append(
                    {
                        "seed": seed,
                        "k": k,
                        "signal": "lr_bacc",
                        "aggregation": aggregation,
                        "top_b_enrichment": enrichment,
                        "spearman": 0.2,
                        "repeat_rank_stability": 0.8,
                    }
                )
                rows.append(
                    {
                        "seed": seed,
                        "k": k,
                        "signal": "dt_accuracy",
                        "aggregation": aggregation,
                        "top_b_enrichment": 99.0,
                        "spearman": 0.99,
                        "repeat_rank_stability": 0.99,
                    }
                )
    selection = reward_alignment.choose_lr_bacc_protocol(rows)
    assert selection["selected"]["signal"] == "lr_bacc"
    assert selection["selected"]["aggregation"] == "repeated_mean"
    assert selection["selected"]["bank_gate_passed"] is True


def test_best_candidate_ties_keep_frozen_candidate_order() -> None:
    candidates = [(0, 2), (0, 3), (1, 2)]
    assert reward_alignment.choose_best(candidates, [0.5, 0.5, 0.4]) == (0, 2)
