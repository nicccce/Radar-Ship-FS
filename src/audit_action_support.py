#!/usr/bin/env python3
"""Independent structural and sampled-score audit for action-support-v1."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_action_support import CONFIG, KS, METHODS, PHASES, PROTOCOL, ROOT, SEEDS, load_case, sha
from run_search_diagnosis import fit_search, read, read_lines
from run_unified_baselines import _write_json

EPS = 1e-12


def parse_ids(value: str) -> list[int]:
    return [int(item) for item in json.loads(value)]


def main() -> None:
    manifest = read(ROOT / "manifest.json")
    complete = read(ROOT / "complete.json")
    current_hashes = {path: sha(path) for path in manifest["code_protocol_config_hashes"]}
    assert current_hashes == manifest["code_protocol_config_hashes"]
    assert sha(CONFIG) == manifest["code_protocol_config_hashes"][str(CONFIG)]
    assert sha(PROTOCOL) == manifest["code_protocol_config_hashes"][str(PROTOCOL)]
    assert complete["source_test_open_count"] == complete["rl_training_count"] == 0

    analysis = ROOT / "analysis"
    gaps = pd.read_csv(analysis / "support_gaps.csv")
    reachable = pd.read_csv(analysis / "old_reachable_subsets.csv")
    runs = pd.read_csv(analysis / "replay_runs.csv")
    curves = pd.read_csv(analysis / "budget_curves.csv")
    coverage = pd.read_csv(analysis / "coverage_regression.csv")
    costs = pd.read_csv(analysis / "costs.csv")

    expected_origins = {(seed, k, phase) for seed in SEEDS for k in KS for phase in PHASES}
    actual_origins = set(zip(gaps.seed, gaps.k, gaps.phase))
    assert actual_origins == expected_origins
    assert len(gaps) == complete["origins"] == 30
    assert (gaps.old_reachable_2swap_count_including_start == 117).all()
    assert (gaps.old_direct_count_including_start == 17).all()
    expected_full = 1 + gaps.k * (65 - gaps.k)
    assert (gaps.complete_1swap_count_including_start == expected_full).all()
    gap_error = np.max(
        np.abs(
            gaps.action_support_gap_1swap
            - (gaps.complete_1swap_best_score - gaps.old_direct_best_score)
        )
    )
    assert gap_error < EPS
    assert (gaps.action_support_gap_1swap >= -EPS).all()
    assert gaps.historical_matched_policy_best_score.isna().all()
    assert gaps.historical_search_policy_gap.isna().all()
    assert gaps.task3_rule_not_used_as_policy_gap.all()

    reach_counts = reachable.groupby(["seed", "k", "phase"]).size()
    assert len(reach_counts) == 30 and (reach_counts == 117).all()
    assert len(reachable) == complete["old_reachable_rows"] == 3510
    assert set(reachable.minimum_swaps) == {0, 1, 2}
    coordinate_rows = 0
    for seed in SEEDS:
        context = read(
            Path("experiments/search_diagnosis_v1/search")
            / f"nested_dev-seed-{seed}"
            / "context.json"
        )
        original_ids = np.asarray(context["final_feature_ids"], dtype=int)
        for row in reachable[reachable.seed == seed].itertuples(index=False):
            clean = parse_ids(row.clean_indices_0based)
            original = parse_ids(row.original_feature_ids_1based)
            assert original_ids[clean].tolist() == original
            coordinate_rows += 1
    assert coordinate_rows == len(reachable)
    assert reachable.coordinate_crosscheck_passed.all()

    replay_seeds = manifest["frozen_config"]["replay"]["replay_seeds"]
    budget = manifest["frozen_config"]["replay"]["incremental_unique_subset_budget"]
    expected_runs = len(expected_origins) * len(METHODS) * len(replay_seeds)
    assert len(runs) == complete["replay_runs"] == expected_runs == 450
    assert (runs.incremental_unique_subsets == budget).all()
    assert (runs.logical_classifier_fits == 5 * budget).all()
    assert (runs.requests - runs.cache_hits == budget).all()
    old = runs[runs.method == "old_mask"]
    policy_gap_error = np.max(
        np.abs(old.search_policy_gap_at_64 - (old.old_2swap_ceiling_score - old.final_best_score))
    )
    assert policy_gap_error < EPS
    assert (old.search_policy_gap_at_64 >= -EPS).all()

    expected_curve_rows = expected_runs * budget
    assert len(curves) == complete["budget_curve_rows"] == expected_curve_rows == 28800
    curve_groups = curves.groupby(["seed", "k", "phase", "method", "replay_seed"], sort=False)
    for _, group in curve_groups:
        assert group.unique_subset_budget.tolist() == list(range(1, budget + 1))
        assert np.all(np.diff(group.best_so_far_score) >= -EPS)
    assert len(curve_groups) == expected_runs

    assert coverage.passed.all()
    counts = coverage.groupby("case_type").size().to_dict()
    assert counts == {
        "k16_accepted_edge": 6,
        "k16_improved_endpoint": 4,
        "low_mi_conditional_contribution": 23,
    }
    assert (coverage.exchange_support_probability > 0.0).all()
    assert complete["k16_improved_endpoints"] == 4
    assert complete["low_mi_regression_cases"] == 23
    assert complete["all_coverage_regressions_passed"]

    assert len(costs) == len(SEEDS)
    assert costs.physical_classifier_fits.sum() == complete["physical_classifier_fits"]
    assert costs.new_unique_scored_subsets.sum() == complete["physical_new_unique_subsets"]
    assert (costs.physical_classifier_fits == 5 * costs.new_unique_scored_subsets).all()

    recompute_errors = []
    recompute_fits = 0
    for seed in SEEDS:
        _, _, _, _, context, _, _ = load_case(seed, jobs=1)
        bank = read_lines(ROOT / "cases" / f"seed-{seed}" / "candidate_scores.jsonl")
        candidate = next((row for row in bank if row["score_source"] == "action_support_v1_refit"), bank[0])
        subset = tuple(candidate["clean_indices_0based"])
        score, folds, _ = fit_search(context, subset)
        recompute_errors.append(abs(score - candidate["search_accuracy"]))
        recompute_errors.extend(
            abs(float(actual) - float(expected))
            for actual, expected in zip(folds, candidate["fold_scores"])
        )
        recompute_fits += 5
    max_recompute_error = max(recompute_errors)
    assert max_recompute_error < EPS

    _write_json(
        ROOT / "audit.json",
        {
            "status": "passed",
            "manifest_hashes_verified": True,
            "source_test_used": False,
            "rl_training_count": 0,
            "origins_checked": len(gaps),
            "reachable_rows_and_coordinates_checked": coordinate_rows,
            "old_reachable_count_per_origin": 117,
            "replay_runs_checked": len(runs),
            "budget_curve_rows_checked": len(curves),
            "coverage_case_counts": counts,
            "gap_formula_max_error": float(gap_error),
            "policy_gap_formula_max_error": float(policy_gap_error),
            "sampled_score_recompute_fits": recompute_fits,
            "sampled_score_recompute_max_error": float(max_recompute_error),
            "scope": (
                "All structure, coordinates, budgets, gaps, and coverage probabilities; "
                "one independently refitted subset per seed. No source-test or RL training."
            ),
        },
    )
    print("action-support-v1 audit passed")


if __name__ == "__main__":
    main()
