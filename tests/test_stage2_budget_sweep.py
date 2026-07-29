"""Tests for the cardinality-aware Full-IRFS lambda sweep."""

from __future__ import annotations

import dataclasses
import json

import pytest

import run_stage2_budget_sweep as budget_sweep
from stage2_rl_config import (
    BUDGET_SWEEP_BETA,
    BUDGET_SWEEP_FEATURE_BUDGET,
    BUDGET_SWEEP_SEEDS,
    BUDGET_SWEEP_VALUES,
    K_BEST,
)


def test_budget_grid_and_fixed_low_beta() -> None:
    assert BUDGET_SWEEP_BETA == 0.02
    assert BUDGET_SWEEP_FEATURE_BUDGET == K_BEST
    assert BUDGET_SWEEP_VALUES == (0.01, 0.025, 0.05, 0.1)
    assert BUDGET_SWEEP_SEEDS == (42, 43, 44, 45)


def test_lambda_config_changes_only_over_budget_weight() -> None:
    left = dataclasses.asdict(budget_sweep._config_for_lambda(0.01))
    right = dataclasses.asdict(budget_sweep._config_for_lambda(0.1))

    assert left.pop("over_budget_penalty_weight") == 0.01
    assert right.pop("over_budget_penalty_weight") == 0.1
    assert left == right
    assert left["correlation_penalty_weight"] == 0.02
    assert left["feature_budget"] == K_BEST
    with pytest.raises(ValueError, match="non-negative"):
        budget_sweep._config_for_lambda(-0.01)


def test_lambda_tag_and_signature_are_stable() -> None:
    assert [budget_sweep.lambda_tag(value) for value in BUDGET_SWEEP_VALUES] == [
        "lambda_0p01",
        "lambda_0p025",
        "lambda_0p05",
        "lambda_0p1",
    ]
    signature = budget_sweep._selection_signature(42, 0.025)

    assert json.loads(json.dumps(signature)) == signature
    assert signature["beta"] == 0.02
    assert signature["lambda"] == 0.025
    assert signature["feature_budget"] == K_BEST
    assert signature["effective_irfs_config"]["feature_budget"] == K_BEST


def test_resume_rejects_oversized_selection(tmp_path) -> None:
    path = tmp_path / "selection.json"
    signature = budget_sweep._selection_signature(42, 0.01)
    path.write_text(
        json.dumps(
            {
                "experiment_signature": signature,
                "protocol": {
                    "official_test_accessed": False,
                    "held_out_random_test_accessed": False,
                    "outer_test_release_permitted": False,
                },
                "trajectory": [{}] * 250,
                "selected_clean_indices": list(range(K_BEST + 1)),
                "selected_count": K_BEST + 1,
                "initial_candidate": {
                    "included_in_final_candidate_pool": True,
                    "selected_count": K_BEST,
                },
            }
        ),
        encoding="utf-8",
    )

    assert budget_sweep._load_matching(path, signature) is None


def test_dt_aggregate_reports_paired_delta_against_mi_kbest() -> None:
    seed_results = []
    for seed, mi_score, rl_score in ((42, 0.80, 0.85), (43, 0.90, 0.88)):
        common = {
            "compression_ratio": 0.5,
            "dt_development_accuracy": 1.0,
            "dt_fit_and_test_seconds": 0.1,
            "selected_original_feature_ids": [1, 2],
        }
        seed_results.append(
            {
                "seed": seed,
                "methods": [
                    {
                        **common,
                        "name": "kbest_mutual_info",
                        "lambda": None,
                        "selected_count": K_BEST,
                        "selection_best_feasible_dt_inner_cv_accuracy": None,
                        "dt_test_accuracy": mi_score,
                    },
                    {
                        **common,
                        "name": "lambda_0p025_selected",
                        "lambda": 0.025,
                        "selected_count": 24,
                        "selection_best_feasible_dt_inner_cv_accuracy": 0.9,
                        "dt_test_accuracy": rl_score,
                    },
                ],
            }
        )

    aggregate, flat_rows = budget_sweep._dt_aggregate(seed_results)
    selected = next(item for item in aggregate["methods"] if item["name"] == "lambda_0p025_selected")

    assert selected["delta_vs_kbest_mutual_info"]["values"] == pytest.approx([0.05, -0.02])
    assert selected["win_tie_loss_vs_kbest_mutual_info"] == {
        "win": 1,
        "tie": 0,
        "loss": 1,
    }
    assert len(flat_rows) == 4


def test_one_click_runs_all_selection_before_dt(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        budget_sweep,
        "run_selection",
        lambda: calls.append("selection"),
    )
    monkeypatch.setattr(
        budget_sweep,
        "run_dt_test",
        lambda: calls.append("dt_test"),
    )

    budget_sweep.main()

    assert calls == ["selection", "dt_test"]
