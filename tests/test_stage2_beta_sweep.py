"""Lightweight tests for the Full-IRFS beta sweeps."""

from __future__ import annotations

import dataclasses
import json

import pytest

import run_stage2_beta_sweep
import run_stage2_beta_sweep_selection as selection_module
import run_stage2_beta_sweep_trained_gcn as trained_gcn_sweep
from run_stage2_beta_sweep_dt_test import _aggregate
from run_stage2_beta_sweep_selection import (
    _config_for_beta,
    _selection_signature,
    beta_tag,
)
from stage2_rl_config import BETA_SWEEP_SEEDS, BETA_SWEEP_VALUES


def test_beta_grid_and_seed_reduction_are_fixed() -> None:
    assert BETA_SWEEP_VALUES == (0.0, 0.02, 0.1, 0.5)
    assert BETA_SWEEP_SEEDS == (42, 43, 44, 45)


def test_beta_config_changes_only_correlation_penalty() -> None:
    left = dataclasses.asdict(_config_for_beta(0.0))
    right = dataclasses.asdict(_config_for_beta(0.5))

    assert left.pop("correlation_penalty_weight") == 0.0
    assert right.pop("correlation_penalty_weight") == 0.5
    assert left == right
    with pytest.raises(ValueError, match="non-negative"):
        _config_for_beta(-0.01)


def test_beta_tag_and_resume_signature_are_stable() -> None:
    assert [beta_tag(value) for value in BETA_SWEEP_VALUES] == [
        "beta_0",
        "beta_0p02",
        "beta_0p1",
        "beta_0p5",
    ]
    signature = _selection_signature(42, 0.02)

    assert json.loads(json.dumps(signature)) == signature
    assert signature["beta"] == 0.02
    assert signature["effective_irfs_config"]["correlation_penalty_weight"] == 0.02
    assert signature["effective_irfs_config"]["exploration_step_budget"] == 250


def test_dt_aggregate_reports_paired_delta_against_mi_kbest() -> None:
    seed_results = []
    for seed, mi_score, beta_score in ((42, 0.80, 0.85), (43, 0.90, 0.88)):
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
                        "beta": None,
                        "selected_count": 27,
                        "selection_best_dt_inner_cv_accuracy": None,
                        "dt_test_accuracy": mi_score,
                    },
                    {
                        **common,
                        "name": "beta_0p02_selected",
                        "beta": 0.02,
                        "selected_count": 30,
                        "selection_best_dt_inner_cv_accuracy": 0.9,
                        "dt_test_accuracy": beta_score,
                    },
                ],
            }
        )

    aggregate, flat_rows = _aggregate(seed_results)
    beta = next(item for item in aggregate["methods"] if item["name"] == "beta_0p02_selected")

    assert beta["delta_vs_kbest_mutual_info"]["values"] == pytest.approx([0.05, -0.02])
    assert beta["win_tie_loss_vs_kbest_mutual_info"] == {
        "win": 1,
        "tie": 0,
        "loss": 1,
    }
    assert len(flat_rows) == 4


def test_one_click_entry_runs_all_selection_before_dt(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        run_stage2_beta_sweep,
        "run_selection",
        lambda: calls.append("selection"),
    )
    monkeypatch.setattr(
        run_stage2_beta_sweep,
        "run_dt_test",
        lambda: calls.append("dt_test"),
    )

    run_stage2_beta_sweep.main()

    assert calls == ["selection", "dt_test"]


def test_configure_variant_selects_trained_gcn_and_separate_root(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(selection_module, "REPORT_NAME", "original")
    monkeypatch.setattr(selection_module, "STATE_ENCODER", "fixed")
    monkeypatch.setattr(selection_module, "BETA_SWEEP_SELECTION_ROOT", tmp_path / "old")
    monkeypatch.setattr(selection_module, "BETA_SWEEP_TABLE_PREFIX", "old")
    target = tmp_path / "gcn"

    selection_module.configure_variant(
        report_name="full_irfs_trained_gcn",
        state_encoder="trained_gcn",
        selection_root=target,
        table_prefix="gcn_beta",
    )

    config = selection_module._config_for_beta(0.02)
    signature = selection_module._selection_signature(42, 0.02)
    assert config.state_encoder == "trained_gcn"
    assert config.correlation_penalty_weight == 0.02
    assert signature["report_name"] == "full_irfs_trained_gcn"
    assert signature["state_encoder"] == "trained_gcn"
    assert selection_module._selection_path(42, 0.02).is_relative_to(target)


def test_trained_gcn_one_click_configures_then_selects_then_tests(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        trained_gcn_sweep,
        "_configure_trained_gcn",
        lambda: calls.append("configure"),
    )
    monkeypatch.setattr(
        trained_gcn_sweep.selection,
        "main",
        lambda: calls.append("selection"),
    )
    monkeypatch.setattr(
        trained_gcn_sweep.dt_test,
        "main",
        lambda: calls.append("dt_test"),
    )

    trained_gcn_sweep.main()

    assert calls == ["configure", "selection", "dt_test"]
