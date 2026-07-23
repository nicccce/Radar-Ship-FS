"""Tests for Hybrid Teaching guidance-ratio sweeps."""

from __future__ import annotations

import pytest

import run_stage2_guidance_sweep as guidance
from stage2_rl_config import (
    EXPLORATION_STEP_BUDGET,
    GUIDANCE_SCHEDULE_SPECS,
    GUIDANCE_SWEEP_BETA,
)


def test_guidance_schedule_grid_covers_requested_ratios() -> None:
    assert GUIDANCE_SWEEP_BETA == 0.5
    assert GUIDANCE_SCHEDULE_SPECS == (
        ("thirds_83_83_84", 83, 166),
        ("mi_heavy_200_25_25", 200, 225),
        ("mi_only_250_0_0", 250, 250),
        ("dt_heavy_25_200_25", 25, 225),
    )
    assert [guidance._phase_lengths(spec) for spec in GUIDANCE_SCHEDULE_SPECS] == [
        (83, 83, 84),
        (200, 25, 25),
        (250, 0, 0),
        (25, 200, 25),
    ]
    assert all(
        sum(guidance._phase_lengths(spec)) == EXPLORATION_STEP_BUDGET for spec in GUIDANCE_SCHEDULE_SPECS
    )


def test_invalid_guidance_boundaries_are_rejected() -> None:
    with pytest.raises(ValueError, match="invalid guidance"):
        guidance._phase_lengths(("bad", 200, 100))
    with pytest.raises(ValueError, match="invalid guidance"):
        guidance._phase_lengths(("too_long", 200, 251))


def test_configure_schedule_changes_only_code_defined_variant(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(guidance, "GUIDANCE_SWEEP_SELECTION_ROOT", tmp_path / "selection")
    monkeypatch.setattr(guidance, "GUIDANCE_SWEEP_DT_TEST_ROOT", tmp_path / "dt")
    monkeypatch.setattr(guidance.stage2_selection, "HYBRID_SWITCH_STEP", 83)
    monkeypatch.setattr(guidance.stage2_selection, "HYBRID_WITHDRAW_STEP", 166)
    monkeypatch.setattr(guidance.selection, "BETA_SWEEP_VALUES", (0.0,))
    monkeypatch.setattr(guidance.dt_test, "BETA_SWEEP_VALUES", (0.0,))
    monkeypatch.setattr(guidance.selection, "REPORT_NAME", "old")
    monkeypatch.setattr(guidance.selection, "STATE_ENCODER", "trained_gcn")
    monkeypatch.setattr(
        guidance.selection,
        "BETA_SWEEP_SELECTION_ROOT",
        tmp_path / "old_selection",
    )
    monkeypatch.setattr(guidance.selection, "BETA_SWEEP_TABLE_PREFIX", "old")
    monkeypatch.setattr(
        guidance.dt_test,
        "BETA_SWEEP_SELECTION_ROOT",
        tmp_path / "old_selection",
    )
    monkeypatch.setattr(
        guidance.dt_test,
        "BETA_SWEEP_DT_TEST_ROOT",
        tmp_path / "old_dt",
    )
    monkeypatch.setattr(guidance.dt_test, "BETA_SWEEP_TABLE_PREFIX", "old")

    spec = ("mi_heavy_200_25_25", 200, 225)
    guidance._configure_schedule(spec)
    config = guidance.selection._config_for_beta(GUIDANCE_SWEEP_BETA)

    assert config.state_encoder == "fixed"
    assert config.correlation_penalty_weight == 0.5
    assert config.hybrid_switch_step == 200
    assert config.hybrid_withdraw_step == 225
    assert guidance.selection.BETA_SWEEP_VALUES == (0.5,)
    assert guidance.dt_test.BETA_SWEEP_VALUES == (0.5,)
    assert guidance.selection.REPORT_NAME.endswith(spec[0])
    assert guidance.selection.BETA_SWEEP_SELECTION_ROOT == (tmp_path / "selection" / spec[0])
    assert guidance.dt_test.BETA_SWEEP_DT_TEST_ROOT == tmp_path / "dt" / spec[0]


def test_one_click_finishes_all_selections_before_any_dt(monkeypatch) -> None:
    specs = (("a", 1, 2), ("b", 3, 4))
    calls: list[str] = []
    active = {"name": ""}

    def configure(spec) -> None:
        active["name"] = spec[0]
        calls.append(f"configure:{spec[0]}")

    monkeypatch.setattr(guidance, "GUIDANCE_SCHEDULE_SPECS", specs)
    monkeypatch.setattr(guidance, "_configure_schedule", configure)
    monkeypatch.setattr(
        guidance.selection,
        "main",
        lambda: calls.append(f"selection:{active['name']}"),
    )
    monkeypatch.setattr(
        guidance.dt_test,
        "_require_complete_sweep",
        lambda: calls.append(f"preflight:{active['name']}") or {1: {}},
    )
    monkeypatch.setattr(
        guidance.dt_test,
        "main",
        lambda: calls.append(f"dt:{active['name']}"),
    )
    monkeypatch.setattr(
        guidance,
        "_combine_results",
        lambda: calls.append("combine") or {"schedule_summaries": []},
    )

    guidance.main()

    assert calls == [
        "configure:a",
        "selection:a",
        "configure:b",
        "selection:b",
        "configure:a",
        "preflight:a",
        "configure:b",
        "preflight:b",
        "configure:a",
        "dt:a",
        "configure:b",
        "dt:b",
        "combine",
    ]
