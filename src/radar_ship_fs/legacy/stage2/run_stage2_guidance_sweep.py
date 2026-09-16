#!/usr/bin/env python3
"""Scan Hybrid Teaching phase ratios with fixed beta, then run held-out DT validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import run_stage2_beta_sweep_dt_test as dt_test
import run_stage2_beta_sweep_selection as selection
import run_stage2_rl_selection as stage2_selection
from run_stage2_rl_selection import _write_csv, _write_json
from stage2_rl_config import (
    BETA_SWEEP_SEEDS,
    EXPLORATION_STEP_BUDGET,
    GUIDANCE_SCHEDULE_SPECS,
    GUIDANCE_SWEEP_BETA,
    GUIDANCE_SWEEP_DT_TEST_ROOT,
    GUIDANCE_SWEEP_SELECTION_ROOT,
    GUIDANCE_SWEEP_TABLE_PREFIX,
    TABLE_ROOT,
)

ScheduleSpec = tuple[str, int, int]


def _phase_lengths(spec: ScheduleSpec) -> tuple[int, int, int]:
    _name, switch, withdraw = spec
    if not 0 <= switch <= withdraw <= EXPLORATION_STEP_BUDGET:
        raise ValueError(
            f"invalid guidance boundaries: switch={switch}, withdraw={withdraw}, "
            f"budget={EXPLORATION_STEP_BUDGET}"
        )
    return switch, withdraw - switch, EXPLORATION_STEP_BUDGET - withdraw


def _selection_root(name: str) -> Path:
    return GUIDANCE_SWEEP_SELECTION_ROOT / name


def _dt_root(name: str) -> Path:
    return GUIDANCE_SWEEP_DT_TEST_ROOT / name


def _configure_schedule(spec: ScheduleSpec) -> None:
    name, switch, withdraw = spec
    _phase_lengths(spec)

    # _config_for_encoder reads these code-level constants when building the effective config.
    stage2_selection.HYBRID_SWITCH_STEP = switch
    stage2_selection.HYBRID_WITHDRAW_STEP = withdraw

    # Reuse the beta-sweep engine with a one-element beta grid and schedule-specific roots.
    selection.BETA_SWEEP_VALUES = (GUIDANCE_SWEEP_BETA,)
    dt_test.BETA_SWEEP_VALUES = (GUIDANCE_SWEEP_BETA,)
    table_prefix = f"{GUIDANCE_SWEEP_TABLE_PREFIX}_{name}"
    selection.configure_variant(
        report_name=f"full_irfs_fixed_guidance_{name}",
        state_encoder="fixed",
        selection_root=_selection_root(name),
        table_prefix=table_prefix,
    )
    dt_test.configure_output(
        selection_root=_selection_root(name),
        dt_test_root=_dt_root(name),
        table_prefix=table_prefix,
    )


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _combine_results() -> dict[str, Any]:
    schedule_summaries: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    kbest_summary: dict[str, Any] | None = None

    for spec in GUIDANCE_SCHEDULE_SPECS:
        name, switch, withdraw = spec
        relevance_steps, dt_steps, withdrawn_steps = _phase_lengths(spec)
        selection_aggregate = _read_json(_selection_root(name) / "aggregate.json")
        dt_aggregate = _read_json(_dt_root(name) / "aggregate.json")
        selection_summary = selection_aggregate["beta_summaries"][0]
        beta_method = next(
            method
            for method in dt_aggregate["methods"]
            if method["name"] == selection.beta_tag(GUIDANCE_SWEEP_BETA) + "_selected"
        )
        if kbest_summary is None:
            kbest_summary = next(
                method for method in dt_aggregate["methods"] if method["name"] == "kbest_mutual_info"
            )

        summary = {
            "schedule": name,
            "hybrid_switch_step": switch,
            "hybrid_withdraw_step": withdraw,
            "relevance_steps": relevance_steps,
            "dt_importance_steps": dt_steps,
            "withdrawn_steps": withdrawn_steps,
            "beta": GUIDANCE_SWEEP_BETA,
            "selected_count": selection_summary["selected_count"],
            "best_dt_inner_cv_accuracy": selection_summary["best_dt_inner_cv_accuracy"],
            "selected_average_pairwise_abs_correlation": selection_summary[
                "selected_average_pairwise_abs_correlation"
            ],
            "selection_elapsed_seconds": selection_summary["selection_elapsed_seconds"],
            "dt_test_accuracy": beta_method["dt_test_accuracy"],
            "delta_vs_kbest_mutual_info": beta_method["delta_vs_kbest_mutual_info"],
            "win_tie_loss_vs_kbest_mutual_info": beta_method["win_tie_loss_vs_kbest_mutual_info"],
        }
        schedule_summaries.append(summary)
        csv_rows.append(
            {
                "schedule": name,
                "relevance_steps": relevance_steps,
                "dt_importance_steps": dt_steps,
                "withdrawn_steps": withdrawn_steps,
                "beta": GUIDANCE_SWEEP_BETA,
                "selected_count_mean": summary["selected_count"]["mean"],
                "selected_count_std": summary["selected_count"]["std"],
                "best_dt_inner_cv_accuracy_mean": summary["best_dt_inner_cv_accuracy"]["mean"],
                "selected_average_pairwise_abs_correlation_mean": summary[
                    "selected_average_pairwise_abs_correlation"
                ]["mean"],
                "selection_elapsed_seconds_mean": summary["selection_elapsed_seconds"]["mean"],
                "dt_test_accuracy_mean": summary["dt_test_accuracy"]["mean"],
                "dt_test_accuracy_std": summary["dt_test_accuracy"]["std"],
                "delta_vs_kbest_mutual_info_mean": summary["delta_vs_kbest_mutual_info"]["mean"],
                **summary["win_tie_loss_vs_kbest_mutual_info"],
            }
        )

    aggregate = {
        "beta": GUIDANCE_SWEEP_BETA,
        "seeds": list(BETA_SWEEP_SEEDS),
        "budget": EXPLORATION_STEP_BUDGET,
        "all_schedule_selections_completed_before_test_evaluation": True,
        "kbest_mutual_info": kbest_summary,
        "schedule_summaries": schedule_summaries,
    }
    _write_json(
        aggregate,
        GUIDANCE_SWEEP_DT_TEST_ROOT / "aggregate_across_schedules.json",
    )
    _write_csv(
        csv_rows,
        GUIDANCE_SWEEP_DT_TEST_ROOT / "aggregate_across_schedules.csv",
    )
    _write_csv(
        csv_rows,
        TABLE_ROOT / f"{GUIDANCE_SWEEP_TABLE_PREFIX}_dt_test_aggregate.csv",
    )
    return aggregate


def main() -> None:
    print(
        f"guidance sweep: beta={GUIDANCE_SWEEP_BETA:g} "
        f"schedules={[spec[0] for spec in GUIDANCE_SCHEDULE_SPECS]} "
        f"seeds={list(BETA_SWEEP_SEEDS)}; source test unused during selection",
        flush=True,
    )

    # Phase 1: finish every schedule/seed selection on source train data.
    for spec in GUIDANCE_SCHEDULE_SPECS:
        name = spec[0]
        relevance_steps, dt_steps, withdrawn_steps = _phase_lengths(spec)
        print(
            f"selection schedule={name} phases={relevance_steps}/{dt_steps}/{withdrawn_steps}",
            flush=True,
        )
        _configure_schedule(spec)
        selection.main()

    # Phase 2: validate all 4 x 4 artifacts before test evaluation.
    complete_count = 0
    for spec in GUIDANCE_SCHEDULE_SPECS:
        _configure_schedule(spec)
        complete_count += len(dt_test._require_complete_sweep())
    print(
        f"validated all {complete_count} guidance-sweep selections; outer DT test is now permitted",
        flush=True,
    )

    # Phase 3: evaluate each schedule on the same per-seed held-out tests.
    for spec in GUIDANCE_SCHEDULE_SPECS:
        _configure_schedule(spec)
        dt_test.main()

    aggregate = _combine_results()
    print("\nguidance-sweep aggregate against MI-KBest:")
    for item in aggregate["schedule_summaries"]:
        accuracy = item["dt_test_accuracy"]
        delta = item["delta_vs_kbest_mutual_info"]
        record = item["win_tie_loss_vs_kbest_mutual_info"]
        print(
            f"{item['schedule']:<24} phases="
            f"{item['relevance_steps']}/{item['dt_importance_steps']}/"
            f"{item['withdrawn_steps']} features={item['selected_count']['mean']:.1f} "
            f"test={accuracy['mean']:.4f}±{accuracy['std']:.4f} "
            f"delta={delta['mean']:+.4f} "
            f"W/T/L={record['win']}/{record['tie']}/{record['loss']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
