#!/usr/bin/env python3
"""Create descriptive Task 4A summaries and a static budget-curve figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from run_unified_baselines import _write_csv, _write_json

ROOT = Path("experiments/action_support_v1")
ANALYSIS = ROOT / "analysis"
EPS = 1e-12


def signed_counts(values: pd.Series) -> dict[str, int]:
    array = values.to_numpy(dtype=float)
    return {
        "positive": int(np.sum(array > EPS)),
        "zero": int(np.sum(np.abs(array) <= EPS)),
        "negative": int(np.sum(array < -EPS)),
    }


def main() -> None:
    gaps = pd.read_csv(ANALYSIS / "support_gaps.csv")
    runs = pd.read_csv(ANALYSIS / "replay_runs.csv")
    curves = pd.read_csv(ANALYSIS / "budget_curves.csv")
    paired = pd.read_csv(ANALYSIS / "paired_final_differences.csv")

    support_rows = []
    for (k, phase), group in gaps.groupby(["k", "phase"], sort=True):
        support_rows.append(
            {
                "k": int(k),
                "phase": phase,
                "n_origins": len(group),
                "action_support_gap_1swap_mean": group.action_support_gap_1swap.mean(),
                "action_support_gap_1swap_sd": group.action_support_gap_1swap.std(ddof=1),
                "action_support_gap_1swap_max": group.action_support_gap_1swap.max(),
                "origins_with_positive_support_gap": int(
                    (group.action_support_gap_1swap > EPS).sum()
                ),
                "old_2swap_ceiling_gain_mean": (
                    group.old_2swap_ceiling_score - group.start_score
                ).mean(),
                "old_2swap_ceiling_gain_max": (
                    group.old_2swap_ceiling_score - group.start_score
                ).max(),
            }
        )
    _write_csv(ANALYSIS / "support_gap_summary.csv", support_rows)

    policy_rows = []
    old = runs[runs.method == "old_mask"]
    for (k, phase), group in old.groupby(["k", "phase"], sort=True):
        policy_rows.append(
            {
                "k": int(k),
                "phase": phase,
                "n_runs": len(group),
                "search_policy_gap_at_64_mean": group.search_policy_gap_at_64.mean(),
                "search_policy_gap_at_64_sd": group.search_policy_gap_at_64.std(ddof=1),
                "search_policy_gap_at_64_max": group.search_policy_gap_at_64.max(),
                "ceiling_reached_runs": int((group.search_policy_gap_at_64 <= EPS).sum()),
            }
        )
    _write_csv(ANALYSIS / "policy_gap_summary.csv", policy_rows)

    paired_rows = []
    for (k, phase), group in paired.groupby(["k", "phase"], sort=True):
        old_counts = signed_counts(group.quality_plus_global_minus_old)
        uniform_counts = signed_counts(group.quality_plus_global_minus_uniform)
        paired_rows.append(
            {
                "k": int(k),
                "phase": phase,
                "n_paired_runs": len(group),
                "new_minus_old_mean": group.quality_plus_global_minus_old.mean(),
                "new_minus_old_sd": group.quality_plus_global_minus_old.std(ddof=1),
                "new_vs_old_win": old_counts["positive"],
                "new_vs_old_tie": old_counts["zero"],
                "new_vs_old_loss": old_counts["negative"],
                "new_minus_uniform_mean": group.quality_plus_global_minus_uniform.mean(),
                "new_minus_uniform_sd": group.quality_plus_global_minus_uniform.std(ddof=1),
                "new_vs_uniform_win": uniform_counts["positive"],
                "new_vs_uniform_tie": uniform_counts["zero"],
                "new_vs_uniform_loss": uniform_counts["negative"],
            }
        )
    _write_csv(ANALYSIS / "paired_method_summary.csv", paired_rows)

    palette = {
        "old_mask": "#3B5BA5",
        "quality_plus_global": "#D18F1D",
        "uniform_legal": "#767676",
    }
    labels = {
        "old_mask": "Old mask",
        "quality_plus_global": "Quality + global",
        "uniform_legal": "Uniform legal",
    }
    styles = {"old_mask": "-", "quality_plus_global": "--", "uniform_legal": ":"}
    aggregated = (
        curves.groupby(["k", "phase", "method", "unique_subset_budget"], as_index=False)
        .best_so_far_gain.mean()
        .sort_values(["k", "phase", "method", "unique_subset_budget"])
    )
    figure, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), sharex=True, sharey=True)
    for row, phase in enumerate(("forward", "local")):
        for column, k in enumerate((8, 16, 32)):
            axis = axes[row, column]
            panel = aggregated[(aggregated.k == k) & (aggregated.phase == phase)]
            for method in ("old_mask", "quality_plus_global", "uniform_legal"):
                values = panel[panel.method == method]
                axis.plot(
                    values.unique_subset_budget,
                    100.0 * values.best_so_far_gain,
                    label=labels[method],
                    color=palette[method],
                    linestyle=styles[method],
                    linewidth=2.0,
                )
            axis.set_title(f"K={k} · {phase}")
            axis.grid(axis="y", color="#D9D9D9", linewidth=0.8)
            axis.spines[["top", "right"]].set_visible(False)
            if row == 1:
                axis.set_xlabel("Incremental unique scored subsets")
            if column == 0:
                axis.set_ylabel("Mean best-so-far gain (pp)")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.947),
        ncol=3,
        frameon=False,
    )
    figure.suptitle("Task 4A matched-budget proposal replay", y=0.992, fontsize=15)
    figure.text(
        0.5,
        0.01,
        "Mean over 25 seed×replay combinations per panel; spread is descriptive, "
        "not independent-dataset uncertainty.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    figure.tight_layout(rect=(0.02, 0.045, 0.98, 0.89))
    figure.savefig(ANALYSIS / "budget_curves.png", dpi=180)
    figure.savefig(ANALYSIS / "budget_curves.pdf")
    plt.close(figure)

    summary = {
        "support_gap_summary": support_rows,
        "policy_gap_summary": policy_rows,
        "paired_method_summary": paired_rows,
        "figure": str(ANALYSIS / "budget_curves.png"),
        "units": "stored scores are fractions; report and figure gains use percentage points",
        "uncertainty_note": "SD is descriptive across overlapping outer splits and replay seeds",
        "historical_policy_gap": "NA; no matched historical policy ran on the 30 origins",
    }
    _write_json(ANALYSIS / "summary.json", json.loads(json.dumps(summary)))
    print("wrote Task 4A descriptive summaries and budget curves")


if __name__ == "__main__":
    main()
