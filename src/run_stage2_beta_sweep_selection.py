#!/usr/bin/env python3
"""Scan beta for one configured Full-IRFS variant without touching the sealed outer test."""

from __future__ import annotations

import dataclasses
import json
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from config import IrfsConfig
from harness.contract import SelectionContext
from methods.reinforced_run import _rng_from_snapshot, _rng_snapshot, build_reinforced_engine
from run_stage2_rl_selection import (
    _config_for_encoder,
    _flat_trajectory_row,
    _jaccard,
    _summary,
    _trajectory_rows,
    _trajectory_summary,
    _write_csv,
    _write_json,
)
from stage2_cv import build_stage2_cv_context
from stage2_rl_config import (
    BETA_SWEEP_SEEDS,
    BETA_SWEEP_SELECTION_ROOT,
    BETA_SWEEP_TABLE_PREFIX,
    BETA_SWEEP_VALUES,
    DATASET,
    EXPECTED_CLEAN_FEATURES,
    EXPLORATION_STEP_BUDGET,
    INNER_CV_FOLDS,
    RESUME_COMPLETED_SELECTIONS,
    TABLE_ROOT,
    TRAJECTORY_ROLLING_WINDOW,
)

PROTOCOL_VERSION = 1
REPORT_NAME = "full_irfs_fixed"
ENGINE_NAME = "full_irfs"
STATE_ENCODER = "fixed"


def configure_variant(
    *,
    report_name: str,
    state_encoder: str,
    selection_root: Path,
    table_prefix: str,
) -> None:
    """Configure one code-defined sweep variant before calling main."""
    if state_encoder not in {"fixed", "trained_gcn"}:
        raise ValueError(f"unsupported state encoder: {state_encoder}")
    global REPORT_NAME, STATE_ENCODER, BETA_SWEEP_SELECTION_ROOT, BETA_SWEEP_TABLE_PREFIX
    REPORT_NAME = report_name
    STATE_ENCODER = state_encoder
    BETA_SWEEP_SELECTION_ROOT = selection_root
    BETA_SWEEP_TABLE_PREFIX = table_prefix


def beta_tag(beta: float) -> str:
    """Return a stable path/table label such as beta_0p02."""
    value = format(float(beta), ".12g")
    return "beta_" + value.replace("-", "m").replace(".", "p")


def _config_for_beta(beta: float) -> IrfsConfig:
    if beta < 0:
        raise ValueError("beta must be non-negative")
    return dataclasses.replace(
        _config_for_encoder(STATE_ENCODER),
        correlation_penalty_weight=float(beta),
    )


def _selection_signature(seed: int, beta: float) -> dict[str, Any]:
    config = _config_for_beta(beta)
    effective_config = dataclasses.asdict(config)
    # 预算项默认关闭时保持旧 beta/guidance 产物的签名兼容性。
    if effective_config["feature_budget"] is None:
        effective_config.pop("feature_budget")
        effective_config.pop("over_budget_penalty_weight")
    normalized_config = json.loads(json.dumps(effective_config))
    return {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": DATASET,
        "seed": int(seed),
        "report_name": REPORT_NAME,
        "engine_name": ENGINE_NAME,
        "state_encoder": STATE_ENCODER,
        "beta": float(beta),
        "effective_irfs_config": normalized_config,
        "inner_cv_folds": INNER_CV_FOLDS,
        "selection_tie_break": "higher mean inner-CV accuracy, then fewer features",
    }


def _selection_path(seed: int, beta: float) -> Path:
    return BETA_SWEEP_SELECTION_ROOT / beta_tag(beta) / f"seed-{seed}" / "selection.json"


def _load_matching(path: Path, signature: dict[str, Any]) -> dict[str, Any] | None:
    if not RESUME_COMPLETED_SELECTIONS or not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        artifact = json.load(handle)
    if artifact.get("experiment_signature") != signature:
        return None
    if len(artifact.get("trajectory", [])) != EXPLORATION_STEP_BUDGET:
        return None
    if not artifact.get("selected_clean_indices"):
        return None
    return artifact


def _average_pairwise_abs_correlation(X: np.ndarray, subset: Sequence[int]) -> float:
    indices = np.asarray(tuple(subset), dtype=int)
    if indices.size < 2:
        return 0.0
    matrix = np.corrcoef(X[:, indices], rowvar=False)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    upper = np.abs(matrix[np.triu_indices(indices.size, k=1)])
    return float(np.mean(upper)) if upper.size else 0.0


def _run_one(
    base_context: SelectionContext,
    rng_snapshot: tuple,
    *,
    seed: int,
    beta: float,
    original_feature_ids: Sequence[int],
) -> dict[str, Any]:
    config = _config_for_beta(beta)
    signature = _selection_signature(seed, beta)
    path = _selection_path(seed, beta)
    existing = _load_matching(path, signature)
    if existing is not None:
        print(f"seed={seed} beta={beta:g} resume: {path}", flush=True)
        return existing

    context = base_context._replace(
        config=config,
        rng=_rng_from_snapshot(rng_snapshot, seed),
    )
    engine = build_reinforced_engine(ENGINE_NAME, config)
    started = time.perf_counter()
    elapsed_by_step: list[float] = []
    heartbeat_every = max(1, EXPLORATION_STEP_BUDGET // 10)

    def on_step(step: int, budget: int, accuracy: float, best_accuracy: float) -> None:
        elapsed_by_step.append(time.perf_counter() - started)
        if (step + 1) % heartbeat_every == 0 or step + 1 == budget:
            print(
                f"seed={seed} beta={beta:g} step={step + 1:>3}/{budget} "
                f"dt_cv={accuracy:.4f} best={best_accuracy:.4f} "
                f"elapsed={elapsed_by_step[-1]:.0f}s",
                flush=True,
            )

    print(
        f"seed={seed} beta={beta:g} start method={REPORT_NAME} budget={EXPLORATION_STEP_BUDGET}",
        flush=True,
    )
    selection = engine.select(context, on_step=on_step)
    elapsed = time.perf_counter() - started
    if len(selection.per_step) != EXPLORATION_STEP_BUDGET:
        raise RuntimeError(
            f"beta={beta:g} seed={seed} produced {len(selection.per_step)} steps; "
            f"expected {EXPLORATION_STEP_BUDGET}"
        )
    if len(elapsed_by_step) != len(selection.per_step):
        raise RuntimeError("per-step timing hook did not observe every RL step")

    fold_accuracies_by_step = [
        context.probe.fold_accuracies(record.subset, context.split.validation)
        for record in selection.per_step
    ]
    trajectory = _trajectory_rows(
        selection.per_step,
        selected_subset=selection.selected,
        original_feature_ids=original_feature_ids,
        elapsed_by_step=elapsed_by_step,
        rolling_window=TRAJECTORY_ROLLING_WINDOW,
        fold_accuracies_by_step=fold_accuracies_by_step,
    )
    best_accuracy = max(float(record.accuracy) for record in selection.per_step)
    average_correlation = _average_pairwise_abs_correlation(
        context.split.train.X,
        selection.selected,
    )
    artifact = {
        "experiment_signature": signature,
        "protocol": {
            "stage": "stage2_beta_sweep_selection_only",
            "method": REPORT_NAME,
            "search_model": "DecisionTreeClassifier",
            "reward_beta": float(beta),
            "development_rows": int(context.split.train.X.shape[0]),
            "inner_cv_folds": INNER_CV_FOLDS,
            "official_test_accessed": False,
            "held_out_random_test_accessed": False,
            "outer_test_release_permitted": False,
            "selected_subset_rule": "maximum mean DT inner-CV accuracy; ties use fewer features",
        },
        "dataset_metadata": context.split.train.metadata,
        "split_indices": {
            "development_inner_cv": context.split.train.indices.astype(int).tolist(),
            "inner_cv_folds": context.probe.fold_indices(),
        },
        "selected_clean_indices": [int(index) for index in selection.selected],
        "selected_original_feature_ids": [int(original_feature_ids[index]) for index in selection.selected],
        "selected_count": len(selection.selected),
        "compression_ratio": float(1.0 - len(selection.selected) / len(original_feature_ids)),
        "best_dt_inner_cv_accuracy": best_accuracy,
        "selected_average_pairwise_abs_correlation": average_correlation,
        "selected_accuracy_minus_beta_correlation": float(best_accuracy - beta * average_correlation),
        "selection_elapsed_seconds": elapsed,
        "trajectory_summary": _trajectory_summary(trajectory),
        "trajectory": trajectory,
    }
    _write_json(artifact, path)
    _write_csv(
        [
            {
                "beta": beta,
                **_flat_trajectory_row(seed, beta_tag(beta), row),
            }
            for row in trajectory
        ],
        path.parent / "trajectory.csv",
    )
    print(
        f"seed={seed} beta={beta:g} done features={len(selection.selected)} "
        f"best_dt_cv={best_accuracy:.4f} corr={average_correlation:.4f} "
        f"elapsed={elapsed:.0f}s",
        flush=True,
    )
    return artifact


def _aggregate(
    artifacts: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    per_seed_rows: list[dict[str, Any]] = []

    for beta in BETA_SWEEP_VALUES:
        selected = [
            artifact for artifact in artifacts if artifact["experiment_signature"]["beta"] == float(beta)
        ]
        selected.sort(key=lambda item: item["experiment_signature"]["seed"])
        subsets = [artifact["selected_clean_indices"] for artifact in selected]
        jaccards = [_jaccard(left, right) for left, right in combinations(subsets, 2)]
        summaries.append(
            {
                "beta": float(beta),
                "n_seeds": len(selected),
                "selected_count": _summary([artifact["selected_count"] for artifact in selected]),
                "best_dt_inner_cv_accuracy": _summary(
                    [artifact["best_dt_inner_cv_accuracy"] for artifact in selected]
                ),
                "selected_average_pairwise_abs_correlation": _summary(
                    [artifact["selected_average_pairwise_abs_correlation"] for artifact in selected]
                ),
                "selected_accuracy_minus_beta_correlation": _summary(
                    [artifact["selected_accuracy_minus_beta_correlation"] for artifact in selected]
                ),
                "selection_elapsed_seconds": _summary(
                    [artifact["selection_elapsed_seconds"] for artifact in selected]
                ),
                "selection_stability_pairwise_jaccard": _summary(jaccards),
            }
        )
        for artifact in selected:
            per_seed_rows.append(
                {
                    "seed": artifact["experiment_signature"]["seed"],
                    "beta": float(beta),
                    "selected_count": artifact["selected_count"],
                    "best_dt_inner_cv_accuracy": artifact["best_dt_inner_cv_accuracy"],
                    "selected_average_pairwise_abs_correlation": artifact[
                        "selected_average_pairwise_abs_correlation"
                    ],
                    "selected_accuracy_minus_beta_correlation": artifact[
                        "selected_accuracy_minus_beta_correlation"
                    ],
                    "selection_elapsed_seconds": artifact["selection_elapsed_seconds"],
                    "selected_original_feature_ids": ";".join(
                        str(value) for value in artifact["selected_original_feature_ids"]
                    ),
                }
            )

    aggregate = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": DATASET,
        "method": REPORT_NAME,
        "betas": [float(beta) for beta in BETA_SWEEP_VALUES],
        "seeds": list(BETA_SWEEP_SEEDS),
        "outer_test_accessed": False,
        "beta_summaries": summaries,
    }
    return aggregate, per_seed_rows


def _aggregate_csv_rows(aggregate: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "beta": item["beta"],
            "n_seeds": item["n_seeds"],
            "selected_count_mean": item["selected_count"]["mean"],
            "selected_count_std": item["selected_count"]["std"],
            "best_dt_inner_cv_accuracy_mean": item["best_dt_inner_cv_accuracy"]["mean"],
            "best_dt_inner_cv_accuracy_std": item["best_dt_inner_cv_accuracy"]["std"],
            "selected_average_pairwise_abs_correlation_mean": item[
                "selected_average_pairwise_abs_correlation"
            ]["mean"],
            "selected_accuracy_minus_beta_correlation_mean": item["selected_accuracy_minus_beta_correlation"][
                "mean"
            ],
            "selection_elapsed_seconds_mean": item["selection_elapsed_seconds"]["mean"],
            "selection_jaccard_mean": item["selection_stability_pairwise_jaccard"]["mean"],
        }
        for item in aggregate["beta_summaries"]
    ]


def main() -> None:
    base_config = _config_for_beta(BETA_SWEEP_VALUES[0])
    first_context = build_stage2_cv_context(
        base_config,
        seed=BETA_SWEEP_SEEDS[0],
        n_splits=INNER_CV_FOLDS,
    )
    if first_context.n_features != EXPECTED_CLEAN_FEATURES:
        raise ValueError(
            f"expected {EXPECTED_CLEAN_FEATURES} cleaned features, got {first_context.n_features}"
        )
    metadata = first_context.split.train.metadata
    if metadata is None:
        raise ValueError("radar dataset metadata is required")
    original_feature_ids = [int(value) for value in metadata["final_feature_ids"]]

    print(
        f"beta sweep selection: method={REPORT_NAME} "
        f"betas={list(BETA_SWEEP_VALUES)} seeds={list(BETA_SWEEP_SEEDS)} "
        f"budget={EXPLORATION_STEP_BUDGET}; outer test sealed",
        flush=True,
    )
    artifacts: list[dict[str, Any]] = []
    for seed_index, seed in enumerate(BETA_SWEEP_SEEDS):
        context = (
            first_context
            if seed_index == 0
            else build_stage2_cv_context(
                base_config,
                seed=seed,
                n_splits=INNER_CV_FOLDS,
            )
        )
        snapshot = _rng_snapshot(context.rng)
        for beta in BETA_SWEEP_VALUES:
            artifacts.append(
                _run_one(
                    context,
                    snapshot,
                    seed=seed,
                    beta=beta,
                    original_feature_ids=original_feature_ids,
                )
            )

    aggregate, per_seed_rows = _aggregate(artifacts)
    aggregate_rows = _aggregate_csv_rows(aggregate)
    trajectory_rows = [
        {
            "beta": artifact["experiment_signature"]["beta"],
            **_flat_trajectory_row(
                artifact["experiment_signature"]["seed"],
                beta_tag(artifact["experiment_signature"]["beta"]),
                row,
            ),
        }
        for artifact in artifacts
        for row in artifact["trajectory"]
    ]

    _write_json(aggregate, BETA_SWEEP_SELECTION_ROOT / "aggregate.json")
    _write_csv(per_seed_rows, BETA_SWEEP_SELECTION_ROOT / "per_seed_selection.csv")
    _write_csv(aggregate_rows, BETA_SWEEP_SELECTION_ROOT / "aggregate.csv")
    _write_csv(
        trajectory_rows,
        BETA_SWEEP_SELECTION_ROOT / "trajectory_all_runs.csv",
    )
    _write_csv(
        per_seed_rows,
        TABLE_ROOT / f"{BETA_SWEEP_TABLE_PREFIX}_selection_per_seed.csv",
    )
    _write_csv(
        aggregate_rows,
        TABLE_ROOT / f"{BETA_SWEEP_TABLE_PREFIX}_selection_aggregate.csv",
    )
    print(
        f"all {len(artifacts)} selections complete; aggregate={BETA_SWEEP_SELECTION_ROOT / 'aggregate.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
