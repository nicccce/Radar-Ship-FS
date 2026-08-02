#!/usr/bin/env python3
"""Sweep the soft over-budget penalty, then compare feasible selections on held-out DT test."""

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
from harness.orchestrator import build_run_context
from methods.reinforced_run import _rng_from_snapshot, _rng_snapshot, build_reinforced_engine
from probe import DecisionTreeProbe
from run_stage2_beta_sweep_selection import _average_pairwise_abs_correlation
from run_stage2_dt_test import _development_partition, _mi_ranking, _top_k
from run_stage2_rl_selection import _config_for_encoder, _jaccard, _summary, _write_csv, _write_json
from stage2_cv import build_stage2_cv_context
from stage2_rl_config import (
    BUDGET_SWEEP_BETA,
    BUDGET_SWEEP_DT_TEST_ROOT,
    BUDGET_SWEEP_FEATURE_BUDGET,
    BUDGET_SWEEP_SEEDS,
    BUDGET_SWEEP_SELECTION_ROOT,
    BUDGET_SWEEP_TABLE_PREFIX,
    BUDGET_SWEEP_VALUES,
    DATASET,
    EXPECTED_CLEAN_FEATURES,
    EXPLORATION_STEP_BUDGET,
    INNER_CV_FOLDS,
    K_BEST,
    RESUME_COMPLETED_SELECTIONS,
    TABLE_ROOT,
)

PROTOCOL_VERSION = 2
REPORT_NAME = "full_irfs_fixed_budget_penalty"
ENGINE_NAME = "full_irfs"
STATE_ENCODER = "fixed"


def lambda_tag(value: float) -> str:
    """Return a stable path/table label such as lambda_0p025."""
    formatted = format(float(value), ".12g")
    return "lambda_" + formatted.replace("-", "m").replace(".", "p")


def _config_for_lambda(value: float) -> IrfsConfig:
    """Build the fixed Full-IRFS config for one non-negative lambda."""
    if value < 0:
        raise ValueError("lambda must be non-negative")
    return dataclasses.replace(
        _config_for_encoder(STATE_ENCODER),
        correlation_penalty_weight=BUDGET_SWEEP_BETA,
        feature_budget=BUDGET_SWEEP_FEATURE_BUDGET,
        over_budget_penalty_weight=float(value),
    )


def _selection_signature(seed: int, value: float) -> dict[str, Any]:
    config = _config_for_lambda(value)
    normalized_config = json.loads(json.dumps(dataclasses.asdict(config)))
    return {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": DATASET,
        "seed": int(seed),
        "report_name": REPORT_NAME,
        "engine_name": ENGINE_NAME,
        "state_encoder": STATE_ENCODER,
        "beta": float(BUDGET_SWEEP_BETA),
        "lambda": float(value),
        "feature_budget": int(BUDGET_SWEEP_FEATURE_BUDGET),
        "effective_irfs_config": normalized_config,
        "inner_cv_folds": INNER_CV_FOLDS,
        "selection_tie_break": (
            f"among initial and trajectory candidates with |S|<={BUDGET_SWEEP_FEATURE_BUDGET}: "
            "higher mean inner-CV accuracy, then fewer features"
        ),
    }


def _selection_path(seed: int, value: float) -> Path:
    return BUDGET_SWEEP_SELECTION_ROOT / lambda_tag(value) / f"seed-{seed}" / "selection.json"


def _load_matching(path: Path, signature: dict[str, Any]) -> dict[str, Any] | None:
    if not RESUME_COMPLETED_SELECTIONS or not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        artifact = json.load(handle)
    if artifact.get("experiment_signature") != signature:
        return None
    protocol = artifact.get("protocol", {})
    if protocol.get("test_used_during_selection") is not False:
        return None

    if len(artifact.get("trajectory", [])) != EXPLORATION_STEP_BUDGET:
        return None
    if not artifact.get("selected_clean_indices"):
        return None
    if artifact.get("selected_count", BUDGET_SWEEP_FEATURE_BUDGET + 1) > BUDGET_SWEEP_FEATURE_BUDGET:
        return None
    initial = artifact.get("initial_candidate", {})
    if initial.get("included_in_final_candidate_pool") is not True:
        return None
    if initial.get("selected_count", BUDGET_SWEEP_FEATURE_BUDGET + 1) > BUDGET_SWEEP_FEATURE_BUDGET:
        return None
    return artifact


def _trajectory_rows(
    selection,
    *,
    original_feature_ids: Sequence[int],
    elapsed_by_step: Sequence[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selected = tuple(selection.selected)
    for step, (record, elapsed) in enumerate(zip(selection.per_step, elapsed_by_step), start=1):
        subset = tuple(record.subset)
        rows.append(
            {
                "step": step,
                "dt_inner_cv_accuracy": float(record.accuracy),
                "selected_count": len(subset),
                "budget_eligible": len(subset) <= BUDGET_SWEEP_FEATURE_BUDGET,
                "is_final_selected": subset == selected,
                "selected_clean_indices": list(subset),
                "selected_original_feature_ids": [int(original_feature_ids[index]) for index in subset],
                "elapsed_seconds": float(elapsed),
            }
        )
    return rows


def _run_one(
    base_context: SelectionContext,
    rng_snapshot: tuple,
    *,
    seed: int,
    value: float,
    original_feature_ids: Sequence[int],
) -> dict[str, Any]:
    config = _config_for_lambda(value)
    signature = _selection_signature(seed, value)
    path = _selection_path(seed, value)
    existing = _load_matching(path, signature)
    if existing is not None:
        print(f"seed={seed} lambda={value:g} resume: {path}", flush=True)
        return existing

    context = base_context._replace(
        config=config,
        rng=_rng_from_snapshot(rng_snapshot, seed),
    )
    engine = build_reinforced_engine(ENGINE_NAME, config)

    initial_observation: list[tuple[tuple[int, ...], float]] = []

    def on_initial(subset: Sequence[int], accuracy: float | None) -> None:
        if accuracy is None:
            raise RuntimeError("budget-aware initial candidate was not scored")
        initial_observation.append((tuple(int(index) for index in subset), float(accuracy)))

    started = time.perf_counter()
    elapsed_by_step: list[float] = []
    heartbeat_every = max(1, EXPLORATION_STEP_BUDGET // 10)

    def on_step(step: int, budget: int, accuracy: float, best_accuracy: float) -> None:
        elapsed_by_step.append(time.perf_counter() - started)
        if (step + 1) % heartbeat_every == 0 or step + 1 == budget:
            print(
                f"seed={seed} lambda={value:g} step={step + 1:>3}/{budget} "
                f"dt_cv={accuracy:.4f} feasible_best={best_accuracy:.4f} "
                f"elapsed={elapsed_by_step[-1]:.0f}s",
                flush=True,
            )

    print(
        f"seed={seed} lambda={value:g} start beta={BUDGET_SWEEP_BETA:g} "
        f"k={BUDGET_SWEEP_FEATURE_BUDGET} budget={EXPLORATION_STEP_BUDGET}",
        flush=True,
    )
    selection = engine.select(context, on_step=on_step, on_initial=on_initial)
    elapsed = time.perf_counter() - started
    if len(initial_observation) != 1:
        raise RuntimeError("engine did not report exactly one initial candidate")
    initial_subset, initial_accuracy = initial_observation[0]
    if len(initial_subset) > BUDGET_SWEEP_FEATURE_BUDGET:
        raise RuntimeError("initial subset is not budget-eligible")

    if len(selection.per_step) != EXPLORATION_STEP_BUDGET:
        raise RuntimeError(
            f"lambda={value:g} seed={seed} produced {len(selection.per_step)} steps; "
            f"expected {EXPLORATION_STEP_BUDGET}"
        )
    if len(elapsed_by_step) != len(selection.per_step):
        raise RuntimeError("per-step timing hook did not observe every RL step")
    if len(selection.selected) > BUDGET_SWEEP_FEATURE_BUDGET:
        raise RuntimeError("final selection exceeded the hard candidate budget")

    trajectory = _trajectory_rows(
        selection,
        original_feature_ids=original_feature_ids,
        elapsed_by_step=elapsed_by_step,
    )
    selected_accuracy = float(context.probe.probe(selection.selected, context.split.validation).accuracy)
    average_correlation = _average_pairwise_abs_correlation(
        context.split.train.X,
        selection.selected,
    )
    eligible_subsets = {initial_subset}
    eligible_subsets.update(
        tuple(record.subset)
        for record in selection.per_step
        if len(record.subset) <= BUDGET_SWEEP_FEATURE_BUDGET
    )
    selected_source = "initial_subset" if tuple(selection.selected) == initial_subset else "trajectory"
    excess_ratio = max(
        0.0,
        (len(selection.selected) - BUDGET_SWEEP_FEATURE_BUDGET) / BUDGET_SWEEP_FEATURE_BUDGET,
    )
    selected_objective = selected_accuracy - BUDGET_SWEEP_BETA * average_correlation - value * excess_ratio

    artifact = {
        "experiment_signature": signature,
        "protocol": {
            "stage": "stage2_budget_penalty_selection_only",
            "method": REPORT_NAME,
            "search_model": "DecisionTreeClassifier",
            "reward": (
                "mean inner-CV accuracy - beta*correlation - "
                "lambda*max(0,(|S|-feature_budget)/feature_budget)"
            ),
            "development_rows": int(context.split.train.X.shape[0]),
            "inner_cv_folds": INNER_CV_FOLDS,
            "test_used_during_selection": False,
            "selected_subset_rule": (
                "maximum mean DT inner-CV accuracy among initial and trajectory candidates "
                f"with |S|<={BUDGET_SWEEP_FEATURE_BUDGET}; ties use fewer features"
            ),
        },
        "dataset_metadata": context.split.train.metadata,
        "split_indices": {
            "development_inner_cv": context.split.train.indices.astype(int).tolist(),
            "inner_cv_folds": context.probe.fold_indices(),
        },
        "initial_candidate": {
            "included_in_final_candidate_pool": True,
            "selected_clean_indices": list(initial_subset),
            "selected_original_feature_ids": [int(original_feature_ids[index]) for index in initial_subset],
            "selected_count": len(initial_subset),
            "dt_inner_cv_accuracy": initial_accuracy,
        },
        "selected_source": selected_source,
        "selected_clean_indices": [int(index) for index in selection.selected],
        "selected_original_feature_ids": [int(original_feature_ids[index]) for index in selection.selected],
        "selected_count": len(selection.selected),
        "compression_ratio": float(1.0 - len(selection.selected) / len(original_feature_ids)),
        "best_feasible_dt_inner_cv_accuracy": selected_accuracy,
        "selected_average_pairwise_abs_correlation": average_correlation,
        "selected_reward_objective": float(selected_objective),
        "eligible_candidate_occurrences": int(1 + sum(row["budget_eligible"] for row in trajectory)),
        "unique_eligible_candidates": len(eligible_subsets),
        "selection_elapsed_seconds": elapsed,
        "trajectory": trajectory,
    }
    _write_json(artifact, path)
    _write_csv(
        [{"seed": seed, "lambda": value, **row} for row in trajectory],
        path.parent / "trajectory.csv",
    )
    print(
        f"seed={seed} lambda={value:g} done features={len(selection.selected)} "
        f"feasible_dt_cv={selected_accuracy:.4f} source={selected_source} "
        f"eligible_unique={len(eligible_subsets)} elapsed={elapsed:.0f}s",
        flush=True,
    )
    return artifact


def _selection_aggregate(
    artifacts: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    per_seed_rows: list[dict[str, Any]] = []
    for value in BUDGET_SWEEP_VALUES:
        selected = [
            artifact for artifact in artifacts if artifact["experiment_signature"]["lambda"] == float(value)
        ]
        selected.sort(key=lambda item: item["experiment_signature"]["seed"])
        subsets = [artifact["selected_clean_indices"] for artifact in selected]
        jaccards = [_jaccard(left, right) for left, right in combinations(subsets, 2)]
        summaries.append(
            {
                "lambda": float(value),
                "n_seeds": len(selected),
                "selected_count": _summary([artifact["selected_count"] for artifact in selected]),
                "best_feasible_dt_inner_cv_accuracy": _summary(
                    [artifact["best_feasible_dt_inner_cv_accuracy"] for artifact in selected]
                ),
                "selected_average_pairwise_abs_correlation": _summary(
                    [artifact["selected_average_pairwise_abs_correlation"] for artifact in selected]
                ),
                "unique_eligible_candidates": _summary(
                    [artifact["unique_eligible_candidates"] for artifact in selected]
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
                    "lambda": float(value),
                    "selected_count": artifact["selected_count"],
                    "selected_source": artifact["selected_source"],
                    "best_feasible_dt_inner_cv_accuracy": artifact["best_feasible_dt_inner_cv_accuracy"],
                    "selected_average_pairwise_abs_correlation": artifact[
                        "selected_average_pairwise_abs_correlation"
                    ],
                    "eligible_candidate_occurrences": artifact["eligible_candidate_occurrences"],
                    "unique_eligible_candidates": artifact["unique_eligible_candidates"],
                    "selection_elapsed_seconds": artifact["selection_elapsed_seconds"],
                    "selected_original_feature_ids": ";".join(
                        str(item) for item in artifact["selected_original_feature_ids"]
                    ),
                }
            )
    return (
        {
            "protocol_version": PROTOCOL_VERSION,
            "dataset": DATASET,
            "method": REPORT_NAME,
            "beta": BUDGET_SWEEP_BETA,
            "feature_budget": BUDGET_SWEEP_FEATURE_BUDGET,
            "lambdas": [float(value) for value in BUDGET_SWEEP_VALUES],
            "seeds": list(BUDGET_SWEEP_SEEDS),
            "outer_test_accessed": False,
            "lambda_summaries": summaries,
        },
        per_seed_rows,
    )


def run_selection() -> None:
    """Finish every lambda/seed RL selection on source train data."""
    base_config = _config_for_lambda(BUDGET_SWEEP_VALUES[0])
    first_context = build_stage2_cv_context(
        base_config,
        seed=BUDGET_SWEEP_SEEDS[0],
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
        f"budget sweep selection: beta={BUDGET_SWEEP_BETA:g} "
        f"lambdas={list(BUDGET_SWEEP_VALUES)} seeds={list(BUDGET_SWEEP_SEEDS)} "
        f"k={BUDGET_SWEEP_FEATURE_BUDGET}; source test unused during selection",
        flush=True,
    )
    artifacts: list[dict[str, Any]] = []
    for seed_index, seed in enumerate(BUDGET_SWEEP_SEEDS):
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
        for value in BUDGET_SWEEP_VALUES:
            artifacts.append(
                _run_one(
                    context,
                    snapshot,
                    seed=seed,
                    value=value,
                    original_feature_ids=original_feature_ids,
                )
            )

    aggregate, per_seed_rows = _selection_aggregate(artifacts)
    aggregate_rows = [
        {
            "lambda": item["lambda"],
            "n_seeds": item["n_seeds"],
            "selected_count_mean": item["selected_count"]["mean"],
            "selected_count_std": item["selected_count"]["std"],
            "best_feasible_dt_inner_cv_accuracy_mean": item["best_feasible_dt_inner_cv_accuracy"]["mean"],
            "best_feasible_dt_inner_cv_accuracy_std": item["best_feasible_dt_inner_cv_accuracy"]["std"],
            "selected_average_pairwise_abs_correlation_mean": item[
                "selected_average_pairwise_abs_correlation"
            ]["mean"],
            "unique_eligible_candidates_mean": item["unique_eligible_candidates"]["mean"],
            "selection_elapsed_seconds_mean": item["selection_elapsed_seconds"]["mean"],
            "selection_jaccard_mean": item["selection_stability_pairwise_jaccard"]["mean"],
        }
        for item in aggregate["lambda_summaries"]
    ]
    _write_json(aggregate, BUDGET_SWEEP_SELECTION_ROOT / "aggregate.json")
    _write_csv(per_seed_rows, BUDGET_SWEEP_SELECTION_ROOT / "per_seed_selection.csv")
    _write_csv(aggregate_rows, BUDGET_SWEEP_SELECTION_ROOT / "aggregate.csv")
    _write_csv(
        per_seed_rows,
        TABLE_ROOT / f"{BUDGET_SWEEP_TABLE_PREFIX}_selection_per_seed.csv",
    )
    _write_csv(
        aggregate_rows,
        TABLE_ROOT / f"{BUDGET_SWEEP_TABLE_PREFIX}_selection_aggregate.csv",
    )


def _load_complete_selection(seed: int, value: float) -> dict[str, Any]:
    path = _selection_path(seed, value)
    if not path.is_file():
        raise FileNotFoundError(f"missing budget-sweep selection: {path}")
    with path.open("r", encoding="utf-8") as handle:
        artifact = json.load(handle)
    if artifact.get("experiment_signature") != _selection_signature(seed, value):
        raise ValueError(f"selection signature mismatch: {path}")
    protocol = artifact.get("protocol", {})
    if protocol.get("test_used_during_selection") is not False:
        raise ValueError(f"selection used test data: {path}")
    if len(artifact.get("trajectory", [])) != EXPLORATION_STEP_BUDGET:
        raise ValueError(f"incomplete selection trajectory: {path}")
    if not artifact.get("selected_clean_indices"):
        raise ValueError(f"selection contains no features: {path}")
    if artifact.get("selected_count", BUDGET_SWEEP_FEATURE_BUDGET + 1) > BUDGET_SWEEP_FEATURE_BUDGET:
        raise ValueError(f"selection exceeds feature budget: {path}")
    initial = artifact.get("initial_candidate", {})
    if initial.get("included_in_final_candidate_pool") is not True:
        raise ValueError(f"initial candidate was not included: {path}")
    if initial.get("selected_count", BUDGET_SWEEP_FEATURE_BUDGET + 1) > BUDGET_SWEEP_FEATURE_BUDGET:
        raise ValueError(f"initial candidate exceeds feature budget: {path}")
    return artifact


def _require_complete_sweep() -> dict[tuple[int, float], dict[str, Any]]:
    """Validate all lambda/seed selection artifacts before test evaluation."""
    complete: dict[tuple[int, float], dict[str, Any]] = {}
    errors: list[str] = []
    for seed in BUDGET_SWEEP_SEEDS:
        for value in BUDGET_SWEEP_VALUES:
            try:
                complete[(seed, float(value))] = _load_complete_selection(seed, value)
            except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
                errors.append(str(exc))
    if errors:
        detail = "\n".join(f"- {error}" for error in errors)
        raise RuntimeError("outer DT test blocked until every lambda/seed selection is complete:\n" + detail)
    return complete


def _run_seed_dt(
    seed: int,
    complete: dict[tuple[int, float], dict[str, Any]],
) -> dict[str, Any]:
    config = _config_for_lambda(BUDGET_SWEEP_VALUES[0])
    context = build_run_context(config, seed=seed)
    if context.n_features != EXPECTED_CLEAN_FEATURES:
        raise ValueError(f"expected {EXPECTED_CLEAN_FEATURES} cleaned features, got {context.n_features}")
    metadata = context.split.train.metadata
    if metadata is None:
        raise ValueError("radar dataset metadata is required")
    original_ids = [int(value) for value in metadata["final_feature_ids"]]
    development = _development_partition(context)

    selections: dict[float, dict[str, Any]] = {}
    for value in BUDGET_SWEEP_VALUES:
        artifact = complete[(seed, float(value))]
        saved_indices = artifact.get("split_indices", {}).get("development_inner_cv")
        if saved_indices != development.indices.astype(int).tolist():
            raise ValueError(f"development rows differ for seed={seed}, lambda={value:g}")
        selections[float(value)] = artifact

    test = context.split.test
    ranked, mi_scores = _mi_ranking(development.X, development.y, seed)
    final_probe = DecisionTreeProbe(development, config, context.rng)
    candidates: list[tuple[str, tuple[int, ...], str, float | None, float | None]] = [
        (
            "all_features",
            tuple(range(context.n_features)),
            "all cleaned features",
            None,
            None,
        ),
        (
            "kbest_mutual_info",
            _top_k(ranked, K_BEST),
            f"MI fit on all development rows; k={K_BEST}",
            None,
            None,
        ),
    ]
    for value in BUDGET_SWEEP_VALUES:
        artifact = selections[float(value)]
        candidates.append(
            (
                f"{lambda_tag(value)}_selected",
                tuple(int(index) for index in artifact["selected_clean_indices"]),
                "unchanged feasible selection from the budget-penalty sweep",
                float(value),
                float(artifact["best_feasible_dt_inner_cv_accuracy"]),
            )
        )

    methods: list[dict[str, Any]] = []
    for name, features, selection_rule, value, selection_accuracy in candidates:
        started = time.perf_counter()
        probe_result = final_probe.probe(features, test)
        elapsed = time.perf_counter() - started
        train_accuracy = float(
            probe_result.tree.score(
                development.X[:, np.asarray(features, dtype=int)],
                development.y,
            )
        )
        methods.append(
            {
                "name": name,
                "lambda": value,
                "selection_rule": selection_rule,
                "selected_clean_indices": list(features),
                "selected_original_feature_ids": [original_ids[index] for index in features],
                "selected_count": len(features),
                "compression_ratio": float(1.0 - len(features) / context.n_features),
                "selection_best_feasible_dt_inner_cv_accuracy": selection_accuracy,
                "dt_development_accuracy": train_accuracy,
                "dt_test_accuracy": float(probe_result.accuracy),
                "dt_fit_and_test_seconds": elapsed,
            }
        )
        print(
            f"seed={seed} {name:<29} features={len(features):>2} "
            f"dt_train={train_accuracy:.4f} dt_test={probe_result.accuracy:.4f}",
            flush=True,
        )

    result = {
        "protocol": {
            "stage": "stage2_budget_penalty_dt_final_test",
            "all_sweep_selections_completed_before_test_evaluation": True,
            "rl_retrained": False,
            "rl_selected_features_modified": False,
            "test_role": "final_evaluation_only",
            "final_model": "DecisionTreeClassifier through DecisionTreeProbe",
            "selection_sources": {
                lambda_tag(value): str(_selection_path(seed, value)) for value in BUDGET_SWEEP_VALUES
            },
        },
        "seed": seed,
        "config": {
            "inner_cv_folds": INNER_CV_FOLDS,
            "kbest_k": K_BEST,
            "feature_budget": BUDGET_SWEEP_FEATURE_BUDGET,
            "beta": BUDGET_SWEEP_BETA,
            "lambdas": [float(value) for value in BUDGET_SWEEP_VALUES],
        },
        "dataset_metadata": metadata,
        "split_indices": {
            "development_train_plus_validation": development.indices.astype(int).tolist(),
            "source_test": test.indices.astype(int).tolist(),
        },
        "mutual_information_scores": {
            str(original_ids[index]): float(mi_scores[index]) for index in range(context.n_features)
        },
        "methods": methods,
    }
    _write_json(result, BUDGET_SWEEP_DT_TEST_ROOT / f"seed-{seed}" / "results.json")
    return result


def _dt_aggregate(
    seed_results: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    names = [method["name"] for method in seed_results[0]["methods"]]
    kbest_by_seed = {
        result["seed"]: next(
            method["dt_test_accuracy"]
            for method in result["methods"]
            if method["name"] == "kbest_mutual_info"
        )
        for result in seed_results
    }
    summaries: list[dict[str, Any]] = []
    flat_rows: list[dict[str, Any]] = []
    for name in names:
        rows = [
            next(method for method in result["methods"] if method["name"] == name) for result in seed_results
        ]
        deltas = [
            float(row["dt_test_accuracy"] - kbest_by_seed[result["seed"]])
            for result, row in zip(seed_results, rows)
        ]
        selection_values = [
            float(row["selection_best_feasible_dt_inner_cv_accuracy"])
            for row in rows
            if row["selection_best_feasible_dt_inner_cv_accuracy"] is not None
        ]
        summary = {
            "name": name,
            "lambda": rows[0]["lambda"],
            "n_seeds": len(rows),
            "selected_count": _summary([row["selected_count"] for row in rows]),
            "selection_best_feasible_dt_inner_cv_accuracy": (
                _summary(selection_values) if selection_values else None
            ),
            "dt_test_accuracy": _summary([row["dt_test_accuracy"] for row in rows]),
            "delta_vs_kbest_mutual_info": _summary(deltas),
            "win_tie_loss_vs_kbest_mutual_info": {
                "win": sum(delta > 1e-12 for delta in deltas),
                "tie": sum(abs(delta) <= 1e-12 for delta in deltas),
                "loss": sum(delta < -1e-12 for delta in deltas),
            },
        }
        summaries.append(summary)
        for result, row, delta in zip(seed_results, rows, deltas):
            flat_rows.append(
                {
                    "seed": result["seed"],
                    "method": name,
                    "lambda": row["lambda"],
                    "selected_count": row["selected_count"],
                    "selection_best_feasible_dt_inner_cv_accuracy": row[
                        "selection_best_feasible_dt_inner_cv_accuracy"
                    ],
                    "dt_test_accuracy": row["dt_test_accuracy"],
                    "delta_vs_kbest_mutual_info": delta,
                    "dt_fit_and_test_seconds": row["dt_fit_and_test_seconds"],
                    "selected_original_feature_ids": ";".join(
                        str(item) for item in row["selected_original_feature_ids"]
                    ),
                }
            )
    return (
        {
            "dataset": DATASET,
            "seeds": list(BUDGET_SWEEP_SEEDS),
            "beta": BUDGET_SWEEP_BETA,
            "feature_budget": BUDGET_SWEEP_FEATURE_BUDGET,
            "lambdas": [float(value) for value in BUDGET_SWEEP_VALUES],
            "selection_root": str(BUDGET_SWEEP_SELECTION_ROOT),
            "protocol": (
                "all budget-penalty selections finish on inner-CV DT before final DT "
                "fits source train and evaluates source test"
            ),
            "methods": summaries,
        },
        flat_rows,
    )


def run_dt_test() -> None:
    """Validate the full sweep, then evaluate each seed on source test."""
    complete = _require_complete_sweep()
    print(
        f"validated all {len(complete)} budget-sweep selections; outer DT test permitted",
        flush=True,
    )
    seed_results = [_run_seed_dt(seed, complete) for seed in BUDGET_SWEEP_SEEDS]
    aggregate, flat_rows = _dt_aggregate(seed_results)
    aggregate_rows = [
        {
            "method": item["name"],
            "lambda": item["lambda"],
            "n_seeds": item["n_seeds"],
            "selected_count_mean": item["selected_count"]["mean"],
            "selected_count_std": item["selected_count"]["std"],
            "selection_best_feasible_dt_inner_cv_accuracy_mean": (
                item["selection_best_feasible_dt_inner_cv_accuracy"]["mean"]
                if item["selection_best_feasible_dt_inner_cv_accuracy"] is not None
                else ""
            ),
            "dt_test_accuracy_mean": item["dt_test_accuracy"]["mean"],
            "dt_test_accuracy_std": item["dt_test_accuracy"]["std"],
            "delta_vs_kbest_mutual_info_mean": item["delta_vs_kbest_mutual_info"]["mean"],
            **item["win_tie_loss_vs_kbest_mutual_info"],
        }
        for item in aggregate["methods"]
    ]
    _write_json(aggregate, BUDGET_SWEEP_DT_TEST_ROOT / "aggregate.json")
    _write_csv(flat_rows, BUDGET_SWEEP_DT_TEST_ROOT / "per_seed_results.csv")
    _write_csv(aggregate_rows, BUDGET_SWEEP_DT_TEST_ROOT / "aggregate.csv")
    _write_csv(
        flat_rows,
        TABLE_ROOT / f"{BUDGET_SWEEP_TABLE_PREFIX}_dt_test_per_seed.csv",
    )
    _write_csv(
        aggregate_rows,
        TABLE_ROOT / f"{BUDGET_SWEEP_TABLE_PREFIX}_dt_test_aggregate.csv",
    )

    print("\nbudget-sweep DT aggregate against MI-KBest:")
    for item in aggregate["methods"]:
        accuracy = item["dt_test_accuracy"]
        delta = item["delta_vs_kbest_mutual_info"]
        record = item["win_tie_loss_vs_kbest_mutual_info"]
        print(
            f"{item['name']:<29} features={item['selected_count']['mean']:.1f} "
            f"test={accuracy['mean']:.4f}±{accuracy['std']:.4f} "
            f"delta={delta['mean']:+.4f} "
            f"W/T/L={record['win']}/{record['tie']}/{record['loss']}",
            flush=True,
        )


def main() -> None:
    run_selection()
    run_dt_test()


if __name__ == "__main__":
    main()
