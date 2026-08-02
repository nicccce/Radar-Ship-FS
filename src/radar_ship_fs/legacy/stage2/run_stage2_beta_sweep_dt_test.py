#!/usr/bin/env python3
"""Run held-out DT validation only after every beta-sweep selection is complete."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from harness.orchestrator import build_run_context
from probe import DecisionTreeProbe
from run_stage2_beta_sweep_selection import (
    _config_for_beta,
    _selection_path,
    _selection_signature,
    beta_tag,
)
from run_stage2_dt_test import _development_partition, _mi_ranking, _top_k
from run_stage2_rl_selection import _summary, _write_csv, _write_json
from stage2_rl_config import (
    BETA_SWEEP_DT_TEST_ROOT,
    BETA_SWEEP_SEEDS,
    BETA_SWEEP_SELECTION_ROOT,
    BETA_SWEEP_TABLE_PREFIX,
    BETA_SWEEP_VALUES,
    DATASET,
    EXPECTED_CLEAN_FEATURES,
    EXPLORATION_STEP_BUDGET,
    INNER_CV_FOLDS,
    K_BEST,
    TABLE_ROOT,
)


def configure_output(
    *,
    selection_root: Path,
    dt_test_root: Path,
    table_prefix: str,
) -> None:
    """Configure output locations for the already-configured selection variant."""
    global BETA_SWEEP_SELECTION_ROOT, BETA_SWEEP_DT_TEST_ROOT, BETA_SWEEP_TABLE_PREFIX
    BETA_SWEEP_SELECTION_ROOT = selection_root
    BETA_SWEEP_DT_TEST_ROOT = dt_test_root
    BETA_SWEEP_TABLE_PREFIX = table_prefix


def _load_complete_selection(seed: int, beta: float) -> dict[str, Any]:
    path = _selection_path(seed, beta)
    if not path.is_file():
        raise FileNotFoundError(f"missing beta-sweep selection: {path}")
    with path.open("r", encoding="utf-8") as handle:
        artifact = json.load(handle)

    if artifact.get("experiment_signature") != _selection_signature(seed, beta):
        raise ValueError(f"selection signature mismatch: {path}")
    protocol = artifact.get("protocol", {})
    if protocol.get("test_used_during_selection") is not False:
        raise ValueError(f"selection used test data: {path}")
    if len(artifact.get("trajectory", [])) != EXPLORATION_STEP_BUDGET:
        raise ValueError(f"incomplete selection trajectory: {path}")
    if not artifact.get("selected_clean_indices"):
        raise ValueError(f"selection contains no features: {path}")
    return artifact


def _require_complete_sweep() -> dict[tuple[int, float], dict[str, Any]]:
    """Validate all 16 selection artifacts before test evaluation."""
    complete: dict[tuple[int, float], dict[str, Any]] = {}
    errors: list[str] = []
    for seed in BETA_SWEEP_SEEDS:
        for beta in BETA_SWEEP_VALUES:
            try:
                complete[(seed, float(beta))] = _load_complete_selection(seed, beta)
            except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
                errors.append(str(exc))
    if errors:
        detail = "\n".join(f"- {error}" for error in errors)
        raise RuntimeError("outer DT test blocked until every beta/seed selection is complete:\n" + detail)
    return complete


def _run_seed(
    seed: int,
    complete: dict[tuple[int, float], dict[str, Any]],
) -> dict[str, Any]:
    config = _config_for_beta(BETA_SWEEP_VALUES[0])
    context = build_run_context(config, seed=seed)
    if context.n_features != EXPECTED_CLEAN_FEATURES:
        raise ValueError(f"expected {EXPECTED_CLEAN_FEATURES} cleaned features, got {context.n_features}")
    metadata = context.split.train.metadata
    if metadata is None:
        raise ValueError("radar dataset metadata is required")
    original_ids = [int(value) for value in metadata["final_feature_ids"]]
    development = _development_partition(context)

    selections: dict[float, dict[str, Any]] = {}
    for beta in BETA_SWEEP_VALUES:
        artifact = complete[(seed, float(beta))]
        saved_indices = artifact.get("split_indices", {}).get("development_inner_cv")
        if saved_indices != development.indices.astype(int).tolist():
            raise ValueError(f"development rows differ for seed={seed}, beta={beta:g}")
        selections[float(beta)] = artifact

    # Evaluate test only after every sweep artifact and this seed's split are verified.
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
    for beta in BETA_SWEEP_VALUES:
        artifact = selections[float(beta)]
        candidates.append(
            (
                f"{beta_tag(beta)}_selected",
                tuple(int(index) for index in artifact["selected_clean_indices"]),
                "features read unchanged from the configured Full-IRFS beta sweep",
                float(beta),
                float(artifact["best_dt_inner_cv_accuracy"]),
            )
        )

    methods: list[dict[str, Any]] = []
    for name, features, selection_rule, beta, selection_accuracy in candidates:
        started = time.perf_counter()
        probe_result = final_probe.probe(features, test)
        elapsed = time.perf_counter() - started
        train_accuracy = float(
            probe_result.tree.score(
                development.X[:, np.asarray(features, dtype=int)],
                development.y,
            )
        )
        row = {
            "name": name,
            "beta": beta,
            "selection_rule": selection_rule,
            "selected_clean_indices": list(features),
            "selected_original_feature_ids": [original_ids[index] for index in features],
            "selected_count": len(features),
            "compression_ratio": float(1.0 - len(features) / context.n_features),
            "selection_best_dt_inner_cv_accuracy": selection_accuracy,
            "dt_development_accuracy": train_accuracy,
            "dt_test_accuracy": float(probe_result.accuracy),
            "dt_fit_and_test_seconds": elapsed,
        }
        methods.append(row)
        print(
            f"seed={seed} {name:<28} features={len(features):>2} "
            f"dt_train={train_accuracy:.4f} dt_test={probe_result.accuracy:.4f}",
            flush=True,
        )

    result = {
        "protocol": {
            "stage": "stage2_beta_sweep_dt_final_test",
            "all_sweep_selections_completed_before_test_evaluation": True,
            "rl_retrained": False,
            "rl_selected_features_modified": False,
            "development_fit_rows": int(development.X.shape[0]),
            "source_test_rows": int(test.X.shape[0]),
            "test_role": "final_evaluation_only",
            "final_model": "DecisionTreeClassifier through DecisionTreeProbe",
            "selection_sources": {
                beta_tag(beta): str(_selection_path(seed, beta)) for beta in BETA_SWEEP_VALUES
            },
        },
        "seed": seed,
        "config": {
            "inner_cv_folds": INNER_CV_FOLDS,
            "kbest_k": K_BEST,
            "betas": [float(beta) for beta in BETA_SWEEP_VALUES],
            "validation_fraction": config.validation_fraction,
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
    _write_json(result, BETA_SWEEP_DT_TEST_ROOT / f"seed-{seed}" / "results.json")
    return result


def _aggregate(
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
            float(row["selection_best_dt_inner_cv_accuracy"])
            for row in rows
            if row["selection_best_dt_inner_cv_accuracy"] is not None
        ]
        summary = {
            "name": name,
            "beta": rows[0]["beta"],
            "n_seeds": len(rows),
            "selected_count": _summary([row["selected_count"] for row in rows]),
            "selection_best_dt_inner_cv_accuracy": (_summary(selection_values) if selection_values else None),
            "dt_development_accuracy": _summary([row["dt_development_accuracy"] for row in rows]),
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
                    "beta": row["beta"],
                    "selected_count": row["selected_count"],
                    "selection_best_dt_inner_cv_accuracy": row["selection_best_dt_inner_cv_accuracy"],
                    "dt_development_accuracy": row["dt_development_accuracy"],
                    "dt_test_accuracy": row["dt_test_accuracy"],
                    "delta_vs_kbest_mutual_info": delta,
                    "dt_fit_and_test_seconds": row["dt_fit_and_test_seconds"],
                    "selected_original_feature_ids": ";".join(
                        str(value) for value in row["selected_original_feature_ids"]
                    ),
                }
            )

    aggregate = {
        "dataset": DATASET,
        "seeds": list(BETA_SWEEP_SEEDS),
        "betas": [float(beta) for beta in BETA_SWEEP_VALUES],
        "selection_root": str(BETA_SWEEP_SELECTION_ROOT),
        "protocol": (
            "all beta selections finish on inner-CV DT before final DT fits "
            "source train and evaluates source test"
        ),
        "methods": summaries,
    }
    return aggregate, flat_rows


def _aggregate_csv_rows(aggregate: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "method": item["name"],
            "beta": item["beta"],
            "n_seeds": item["n_seeds"],
            "selected_count_mean": item["selected_count"]["mean"],
            "selected_count_std": item["selected_count"]["std"],
            "selection_best_dt_inner_cv_accuracy_mean": (
                item["selection_best_dt_inner_cv_accuracy"]["mean"]
                if item["selection_best_dt_inner_cv_accuracy"] is not None
                else ""
            ),
            "dt_test_accuracy_mean": item["dt_test_accuracy"]["mean"],
            "dt_test_accuracy_std": item["dt_test_accuracy"]["std"],
            "delta_vs_kbest_mutual_info_mean": item["delta_vs_kbest_mutual_info"]["mean"],
            **item["win_tie_loss_vs_kbest_mutual_info"],
        }
        for item in aggregate["methods"]
    ]


def main() -> None:
    complete = _require_complete_sweep()
    print(
        f"validated all {len(complete)} selection artifacts; outer DT test is now permitted",
        flush=True,
    )
    seed_results = [_run_seed(seed, complete) for seed in BETA_SWEEP_SEEDS]
    aggregate, flat_rows = _aggregate(seed_results)
    aggregate_rows = _aggregate_csv_rows(aggregate)

    _write_json(aggregate, BETA_SWEEP_DT_TEST_ROOT / "aggregate.json")
    _write_csv(flat_rows, BETA_SWEEP_DT_TEST_ROOT / "per_seed_results.csv")
    _write_csv(aggregate_rows, BETA_SWEEP_DT_TEST_ROOT / "aggregate.csv")
    _write_csv(
        flat_rows,
        TABLE_ROOT / f"{BETA_SWEEP_TABLE_PREFIX}_dt_test_per_seed.csv",
    )
    _write_csv(
        aggregate_rows,
        TABLE_ROOT / f"{BETA_SWEEP_TABLE_PREFIX}_dt_test_aggregate.csv",
    )

    print("\nDT aggregate against MI-KBest:")
    for item in aggregate["methods"]:
        accuracy = item["dt_test_accuracy"]
        delta = item["delta_vs_kbest_mutual_info"]
        record = item["win_tie_loss_vs_kbest_mutual_info"]
        print(
            f"{item['name']:<28} features={item['selected_count']['mean']:.1f} "
            f"test={accuracy['mean']:.4f}±{accuracy['std']:.4f} "
            f"delta={delta['mean']:+.4f} "
            f"W/T/L={record['win']}/{record['tie']}/{record['loss']}",
            flush=True,
        )
    print(
        f"artifacts={BETA_SWEEP_DT_TEST_ROOT / 'aggregate.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
