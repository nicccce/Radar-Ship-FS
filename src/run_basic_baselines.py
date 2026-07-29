#!/usr/bin/env python3
"""Run All Features and MI-KBest on the supplied train/test files.

Feature selection and LR use all source train rows. Metrics are evaluated on source test rows.
"""

from __future__ import annotations

import time
from typing import Any

from basic_baseline_utils import (
    _aggregate,
    _kbest_mutual_information,
    _write_aggregate_csv,
    _write_csv,
    _write_json,
)
from harness.lr_final import lr_metrics_to_dict, score_selected_features_with_lr
from harness.orchestrator import build_run_context
from run_stage2_rl_final_lr import _development_and_test
from run_stage2_rl_selection import _config_for_encoder
from stage2_rl_config import (
    BASIC_ROOT,
    DATASET,
    EXPECTED_CLEAN_FEATURES,
    K_BEST,
    LR_C,
    LR_CLASS_WEIGHT,
    LR_MAX_ITER,
    LR_SOLVER,
    SEEDS,
    TABLE_PREFIX,
    TABLE_ROOT,
)


def _run_seed(seed: int) -> dict[str, Any]:
    config = _config_for_encoder("fixed")
    context = build_run_context(config, seed=seed)
    if context.n_features != EXPECTED_CLEAN_FEATURES:
        raise ValueError(
            f"expected {EXPECTED_CLEAN_FEATURES} cleaned radar features, got {context.n_features}"
        )
    metadata = context.split.train.metadata
    if metadata is None:
        raise ValueError("radar dataset metadata is required")
    original_ids = [int(value) for value in metadata["final_feature_ids"]]
    X_development, y_development, X_test, y_test = _development_and_test(context)

    selection_started = time.perf_counter()
    all_features = tuple(range(context.n_features))
    all_selection_seconds = time.perf_counter() - selection_started

    selection_started = time.perf_counter()
    kbest_features, mi_scores = _kbest_mutual_information(
        X_development,
        y_development,
        k=K_BEST,
        seed=seed,
    )
    kbest_selection_seconds = time.perf_counter() - selection_started

    method_inputs = (
        ("all_features", all_features, None, all_selection_seconds),
        ("kbest_mutual_info", kbest_features, mi_scores, kbest_selection_seconds),
    )
    methods: list[dict[str, Any]] = []
    for name, subset, feature_scores, selection_seconds in method_inputs:
        lr_started = time.perf_counter()
        metrics = score_selected_features_with_lr(
            X_development,
            y_development,
            X_test,
            y_test,
            subset,
            C=LR_C,
            solver=LR_SOLVER,
            max_iter=LR_MAX_ITER,
            class_weight=LR_CLASS_WEIGHT,
            random_state=seed,
        )
        lr_seconds = time.perf_counter() - lr_started
        method: dict[str, Any] = {
            "name": name,
            "selected_clean_indices": list(subset),
            "selected_original_feature_ids": [original_ids[index] for index in subset],
            "selected_count": len(subset),
            "compression_ratio": float(1.0 - len(subset) / context.n_features),
            "metrics": lr_metrics_to_dict(metrics),
            "selection_elapsed_seconds": selection_seconds,
            "lr_fit_and_score_seconds": lr_seconds,
            "total_elapsed_seconds": selection_seconds + lr_seconds,
        }
        if feature_scores is not None:
            method["mutual_information_scores"] = {
                str(original_ids[index]): float(feature_scores[index]) for index in range(context.n_features)
            }
        methods.append(method)
        print(
            f"seed={seed} {name:<20} features={len(subset):>2} "
            f"train_acc={metrics.train_accuracy:.4f} test_acc={metrics.test_accuracy:.4f}",
            flush=True,
        )

    test = context.split.test
    result = {
        "protocol": {
            "row_split": "source train for fitting; source test for evaluation",
            "validation_used": False,
            "selection_fit_rows": int(X_development.shape[0]),
            "final_lr_fit_rows": int(X_development.shape[0]),
            "source_test_rows": int(X_test.shape[0]),
            "test_role": "final_evaluation_only",
            "kbest_k_fixed_before_test": K_BEST,
        },
        "seed": seed,
        "config": {
            "dataset": DATASET,
            "k_best": K_BEST,
            "lr_C": LR_C,
            "lr_solver": LR_SOLVER,
            "lr_max_iter": LR_MAX_ITER,
            "lr_class_weight": LR_CLASS_WEIGHT,
        },
        "dataset_metadata": metadata,
        "split_indices": {
            "development_train_plus_validation": sorted(
                context.split.train.indices.astype(int).tolist()
                + context.split.validation.indices.astype(int).tolist()
            ),
            "source_test": test.indices.astype(int).tolist(),
        },
        "methods": methods,
    }
    _write_json(result, BASIC_ROOT / f"seed-{seed}" / "results.json")
    return result


def main() -> None:
    print(
        f"dataset={DATASET} seeds={list(SEEDS)}; no validation baseline: train+validation -> fit, "
        "source test -> final metrics",
        flush=True,
    )
    seed_results = [_run_seed(seed) for seed in SEEDS]
    aggregate, flat_rows = _aggregate(seed_results)
    aggregate.update(
        {
            "dataset": DATASET,
            "row_split": "source train for fitting; source test for evaluation",
            "selection_and_final_fit_rows": seed_results[0]["protocol"]["selection_fit_rows"],
            "source_test_rows": seed_results[0]["protocol"]["source_test_rows"],
        }
    )
    _write_json(aggregate, BASIC_ROOT / "aggregate.json")
    _write_csv(flat_rows, BASIC_ROOT / "per_seed_results.csv")
    _write_aggregate_csv(aggregate, BASIC_ROOT / "aggregate.csv")
    _write_csv(flat_rows, TABLE_ROOT / f"{TABLE_PREFIX}_basic_lr_per_seed.csv")
    _write_aggregate_csv(aggregate, TABLE_ROOT / f"{TABLE_PREFIX}_basic_lr_aggregate.csv")
    for method in aggregate["methods"]:
        accuracy = method["metrics"]["test_accuracy"]
        print(
            f"{method['name']:<20} test_acc={accuracy['mean']:.4f}±{accuracy['std']:.4f}",
            flush=True,
        )
    print(f"artifacts: {BASIC_ROOT / 'aggregate.json'}", flush=True)


if __name__ == "__main__":
    main()
