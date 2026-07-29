#!/usr/bin/env python3
"""读取阶段 2 RL 筛选出的特征，并在独立进程中运行统一最终 LR。

本入口不构建 RL engine、不产生动作也不改写选择结果。它只读取
选择阶段落盘的 selection.json，把每个 seed 的训练和验证分区合并为最终拟合集，再只在该 seed
重新随机划出的测试分区上评价。实验参数固定在 stage2_rl_config.py。
"""

from __future__ import annotations

import csv
import json
import statistics
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from harness.contract import SelectionContext
from harness.lr_final import lr_metrics_to_dict, score_selected_features_with_lr
from harness.orchestrator import build_run_context
from run_stage2_rl_selection import _config_for_encoder
from stage2_rl_config import (
    ACTIVE_RL_METHOD_SPECS,
    DATASET,
    EXPECTED_CLEAN_FEATURES,
    FINAL_LR_ROOT,
    LR_C,
    LR_CLASS_WEIGHT,
    LR_MAX_ITER,
    LR_SOLVER,
    SEEDS,
    SELECTION_ROOT,
    TABLE_PREFIX,
    TABLE_ROOT,
)


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _summary(values: Sequence[float]) -> dict[str, Any]:
    numeric = [float(value) for value in values]
    return {
        "mean": statistics.fmean(numeric),
        "std": statistics.stdev(numeric) if len(numeric) > 1 else 0.0,
        "min": min(numeric),
        "max": max(numeric),
        "values": numeric,
    }


def _jaccard(left: Sequence[int], right: Sequence[int]) -> float:
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    return float(len(left_set & right_set) / len(union)) if union else 1.0


def _development_and_test(
    context: SelectionContext,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return all development rows and the source test rows."""
    test = context.split.test
    return (
        np.vstack((context.split.train.X, context.split.validation.X)),
        np.concatenate((context.split.train.y, context.split.validation.y)),
        test.X,
        test.y,
    )


def _load_selection(seed: int, method: str) -> tuple[dict[str, Any], Path]:
    path = SELECTION_ROOT / f"seed-{seed}" / method / "selection.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing selection {path}; finish run_stage2_rl_selection.py first")
    with path.open("r", encoding="utf-8") as handle:
        artifact = json.load(handle)
    signature = artifact.get("experiment_signature", {})
    if signature.get("dataset") != DATASET or signature.get("seed") != seed:
        raise ValueError(f"selection identity mismatch in {path}")
    if signature.get("report_name") != method:
        raise ValueError(f"selection method mismatch in {path}")
    protocol = artifact.get("protocol", {})
    if (
        protocol.get("test_used_during_selection") is not False
        or protocol.get("lr_final_called") is not False
    ):
        raise ValueError(f"selection artifact is not selection-only: {path}")
    subset = artifact.get("selected_clean_indices", [])
    if not subset:
        raise ValueError(f"selection artifact has no selected features: {path}")
    return artifact, path


def _run_one(
    context: SelectionContext,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    seed: int,
    method: str,
) -> dict[str, Any]:
    selection, selection_path = _load_selection(seed, method)
    saved_split = selection.get("split_indices", {})
    development_indices = np.concatenate(
        (context.split.train.indices, context.split.validation.indices)
    ).astype(int)
    if saved_split.get("development_inner_cv") != development_indices.tolist():
        raise ValueError(f"development rows differ from saved selection: {selection_path}")
    subset = tuple(int(index) for index in selection["selected_clean_indices"])
    started = time.perf_counter()
    metrics = score_selected_features_with_lr(
        X_train,
        y_train,
        X_test,
        y_test,
        subset,
        C=LR_C,
        solver=LR_SOLVER,
        max_iter=LR_MAX_ITER,
        class_weight=LR_CLASS_WEIGHT,
        random_state=seed,
    )
    elapsed = time.perf_counter() - started
    result = {
        "protocol": {
            "stage": "stage2_selected_features_final_lr_only",
            "selection_source_artifact": str(selection_path),
            "selection_was_modified": False,
            "final_model": "StandardScaler + LogisticRegression",
            "final_lr_fit_rows": int(X_train.shape[0]),
            "source_test_rows": int(X_test.shape[0]),
            "test_role": "final_evaluation_only",
            "row_split_protocol": "source train for development; source test for evaluation",
            "search_reward_model": "DecisionTreeClassifier",
        },
        "seed": seed,
        "method": method,
        "lr_config": {
            "C": LR_C,
            "solver": LR_SOLVER,
            "max_iter": LR_MAX_ITER,
            "class_weight": LR_CLASS_WEIGHT,
            "random_state": seed,
        },
        "dataset_metadata": context.split.train.metadata,
        "selected_clean_indices": list(subset),
        "selected_original_feature_ids": selection["selected_original_feature_ids"],
        "selected_count": len(subset),
        "compression_ratio": float(1.0 - len(subset) / context.n_features),
        "selection_best_dt_inner_cv_accuracy": selection["best_dt_inner_cv_accuracy"],
        "selection_elapsed_seconds": selection["selection_elapsed_seconds"],
        "metrics": lr_metrics_to_dict(metrics),
        "lr_fit_and_score_seconds": elapsed,
    }
    output_path = FINAL_LR_ROOT / f"seed-{seed}" / method / "lr_final.json"
    _write_json(result, output_path)
    print(
        f"seed={seed} method={method:<23} features={len(subset):>2} "
        f"lr_test_acc={metrics.test_accuracy:.4f} balanced_acc={metrics.balanced_accuracy:.4f} "
        f"artifact={output_path}",
        flush=True,
    )
    return result


def _aggregate(results: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metric_names = (
        "train_accuracy",
        "test_accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "f1_macro",
        "roc_auc",
    )
    per_seed_rows: list[dict[str, Any]] = []
    aggregate_methods: list[dict[str, Any]] = []
    for method, _engine_name, _state_encoder in ACTIVE_RL_METHOD_SPECS:
        method_results = sorted(
            (result for result in results if result["method"] == method),
            key=lambda item: item["seed"],
        )
        selections = [result["selected_clean_indices"] for result in method_results]
        jaccards = [_jaccard(left, right) for left, right in combinations(selections, 2)]
        aggregate_methods.append(
            {
                "name": method,
                "n_seeds": len(method_results),
                "selected_count": _summary([result["selected_count"] for result in method_results]),
                "compression_ratio": _summary([result["compression_ratio"] for result in method_results]),
                "selection_best_dt_inner_cv_accuracy": _summary(
                    [result["selection_best_dt_inner_cv_accuracy"] for result in method_results]
                ),
                "metrics": {
                    metric: _summary([result["metrics"][metric] for result in method_results])
                    for metric in metric_names
                },
                "selection_stability_pairwise_jaccard": _summary(jaccards),
                "selection_elapsed_seconds": _summary(
                    [result["selection_elapsed_seconds"] for result in method_results]
                ),
                "lr_fit_and_score_seconds": _summary(
                    [result["lr_fit_and_score_seconds"] for result in method_results]
                ),
            }
        )
        for result in method_results:
            per_seed_rows.append(
                {
                    "seed": result["seed"],
                    "method": method,
                    "selected_count": result["selected_count"],
                    "compression_ratio": result["compression_ratio"],
                    "selection_best_dt_inner_cv_accuracy": result["selection_best_dt_inner_cv_accuracy"],
                    **{metric: result["metrics"][metric] for metric in metric_names},
                    "selection_elapsed_seconds": result["selection_elapsed_seconds"],
                    "lr_fit_and_score_seconds": result["lr_fit_and_score_seconds"],
                    "selected_original_feature_ids": ";".join(
                        str(value) for value in result["selected_original_feature_ids"]
                    ),
                }
            )
    return (
        {
            "dataset": DATASET,
            "seeds": list(SEEDS),
            "protocol": "features selected by DT-guided RL, evaluated by independent final LR",
            "methods": aggregate_methods,
        },
        per_seed_rows,
    )


def _aggregate_csv_rows(aggregate: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "method": method["name"],
            "n_seeds": method["n_seeds"],
            "selected_count_mean": method["selected_count"]["mean"],
            "selected_count_std": method["selected_count"]["std"],
            "compression_ratio_mean": method["compression_ratio"]["mean"],
            "dt_inner_cv_accuracy_mean": method["selection_best_dt_inner_cv_accuracy"]["mean"],
            "test_accuracy_mean": method["metrics"]["test_accuracy"]["mean"],
            "test_accuracy_std": method["metrics"]["test_accuracy"]["std"],
            "balanced_accuracy_mean": method["metrics"]["balanced_accuracy"]["mean"],
            "balanced_accuracy_std": method["metrics"]["balanced_accuracy"]["std"],
            "f1_mean": method["metrics"]["f1"]["mean"],
            "f1_macro_mean": method["metrics"]["f1_macro"]["mean"],
            "roc_auc_mean": method["metrics"]["roc_auc"]["mean"],
            "selection_jaccard_mean": method["selection_stability_pairwise_jaccard"]["mean"],
            "selection_elapsed_seconds_mean": method["selection_elapsed_seconds"]["mean"],
        }
        for method in aggregate["methods"]
    ]


def main() -> None:
    config = _config_for_encoder("fixed")
    print(
        "optional final-LR stage: fit all development rows after inner-CV selection; "
        f"source test evaluation; methods={len(ACTIVE_RL_METHOD_SPECS)} seeds={list(SEEDS)}",
        flush=True,
    )
    results: list[dict[str, Any]] = []
    for seed in SEEDS:
        context = build_run_context(config, seed=seed)
        if context.n_features != EXPECTED_CLEAN_FEATURES:
            raise ValueError(
                f"expected {EXPECTED_CLEAN_FEATURES} cleaned radar features, got {context.n_features}"
            )
        X_train, y_train, X_test, y_test = _development_and_test(context)
        print(
            f"seed={seed} final_fit_rows={X_train.shape[0]} source_test_rows={X_test.shape[0]}",
            flush=True,
        )
        for method, _engine_name, _state_encoder in ACTIVE_RL_METHOD_SPECS:
            results.append(
                _run_one(
                    context,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    seed=seed,
                    method=method,
                )
            )
    aggregate, per_seed_rows = _aggregate(results)
    aggregate_rows = _aggregate_csv_rows(aggregate)
    _write_json(aggregate, FINAL_LR_ROOT / "aggregate.json")
    _write_csv(per_seed_rows, FINAL_LR_ROOT / "per_seed_results.csv")
    _write_csv(aggregate_rows, FINAL_LR_ROOT / "aggregate.csv")
    _write_csv(per_seed_rows, TABLE_ROOT / f"{TABLE_PREFIX}_rl_final_lr_per_seed.csv")
    _write_csv(aggregate_rows, TABLE_ROOT / f"{TABLE_PREFIX}_rl_final_lr_aggregate.csv")

    print("\naggregate:")
    for method in aggregate["methods"]:
        accuracy = method["metrics"]["test_accuracy"]
        print(
            f"{method['name']:<23} features={method['selected_count']['mean']:.1f} "
            f"test_acc={accuracy['mean']:.4f}±{accuracy['std']:.4f} "
            f"jaccard={method['selection_stability_pairwise_jaccard']['mean']:.4f}",
            flush=True,
        )
    print(f"final LR aggregate: {FINAL_LR_ROOT / 'aggregate.json'}", flush=True)


if __name__ == "__main__":
    main()
