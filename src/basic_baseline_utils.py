"""Shared Mutual-Information selection and result-reporting helpers."""

from __future__ import annotations

import csv
import json
import statistics
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_selection import mutual_info_classif

from stage2_rl_config import DATASET, SEEDS


def _kbest_mutual_information(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    k: int,
    seed: int,
) -> tuple[tuple[int, ...], np.ndarray]:
    """仅用调用方提供的训练数据计算互信息排名，并保留预先固定数量的特征。"""
    if not 1 <= k <= X_train.shape[1]:
        raise ValueError(f"k must be in [1, {X_train.shape[1]}], got {k}")

    # mutual_info_classif 的估计过程可能受随机种子影响，因此每个实验种子分别计算。
    scores = mutual_info_classif(X_train, y_train, random_state=seed)
    # 极端情况下若某列得到 NaN，则把它放到排名末尾，而不是让排序结果失真。
    safe_scores = np.nan_to_num(scores, nan=-np.inf)
    ranked = np.argsort(-safe_scores, kind="stable")
    # 返回排序后的列下标，便于保存、比较和稳定地切片矩阵。
    return tuple(sorted(int(index) for index in ranked[:k])), scores


def _jaccard(left: list[int], right: list[int]) -> float:
    """计算两个种子所选特征集合的 Jaccard 相似度，衡量选择稳定性。"""
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    return float(len(left_set & right_set) / len(union)) if union else 1.0


def _summary(values: list[float]) -> dict[str, Any]:
    """汇总多随机种子结果，同时保留每个种子的原始数值。"""
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "values": values,
    }


def _write_json(payload: dict[str, Any], path: Path) -> None:
    """先写临时文件再替换目标文件，避免中断时留下半个 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def _aggregate(seed_results: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """按方法聚合多种子指标，并生成适合写入 CSV 的逐种子扁平记录。"""
    method_names = [method["name"] for method in seed_results[0]["methods"]]
    aggregate_methods: list[dict[str, Any]] = []
    flat_rows: list[dict[str, Any]] = []
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

    for name in method_names:
        per_seed = [
            next(method for method in seed_result["methods"] if method["name"] == name)
            for seed_result in seed_results
        ]
        selections = [method["selected_original_feature_ids"] for method in per_seed]
        # 对全部种子两两计算 Jaccard；越接近 1，所选特征越稳定。
        jaccards = [_jaccard(left, right) for left, right in combinations(selections, 2)]
        metric_summary = {
            metric: _summary([float(method["metrics"][metric]) for method in per_seed])
            for metric in metric_names
        }
        aggregate_methods.append(
            {
                "name": name,
                "n_seeds": len(per_seed),
                "selected_count": _summary([float(method["selected_count"]) for method in per_seed]),
                "metrics": metric_summary,
                "selection_stability_jaccard": _summary(jaccards) if jaccards else None,
            }
        )

        for seed_result, method in zip(seed_results, per_seed):
            metrics = method["metrics"]
            flat_rows.append(
                {
                    "seed": seed_result["seed"],
                    "method": name,
                    "selected_count": method["selected_count"],
                    "compression_ratio": method["compression_ratio"],
                    **{metric: metrics[metric] for metric in metric_names},
                    "selection_elapsed_seconds": method["selection_elapsed_seconds"],
                    "lr_fit_and_score_seconds": method["lr_fit_and_score_seconds"],
                    "total_elapsed_seconds": method["total_elapsed_seconds"],
                    "selected_original_feature_ids": ";".join(
                        str(index) for index in method["selected_original_feature_ids"]
                    ),
                }
            )

    aggregate = {
        "dataset": DATASET,
        "seeds": list(SEEDS),
        "validation_used": False,
        "selection_and_final_fit_rows": seed_results[0]["protocol"]["selection_fit_rows"],
        "source_test_rows": seed_results[0]["protocol"]["source_test_rows"],
        "methods": aggregate_methods,
    }
    return aggregate, flat_rows


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """写出逐种子或聚合后的表格数据。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_aggregate_csv(aggregate: dict[str, Any], path: Path) -> None:
    """从完整聚合结构中提取常用指标，生成方便画图的紧凑表格。"""
    rows = []
    for method in aggregate["methods"]:
        rows.append(
            {
                "method": method["name"],
                "n_seeds": method["n_seeds"],
                "selected_count_mean": method["selected_count"]["mean"],
                "test_accuracy_mean": method["metrics"]["test_accuracy"]["mean"],
                "test_accuracy_std": method["metrics"]["test_accuracy"]["std"],
                "balanced_accuracy_mean": method["metrics"]["balanced_accuracy"]["mean"],
                "f1_mean": method["metrics"]["f1"]["mean"],
                "roc_auc_mean": method["metrics"]["roc_auc"]["mean"],
                "selection_jaccard_mean": method["selection_stability_jaccard"]["mean"],
            }
        )
    _write_csv(rows, path)
