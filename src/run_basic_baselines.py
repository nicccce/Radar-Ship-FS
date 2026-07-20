#!/usr/bin/env python3
"""运行不使用验证集的雷达数据基础基线实验。

当前项目约定直接在代码中配置实验，因此数据集、随机种子、KBest 的 k 值和逻辑回归参数
都集中写在本文件顶部。All Features 和固定 k 的 Mutual Information KBest 均使用完整的
源训练文件完成特征选择与模型拟合；源测试文件只用于最终的逻辑回归评估。
"""

from __future__ import annotations

import csv
import json
import statistics
import time
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_selection import mutual_info_classif

from config import load_config
from data.loader import LoadedDataset, load
from harness.lr_final import lr_metrics_to_dict, score_frozen_subset_with_lr

# =============================================================================
# 实验设置：需要切换实验时直接修改这里，不通过命令行参数传入。
# =============================================================================
DATASET = "radar_ship"  # loader.py 中注册的数据集名称
DATA_DIR = "../dataset"  # 相对于项目运行目录的原始 SVM-light 数据目录
SEEDS = (42, 43, 44, 45, 46)  # 多随机种子用于观察 MI 特征选择的稳定性
K_BEST = 27  # 查看测试结果前固定：清理后 54 个候选特征的一半

# 所有方法必须使用完全相同的最终 LR 设置，保证比较的差异只来自特征集合。
LR_C = 1.0  # 正则化强度的倒数：C 越小，正则化越强
LR_SOLVER = "liblinear"
LR_MAX_ITER = 5000  # 给求解器足够迭代次数，避免尚未收敛就停止
LR_CLASS_WEIGHT = "balanced"  # 按训练集类别频数自动设置类别权重

# experiments 保存完整可追溯结果，results/tables 保存后续画图方便读取的表格。
EXPERIMENT_ROOT = Path("experiments") / "radar_ship_basic_lr"
TABLE_ROOT = Path("results") / "tables"


def _source_train_test(dataset: LoadedDataset) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """按 loader 记录的官方边界还原源训练集和源测试集，不再划分验证集。"""
    if dataset.predefined_test_indices is None:
        raise ValueError("basic radar baselines require a loader-provided source test partition")

    # loader 将源训练集和源测试集合并存储，同时用 predefined_test_indices 标记测试行。
    # 除官方测试行以外的所有样本都回到训练集，用于特征选择和最终 LR 拟合。
    all_indices = np.arange(dataset.X.shape[0], dtype=int)
    test_indices = np.asarray(dataset.predefined_test_indices, dtype=int)
    train_indices = np.setdiff1d(all_indices, test_indices, assume_unique=True)
    return (
        dataset.X[train_indices],
        dataset.y[train_indices],
        dataset.X[test_indices],
        dataset.y[test_indices],
    )


def _kbest_mutual_information(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    k: int,
    seed: int,
) -> tuple[tuple[int, ...], np.ndarray]:
    """仅用完整源训练集计算互信息排名，并保留预先固定数量的特征。"""
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


def _run_seed(dataset: LoadedDataset, seed: int) -> dict[str, Any]:
    """完成一个随机种子的 All Features、MI KBest 选择及统一 LR 评估。"""
    X_train, y_train, X_test, y_test = _source_train_test(dataset)
    # clean index 是清理后 0 起始列号；original id 是原始 SVM-light 的 1 起始特征编号。
    original_ids = list(dataset.metadata["final_feature_ids"])

    # All Features 不执行选择，直接冻结清理后的全部 54 个特征。
    selection_started = time.perf_counter()
    all_features = tuple(range(dataset.n_features))
    all_selection_seconds = time.perf_counter() - selection_started

    # MI 只读取训练数据；官方测试集不参与特征子集的确定。
    selection_started = time.perf_counter()
    kbest_features, mi_scores = _kbest_mutual_information(
        X_train,
        y_train,
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
        # 两种方法都进入同一个最终评分器：StandardScaler + LogisticRegression。
        lr_started = time.perf_counter()
        metrics = score_frozen_subset_with_lr(
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
        lr_seconds = time.perf_counter() - lr_started
        # 同时保存清理后列号和原始特征编号，便于复现实验及解释选择结果。
        method: dict[str, Any] = {
            "name": name,
            "selected_clean_indices": list(subset),
            "selected_original_feature_ids": [original_ids[index] for index in subset],
            "selected_count": len(subset),
            "compression_ratio": float(1.0 - len(subset) / dataset.n_features),
            "metrics": lr_metrics_to_dict(metrics),
            "selection_elapsed_seconds": selection_seconds,
            "lr_fit_and_score_seconds": lr_seconds,
            "total_elapsed_seconds": selection_seconds + lr_seconds,
        }
        if feature_scores is not None:
            # MI 方法额外保存每个候选特征的互信息分数，便于后续排序和画图。
            method["mutual_information_scores"] = {
                str(original_ids[index]): float(feature_scores[index]) for index in range(dataset.n_features)
            }
        methods.append(method)
        print(
            f"seed={seed} {name:<20} features={len(subset):>2} "
            f"train_acc={metrics.train_accuracy:.4f} test_acc={metrics.test_accuracy:.4f}",
            flush=True,
        )

    # 把实验协议和参数一并写入结果，防止只看指标时丢失评估条件。
    result = {
        "protocol": {
            "validation_used": False,
            "selection_fit_rows": int(X_train.shape[0]),
            "final_lr_fit_rows": int(X_train.shape[0]),
            "held_out_test_rows": int(X_test.shape[0]),
            "test_role": "final_evaluation_only",
            "kbest_k_fixed_before_test": K_BEST,
        },
        "seed": seed,
        "config": {
            "dataset": DATASET,
            "data_dir": DATA_DIR,
            "k_best": K_BEST,
            "lr_C": LR_C,
            "lr_solver": LR_SOLVER,
            "lr_max_iter": LR_MAX_ITER,
            "lr_class_weight": LR_CLASS_WEIGHT,
        },
        "dataset_metadata": dataset.metadata,
        "methods": methods,
    }
    # 每个随机种子单独保存，长实验中途停止时仍可保留已经完成的结果。
    _write_json(result, EXPERIMENT_ROOT / f"seed-{seed}" / "results.json")
    return result


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
        "held_out_test_rows": seed_results[0]["protocol"]["held_out_test_rows"],
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


def main() -> None:
    # 通过统一 loader 读取并清理数据；常数和重复特征的规则只由源训练集拟合。
    config = load_config({"dataset": DATASET, "data_dir": DATA_DIR})
    dataset = load(config)
    if dataset.n_features != 54:
        raise ValueError(f"expected 54 cleaned radar features, got {dataset.n_features}")

    print(
        f"dataset={DATASET} source_train={dataset.metadata['source_train_rows']} "
        f"source_test={dataset.metadata['source_test_rows']} cleaned_features={dataset.n_features}"
    )
    print("validation_used=False; source training file is returned in full to selection + final LR")
    # 不使用验证集：完整源训练文件用于选择和拟合，源测试文件仅作最终评价。
    seed_results = [_run_seed(dataset, seed) for seed in SEEDS]
    aggregate, flat_rows = _aggregate(seed_results)

    # 保存完整 JSON、逐种子 CSV 和精简聚合 CSV，供复查及后续绘图使用。
    aggregate_path = EXPERIMENT_ROOT / "aggregate.json"
    _write_json(aggregate, aggregate_path)
    _write_csv(flat_rows, EXPERIMENT_ROOT / "per_seed_results.csv")
    _write_csv(flat_rows, TABLE_ROOT / "radar_ship_basic_lr_per_seed.csv")
    _write_aggregate_csv(aggregate, EXPERIMENT_ROOT / "aggregate.csv")
    _write_aggregate_csv(aggregate, TABLE_ROOT / "radar_ship_basic_lr_aggregate.csv")

    print("\naggregate:")
    for method in aggregate["methods"]:
        accuracy = method["metrics"]["test_accuracy"]
        stability = method["selection_stability_jaccard"]
        print(
            f"{method['name']:<20} test_acc={accuracy['mean']:.4f}±{accuracy['std']:.4f} "
            f"jaccard={stability['mean']:.4f}"
        )
    print(f"artifacts: {aggregate_path}")
    print(f"tables: {TABLE_ROOT / 'radar_ship_basic_lr_aggregate.csv'}")


if __name__ == "__main__":
    main()
