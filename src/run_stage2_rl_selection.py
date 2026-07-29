#!/usr/bin/env python3
"""运行阶段 2 的三种 RL 特征选择，并保存筛选出的特征和完整训练轨迹。

此入口没有命令行实验参数，实验设置全部位于 ``stage2_rl_config.py``。脚本不会释放随机
测试分区，也不会导入或调用 ``lr_final``；它只在 development 内使用固定分层 5 折
Decision Tree 平均准确率产生 RL 反馈。每个方法结束后立即单独落盘，便于长实验断点续跑。
"""

from __future__ import annotations

import csv
import dataclasses
import json
import statistics
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from config import IrfsConfig, load_config
from harness.contract import SelectionContext, StepRecord
from methods.reinforced_run import (
    _rng_from_snapshot,
    _rng_snapshot,
    build_reinforced_engine,
)
from stage2_cv import build_stage2_cv_context
from stage2_rl_config import (
    DATA_DIR,
    DATA_VERSION,
    DATASET,
    EXPECTED_CLEAN_FEATURES,
    EXPLORATION_STEP_BUDGET,
    HYBRID_SWITCH_STEP,
    HYBRID_WITHDRAW_STEP,
    INNER_CV_FOLDS,
    RESUME_COMPLETED_SELECTIONS,
    RL_METHOD_SPECS,
    SEEDS,
    SELECTION_ROOT,
    TABLE_PREFIX,
    TABLE_ROOT,
    TEST_FRACTION,
    TRAJECTORY_ROLLING_WINDOW,
    VALIDATION_FRACTION,
)

PROTOCOL_VERSION = 2


def _write_json(payload: dict[str, Any], path: Path) -> None:
    """原子写 JSON，避免训练中断时留下无法读取的半文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """原子写 CSV；rows 为空时只是不生成没有语义的空表。"""
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


def _config_for_encoder(state_encoder: str) -> IrfsConfig:
    """构造显式、可序列化的有效配置；MARLFS 的占位状态映射为 fixed 后仍会被引擎忽略。"""
    effective_encoder = "fixed" if state_encoder == "minimal_relevance_redundancy" else state_encoder
    return load_config(
        {
            "dataset": DATASET,
            "data_dir": DATA_DIR,
            "radar_ship_version": DATA_VERSION,
            "seeds": SEEDS,
            "test_fraction": TEST_FRACTION,
            "validation_fraction": VALIDATION_FRACTION,
            "exploration_step_budget": EXPLORATION_STEP_BUDGET,
            "hybrid_switch_step": HYBRID_SWITCH_STEP,
            "hybrid_withdraw_step": HYBRID_WITHDRAW_STEP,
            "state_encoder": effective_encoder,
        }
    )


def _method_signature(
    *, seed: int, report_name: str, engine_name: str, state_encoder: str, config: IrfsConfig
) -> dict[str, Any]:
    """断点续跑时用于拒绝旧配置产物的精确签名。"""
    # dataclasses.asdict 会保留 tuple，但 JSON 落盘后 tuple 会变成 list。先做一次 JSON 规范化，
    # 确保重启后从文件读出的签名仍能与当前代码内配置逐项相等。
    effective_config = dataclasses.asdict(config)
    # 新增的预算项默认关闭；从旧实验签名中省略默认值，避免无预算实验被误判为配置变化。
    if effective_config["feature_budget"] is None:
        effective_config.pop("feature_budget")
        effective_config.pop("over_budget_penalty_weight")
    normalized_config = json.loads(json.dumps(effective_config))
    return {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": DATASET,
        "seed": int(seed),
        "report_name": report_name,
        "engine_name": engine_name,
        "reported_state_encoder": state_encoder,
        "effective_irfs_config": normalized_config,
        "inner_cv_folds": INNER_CV_FOLDS,
        "selection_tie_break": "higher mean inner-CV accuracy, then fewer features",
    }


def _load_matching_artifact(path: Path, signature: dict[str, Any]) -> dict[str, Any] | None:
    if not RESUME_COMPLETED_SELECTIONS or not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        artifact = json.load(handle)
    if artifact.get("experiment_signature") != signature:
        return None
    trajectory = artifact.get("trajectory", [])
    if len(trajectory) != EXPLORATION_STEP_BUDGET or not artifact.get("selected_clean_indices"):
        return None
    return artifact


def _trajectory_rows(
    per_step: Sequence[StepRecord],
    *,
    selected_subset: Sequence[int],
    original_feature_ids: Sequence[int],
    elapsed_by_step: Sequence[float],
    rolling_window: int,
    fold_accuracies_by_step: Sequence[Sequence[float]] | None = None,
) -> list[dict[str, Any]]:
    """把最小 StepRecord 扩展成后续稳定性作图所需的逐步观测。"""
    rows: list[dict[str, Any]] = []
    best = -np.inf
    previous_subset: tuple[int, ...] | None = None
    previous_elapsed = 0.0
    accuracies: list[float] = []
    original_ids = [int(value) for value in original_feature_ids]

    if fold_accuracies_by_step is not None and len(fold_accuracies_by_step) != len(per_step):
        raise ValueError("fold_accuracies_by_step must contain one entry per RL step")

    for zero_based_step, record in enumerate(per_step):
        accuracy = float(record.accuracy)
        subset = tuple(int(index) for index in record.subset)
        accuracies.append(accuracy)
        fold_accuracies = (
            tuple(float(value) for value in fold_accuracies_by_step[zero_based_step])
            if fold_accuracies_by_step is not None
            else (accuracy,)
        )
        is_new_best = accuracy > best
        if is_new_best:
            best = accuracy
        elapsed = float(elapsed_by_step[zero_based_step])
        recent = accuracies[max(0, len(accuracies) - rolling_window) :]
        changed = None if previous_subset is None else len(set(previous_subset) ^ set(subset))
        adjacent_jaccard = None if previous_subset is None else _jaccard(previous_subset, subset)
        rows.append(
            {
                "step": zero_based_step + 1,
                "dt_inner_cv_accuracy": accuracy,
                "dt_inner_cv_fold_accuracies": list(fold_accuracies),
                "dt_inner_cv_fold_std": statistics.pstdev(fold_accuracies),
                "dt_inner_cv_fold_min": min(fold_accuracies),
                "dt_inner_cv_fold_max": max(fold_accuracies),
                "running_best_inner_cv_accuracy": float(best),
                "is_new_best_accuracy": bool(is_new_best),
                "is_final_selected_subset": subset == tuple(selected_subset),
                "selected_count": len(subset),
                "compression_ratio": float(1.0 - len(subset) / len(original_ids)),
                "changed_feature_count": changed,
                "jaccard_with_previous": adjacent_jaccard,
                "jaccard_with_selected_subset": _jaccard(subset, selected_subset),
                "rolling_accuracy_mean": statistics.fmean(recent),
                "rolling_accuracy_std": statistics.pstdev(recent) if len(recent) > 1 else 0.0,
                "elapsed_seconds": elapsed,
                "step_elapsed_seconds": elapsed - previous_elapsed,
                "selected_clean_indices": list(subset),
                "selected_original_feature_ids": [original_ids[index] for index in subset],
            }
        )
        previous_subset = subset
        previous_elapsed = elapsed
    return rows


def _trajectory_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    accuracies = [float(row["dt_inner_cv_accuracy"]) for row in rows]
    counts = [float(row["selected_count"]) for row in rows]
    changed = [
        float(row["changed_feature_count"]) for row in rows if row["changed_feature_count"] is not None
    ]
    adjacent = [
        float(row["jaccard_with_previous"]) for row in rows if row["jaccard_with_previous"] is not None
    ]
    first_max_accuracy_step = next(
        int(row["step"])
        for row in rows
        if row["is_new_best_accuracy"] and row["dt_inner_cv_accuracy"] == max(accuracies)
    )
    selected_step = next(int(row["step"]) for row in rows if row["is_final_selected_subset"])
    tail = accuracies[-TRAJECTORY_ROLLING_WINDOW:]
    return {
        "best_step": selected_step,
        "first_max_accuracy_step": first_max_accuracy_step,
        "best_dt_inner_cv_accuracy": max(accuracies),
        "accuracy": _summary(accuracies),
        "selected_count": _summary(counts),
        "changed_feature_count_after_first_step": _summary(changed),
        "adjacent_subset_jaccard_after_first_step": _summary(adjacent),
        "tail_window_steps": len(tail),
        "tail_accuracy_mean": statistics.fmean(tail),
        "tail_accuracy_std": statistics.pstdev(tail) if len(tail) > 1 else 0.0,
    }


def _flat_trajectory_row(seed: int, method: str, row: dict[str, Any]) -> dict[str, Any]:
    """移除嵌套列表，生成适合 pandas/R/Excel 直接读取的长表。"""
    return {
        "seed": seed,
        "method": method,
        **{
            key: value
            for key, value in row.items()
            if not key.startswith("selected_") and key != "dt_inner_cv_fold_accuracies"
        },
        "selected_clean_indices": ";".join(str(value) for value in row["selected_clean_indices"]),
        "selected_original_feature_ids": ";".join(
            str(value) for value in row["selected_original_feature_ids"]
        ),
        "dt_inner_cv_fold_accuracies": ";".join(str(value) for value in row["dt_inner_cv_fold_accuracies"]),
    }


def _run_method(
    base_context: SelectionContext,
    rng_snapshot: tuple,
    *,
    seed: int,
    report_name: str,
    engine_name: str,
    state_encoder: str,
    original_feature_ids: Sequence[int],
) -> dict[str, Any]:
    config = _config_for_encoder(state_encoder)
    signature = _method_signature(
        seed=seed,
        report_name=report_name,
        engine_name=engine_name,
        state_encoder=state_encoder,
        config=config,
    )
    method_dir = SELECTION_ROOT / f"seed-{seed}" / report_name
    selection_path = method_dir / "selection.json"
    existing = _load_matching_artifact(selection_path, signature)
    if existing is not None:
        print(f"seed={seed} method={report_name} resume: using {selection_path}", flush=True)
        return existing

    # 复用完全相同的 split 和 DT probe，但恢复到相同的 post-split RNG 状态，避免方法顺序影响结果。
    context = base_context._replace(
        config=config,
        rng=_rng_from_snapshot(rng_snapshot, seed),
    )
    engine = build_reinforced_engine(engine_name, config)
    started = time.perf_counter()
    elapsed_by_step: list[float] = []
    heartbeat_every = max(1, EXPLORATION_STEP_BUDGET // 10)

    def on_step(step: int, budget: int, accuracy: float, best_accuracy: float) -> None:
        elapsed_by_step.append(time.perf_counter() - started)
        if (step + 1) % heartbeat_every == 0 or step + 1 == budget:
            print(
                f"seed={seed} method={report_name:<23} step={step + 1:>3}/{budget} "
                f"dt_cv={accuracy:.4f} best={best_accuracy:.4f} "
                f"elapsed={elapsed_by_step[-1]:.0f}s",
                flush=True,
            )

    print(
        f"seed={seed} method={report_name} start state={state_encoder} budget={EXPLORATION_STEP_BUDGET}",
        flush=True,
    )
    selection = engine.select(context, on_step=on_step)
    selection_seconds = time.perf_counter() - started
    if len(selection.per_step) != EXPLORATION_STEP_BUDGET:
        raise RuntimeError(
            f"{report_name} produced {len(selection.per_step)} steps, expected {EXPLORATION_STEP_BUDGET}"
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
    best_accuracy = max(float(step.accuracy) for step in selection.per_step)
    artifact = {
        "experiment_signature": signature,
        "protocol": {
            "stage": "stage2_rl_feature_selection_only",
            "search_model": "DecisionTreeClassifier",
            "search_feedback": "mean_stratified_inner_cv_accuracy_plus_configured_reward_terms",
            "development_rows": int(context.split.train.X.shape[0]),
            "inner_cv_folds": INNER_CV_FOLDS,
            "separate_reward_validation_rows": 0,
            "official_test_accessed": False,
            "held_out_random_test_accessed": False,
            "row_split": "merge source files; outer stratified 80/20; inner stratified 5-fold CV",
            "lr_final_called": False,
            "selected_subset_rule": "maximum mean DT inner-CV accuracy; ties use fewer features",
        },
        "dataset_metadata": context.split.train.metadata,
        "split_indices": {
            "development_inner_cv": [int(value) for value in context.split.train.indices],
            "inner_cv_folds": context.probe.fold_indices(),
        },
        "selected_clean_indices": [int(index) for index in selection.selected],
        "selected_original_feature_ids": [int(original_feature_ids[index]) for index in selection.selected],
        "selected_count": len(selection.selected),
        "compression_ratio": float(1.0 - len(selection.selected) / len(original_feature_ids)),
        "best_dt_inner_cv_accuracy": best_accuracy,
        "selection_elapsed_seconds": selection_seconds,
        "trajectory_summary": _trajectory_summary(trajectory),
        "trajectory": trajectory,
    }
    _write_json(artifact, selection_path)
    _write_csv(
        [_flat_trajectory_row(seed, report_name, row) for row in trajectory],
        method_dir / "trajectory.csv",
    )
    print(
        f"seed={seed} method={report_name} done features={len(selection.selected)} "
        f"best_dt_cv={best_accuracy:.4f} elapsed={selection_seconds:.0f}s "
        f"artifact={selection_path}",
        flush=True,
    )
    return artifact


def _aggregate(artifacts: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_method: dict[str, list[dict[str, Any]]] = {
        report_name: [] for report_name, _engine_name, _state_encoder in RL_METHOD_SPECS
    }
    per_seed_rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        signature = artifact["experiment_signature"]
        name = signature["report_name"]
        by_method[name].append(artifact)
        summary = artifact["trajectory_summary"]
        per_seed_rows.append(
            {
                "seed": signature["seed"],
                "method": name,
                "state_encoder": signature["reported_state_encoder"],
                "selected_count": artifact["selected_count"],
                "compression_ratio": artifact["compression_ratio"],
                "best_dt_inner_cv_accuracy": artifact["best_dt_inner_cv_accuracy"],
                "best_step": summary["best_step"],
                "tail_accuracy_mean": summary["tail_accuracy_mean"],
                "tail_accuracy_std": summary["tail_accuracy_std"],
                "selection_elapsed_seconds": artifact["selection_elapsed_seconds"],
                "selected_original_feature_ids": ";".join(
                    str(value) for value in artifact["selected_original_feature_ids"]
                ),
            }
        )

    method_summaries: list[dict[str, Any]] = []
    for name, method_artifacts in by_method.items():
        method_artifacts.sort(key=lambda item: item["experiment_signature"]["seed"])
        selections = [item["selected_clean_indices"] for item in method_artifacts]
        pairwise_jaccards = [_jaccard(left, right) for left, right in combinations(selections, 2)]
        method_summaries.append(
            {
                "name": name,
                "n_seeds": len(method_artifacts),
                "selected_count": _summary([item["selected_count"] for item in method_artifacts]),
                "compression_ratio": _summary([item["compression_ratio"] for item in method_artifacts]),
                "best_dt_inner_cv_accuracy": _summary(
                    [item["best_dt_inner_cv_accuracy"] for item in method_artifacts]
                ),
                "best_step": _summary([item["trajectory_summary"]["best_step"] for item in method_artifacts]),
                "tail_accuracy_mean": _summary(
                    [item["trajectory_summary"]["tail_accuracy_mean"] for item in method_artifacts]
                ),
                "selection_elapsed_seconds": _summary(
                    [item["selection_elapsed_seconds"] for item in method_artifacts]
                ),
                "selection_stability_pairwise_jaccard": _summary(pairwise_jaccards),
            }
        )
    aggregate = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": DATASET,
        "seeds": list(SEEDS),
        "methods": method_summaries,
    }
    return aggregate, per_seed_rows


def _aggregate_trajectories(artifacts: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """按 method/step 聚合五种子轨迹，直接供均值带/标准差带绘图。"""
    rows: list[dict[str, Any]] = []
    for method, _engine_name, _state_encoder in RL_METHOD_SPECS:
        selected = [item for item in artifacts if item["experiment_signature"]["report_name"] == method]
        for step in range(1, EXPLORATION_STEP_BUDGET + 1):
            points = [item["trajectory"][step - 1] for item in selected]
            rows.append(
                {
                    "method": method,
                    "step": step,
                    "n_seeds": len(points),
                    "dt_inner_cv_accuracy_mean": statistics.fmean(
                        point["dt_inner_cv_accuracy"] for point in points
                    ),
                    "dt_inner_cv_accuracy_std": statistics.stdev(
                        point["dt_inner_cv_accuracy"] for point in points
                    ),
                    "running_best_inner_cv_accuracy_mean": statistics.fmean(
                        point["running_best_inner_cv_accuracy"] for point in points
                    ),
                    "running_best_inner_cv_accuracy_std": statistics.stdev(
                        point["running_best_inner_cv_accuracy"] for point in points
                    ),
                    "selected_count_mean": statistics.fmean(point["selected_count"] for point in points),
                    "selected_count_std": statistics.stdev(point["selected_count"] for point in points),
                    "jaccard_with_previous_mean": (
                        ""
                        if step == 1
                        else statistics.fmean(point["jaccard_with_previous"] for point in points)
                    ),
                    "jaccard_with_selected_subset_mean": statistics.fmean(
                        point["jaccard_with_selected_subset"] for point in points
                    ),
                    "elapsed_seconds_mean": statistics.fmean(point["elapsed_seconds"] for point in points),
                }
            )
    return rows


def _aggregate_csv_rows(aggregate: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "method": method["name"],
            "n_seeds": method["n_seeds"],
            "selected_count_mean": method["selected_count"]["mean"],
            "selected_count_std": method["selected_count"]["std"],
            "compression_ratio_mean": method["compression_ratio"]["mean"],
            "best_dt_inner_cv_accuracy_mean": method["best_dt_inner_cv_accuracy"]["mean"],
            "best_dt_inner_cv_accuracy_std": method["best_dt_inner_cv_accuracy"]["std"],
            "best_step_mean": method["best_step"]["mean"],
            "tail_accuracy_mean": method["tail_accuracy_mean"]["mean"],
            "selection_elapsed_seconds_mean": method["selection_elapsed_seconds"]["mean"],
            "selection_jaccard_mean": method["selection_stability_pairwise_jaccard"]["mean"],
        }
        for method in aggregate["methods"]
    ]


def main() -> None:
    # 只构造一次 split/probe；三种方法恢复同一 RNG 快照并共享同一个确定性 DT 缓存。
    base_config = _config_for_encoder("fixed")
    first_context = build_stage2_cv_context(base_config, seed=SEEDS[0], n_splits=INNER_CV_FOLDS)
    if first_context.n_features != EXPECTED_CLEAN_FEATURES:
        raise ValueError(
            f"expected {EXPECTED_CLEAN_FEATURES} cleaned radar features, got {first_context.n_features}"
        )
    metadata = first_context.split.train.metadata
    if metadata is None:
        raise ValueError("radar dataset metadata is required for original feature ids")
    original_feature_ids = metadata["final_feature_ids"]
    print(
        f"dataset={DATASET} seeds={list(SEEDS)} cleaned_features={first_context.n_features} "
        f"budget={EXPLORATION_STEP_BUDGET}",
        flush=True,
    )
    print(
        "selection-only protocol: 80% development, stratified 5-fold DT reward; outer test sealed",
        flush=True,
    )

    artifacts: list[dict[str, Any]] = []
    for seed_index, seed in enumerate(SEEDS):
        context = (
            first_context
            if seed_index == 0
            else build_stage2_cv_context(base_config, seed=seed, n_splits=INNER_CV_FOLDS)
        )
        snapshot = _rng_snapshot(context.rng)
        for report_name, engine_name, state_encoder in RL_METHOD_SPECS:
            artifacts.append(
                _run_method(
                    context,
                    snapshot,
                    seed=seed,
                    report_name=report_name,
                    engine_name=engine_name,
                    state_encoder=state_encoder,
                    original_feature_ids=original_feature_ids,
                )
            )

    aggregate, per_seed_rows = _aggregate(artifacts)
    trajectory_rows = [
        _flat_trajectory_row(
            artifact["experiment_signature"]["seed"],
            artifact["experiment_signature"]["report_name"],
            row,
        )
        for artifact in artifacts
        for row in artifact["trajectory"]
    ]
    aggregate_trajectory_rows = _aggregate_trajectories(artifacts)

    _write_json(aggregate, SELECTION_ROOT / "aggregate.json")
    _write_csv(per_seed_rows, SELECTION_ROOT / "per_seed_selection.csv")
    _write_csv(_aggregate_csv_rows(aggregate), SELECTION_ROOT / "aggregate.csv")
    _write_csv(trajectory_rows, SELECTION_ROOT / "trajectory_all_seeds.csv")
    _write_csv(aggregate_trajectory_rows, SELECTION_ROOT / "trajectory_aggregate.csv")
    _write_csv(per_seed_rows, TABLE_ROOT / f"{TABLE_PREFIX}_rl_selection_per_seed.csv")
    _write_csv(
        _aggregate_csv_rows(aggregate),
        TABLE_ROOT / f"{TABLE_PREFIX}_rl_selection_aggregate.csv",
    )
    _write_csv(
        aggregate_trajectory_rows,
        TABLE_ROOT / f"{TABLE_PREFIX}_rl_trajectory_aggregate.csv",
    )
    print(f"selection aggregate: {SELECTION_ROOT / 'aggregate.json'}", flush=True)
    print(
        f"trajectory table: {TABLE_ROOT / f'{TABLE_PREFIX}_rl_trajectory_aggregate.csv'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
