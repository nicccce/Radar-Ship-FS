#!/usr/bin/env python3
"""Create the task 4B research report and delivery manifest from frozen artifacts."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from run_reward_alignment import ROOT, read_json, sha256, write_json

REPORT = Path("documents/research-plan/04b-reward-alignment.md")


def percent(value: float) -> str:
    return f"{100.0 * value:.4f}"


def markdown_table(frame: pd.DataFrame, columns: list[str], labels: list[str]) -> str:
    rows = ["| " + " | ".join(labels) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in frame[columns].iterrows():
        rows.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(rows)


def main() -> None:
    selection = read_json(ROOT / "analysis" / "selected-scorer.json")
    decision = read_json(ROOT / "analysis" / "final-decision.json")
    audit = read_json(ROOT / "audit.json")
    reliability = pd.read_csv(ROOT / "analysis" / "reliability_summary.csv")
    search = pd.read_csv(ROOT / "analysis" / "aligned_search_summary.csv")
    search_rows = pd.read_csv(ROOT / "analysis" / "aligned_search_results.csv")
    selected = selection["selected"]
    selected_summary = reliability[
        (reliability["signal"] == "lr_bacc") & (reliability["aggregation"] == selected["aggregation"])
    ].copy()
    selected_summary["spearman_mean"] = selected_summary["spearman_mean"].map(lambda value: f"{value:.3f}")
    selected_summary["top_b_enrichment_mean"] = selected_summary["top_b_enrichment_mean"].map(
        lambda value: f"{value:.2f}×"
    )
    selected_summary["repeat_rank_stability_mean"] = selected_summary["repeat_rank_stability_mean"].map(
        lambda value: f"{value:.3f}"
    )
    selected_summary["gain_sign_agreement_mean"] = selected_summary["gain_sign_agreement_mean"].map(
        lambda value: f"{100 * value:.1f}%"
    )

    search_display = search.copy()
    search_display["outer_lr_bacc_gain_mean"] = search_display["outer_lr_bacc_gain_mean"].map(
        lambda value: f"{100 * value:+.4f} pp"
    )
    search_display["outer_lr_bacc_gain_sd"] = search_display["outer_lr_bacc_gain_sd"].map(
        lambda value: f"{100 * value:.4f} pp"
    )
    search_display["signs"] = search_display.apply(
        lambda row: (
            f"{int(row.positive_partitions)}/{int(row.zero_partitions)}/{int(row.negative_partitions)}"
        ),
        axis=1,
    )

    cost = (
        search_rows.groupby("k")
        .agg(
            unique_mean=("unique_scored_subsets", "mean"),
            fits_mean=("classifier_fit_count", "mean"),
            cache_mean=("cache_hits", "mean"),
            wall_mean=("wall_seconds", "mean"),
        )
        .reset_index()
    )
    for column in ("unique_mean", "fits_mean", "cache_mean"):
        cost[column] = cost[column].map(lambda value: f"{value:.1f}")
    cost["wall_mean"] = cost["wall_mean"].map(lambda value: f"{value:.1f}")

    verdict = "GO" if decision["status"] == "go" else "NO-GO"
    reward_text = decision["unique_main_reward"] or "无；本任务不冻结后续主奖励"
    bank_gate = "通过" if selected["bank_gate_passed"] else "未通过"
    reliability_table = markdown_table(
        selected_summary,
        [
            "k",
            "spearman_mean",
            "gain_sign_agreement_mean",
            "top_b_overlap_sum",
            "top_b_enrichment_mean",
            "partitions_enrichment_gt_1",
            "repeat_rank_stability_mean",
        ],
        ["K", "Spearman", "增益符号一致", "top-32 交集合计", "平均富集", "富集>1 的划分", "重复排名稳定性"],
    )
    search_table = markdown_table(
        search_display,
        [
            "k",
            "outer_lr_bacc_gain_mean",
            "outer_lr_bacc_gain_sd",
            "signs",
            "bank_gate_passed",
            "search_gain_gate_passed",
            "cost_gate_passed",
        ],
        ["K", "平均增益", "SD", "正/平/负", "银行门槛", "搜索增益门槛", "成本门槛"],
    )
    cost_table = markdown_table(
        cost,
        ["k", "unique_mean", "fits_mean", "cache_mean", "wall_mean"],
        ["K", "唯一评分子集/seed", "评分拟合/seed", "cache hit/seed", "wall 秒/seed"],
    )
    report = f"""# 任务 4B：奖励可靠性与 LR Balanced Accuracy 对齐

日期：2026-09-14。状态：**{verdict}**。本任务只使用 source-train 的 outer-development 流程，
未打开 source-test，未训练 PPO/DQN。

## 1. 结论

按运行前冻结的字典序规则，候选银行阶段选出的唯一诊断评分器是
`fold-local StandardScaler + LR Balanced Accuracy / {selected["aggregation"]}`。它的银行可靠性门槛
**{bank_gate}**。在该评分器下，已从空集重新运行 K=16/32 的 exhaustive forward greedy，并执行最多两轮
完整 best-improvement single-swap；这不是对旧 DT 候选的重排序。

最终决策为 **{verdict}**。后续唯一主奖励：**{reward_text}**。任务 4C 的奖励准入为
**{"允许" if decision["task_4c_reward_admission"] else "不允许"}**。该判断严格沿用预注册门槛，没有根据
outer-development 结果改选协议、repeat 数、惩罚权重或成功阈值。

## 2. 冻结设计

- 数据：`v16n_2x_noise` source-train 3897 行；5 个 outer 划分继承任务 3 的 seed 42–46 行号。
- 每轮清洗只在 2922 行 outer-train 拟合，再应用到 975 行 outer validation。
- K 固定为 16 和 32。银行包含 forward、MI Top-K、forward 完整单交换邻域、32 个全局随机 exact-K
  子集，以及交换距离 2/4/8 各 16 个扰动。
- 比较 DT Accuracy、DT BAcc、fold-local LR Accuracy、fold-local LR BAcc；每种信号比较单次 5 折、
  3×5 折均值和固定 `mean−0.5×SD`。
- 主 outer 评价固定为 LR Balanced Accuracy；DT 只作独立机制轨道。
- top-B 固定 B=32；成本上限为每个 `seed×K` 4000 个唯一评分子集和 60000 次评分模型拟合。

完整预注册见 `documents/research-plan/reward-alignment-protocol.md`；配置见
`configs/v16n/reward_alignment_v1.toml`。

## 3. 候选银行可靠性

选定诊断评分器的逐 K 摘要如下。富集倍数以均匀随机 top-32 的期望交集为 1×；seed 离散程度仅是
重叠开发划分的描述，不是独立数据集抽样不确定性。

{reliability_table}

四种信号和三种聚合的完整结果保存在 `analysis/reliability.csv` 与
`analysis/reliability_summary.csv`。候选相对 forward anchor 的三次评分增益均值、样本方差与符号稳定性
保存在 `analysis/candidate_gain_stats.csv`。DT 数字未参与主奖励择优。

## 4. 重新搜索的 outer-development 结果

下表是新评分器自身的 forward endpoint 到最多两轮 single-swap endpoint 的配对 LR BAcc 增益；
正/平/负以 `1e-12` 判定。

{search_table}

每个 K 都必须同时达到：至少 4/5 outer 划分为正、平均至少 +0.2 pp、银行 top-B/重复稳定性门槛和
成本门槛。任何一项失败即 no-go；不能用另一 K 或另一聚合协议补位。

## 5. 成本

{cost_table}

银行阶段每个候选物理执行 15 个 DT 和 15 个 LR 折拟合，同时复用预测计算 Accuracy/BAcc；这不会把
共享预测伪装成额外拟合。新搜索只执行选定 LR BAcc 评分器；outer endpoint 的 LR/DT 拟合另计，未混入
搜索预算。

## 6. 审计与限制

- audit 状态：`{audit["status"]}`；银行评分、银行 outer LR、搜索 endpoint outer LR 的抽样独立复算最大
  绝对误差分别为 `{audit["max_abs_bank_score_error"]:.3g}`、
  `{audit["max_abs_bank_outer_error"]:.3g}`、`{audit["max_abs_search_outer_error"]:.3g}`。
- 已核对候选银行完整单交换集合、随机/扰动层数量、outer 行互斥与覆盖、clean↔original 坐标、评分
  拟合恒等式、成本上限、forward/交换逐步 argmax、严格停止以及 reward/archive/stop 同目标。
- StandardScaler 在每个评分训练折内单独拟合；outer 评价时只在完整 outer-train 拟合。
- 五个 outer validation 有历史研究和彼此重叠；本结果属于模型选择流程，不能称为独立泛化估计。
- 候选银行是预注册的局部与随机混合分布；其排序结论不能外推到所有 `C(65,K)` 子集。
- 两轮 single-swap 是冻结的小规模搜索预算；预算终止不等于完整局部最优。

## 7. 产物与复现

正式目录：`experiments/reward_alignment_v1/`。关键产物包括 candidate bank、三重复评分、outer
validation、可靠性表、唯一 scorer 选择、从空集重跑的搜索候选/轨迹、决策和 `audit.json`。

```bash
PYTHONPATH=src /root/miniconda/envs/dl-lab/bin/python src/run_reward_alignment.py --stage bank
PYTHONPATH=src /root/miniconda/envs/dl-lab/bin/python src/run_reward_alignment.py --stage score --jobs 24
PYTHONPATH=src /root/miniconda/envs/dl-lab/bin/python src/run_reward_alignment.py --stage validate --jobs 24
PYTHONPATH=src /root/miniconda/envs/dl-lab/bin/python src/run_reward_alignment.py --stage analyze
PYTHONPATH=src /root/miniconda/envs/dl-lab/bin/python src/run_reward_alignment.py --stage search --jobs 24
PYTHONPATH=src /root/miniconda/envs/dl-lab/bin/python src/run_reward_alignment.py --stage finalize
PYTHONPATH=src /root/miniconda/envs/dl-lab/bin/python src/audit_reward_alignment.py
PYTHONPATH=src /root/miniconda/envs/dl-lab/bin/python src/summarize_reward_alignment.py
```
"""
    REPORT.write_text(report)
    artifact_paths = [
        ROOT / "manifest.json",
        ROOT / "candidate-bank-complete.json",
        ROOT / "bank-score-complete.json",
        ROOT / "bank-validation-complete.json",
        ROOT / "aligned-search-complete.json",
        ROOT / "analysis" / "reliability.csv",
        ROOT / "analysis" / "reliability_summary.csv",
        ROOT / "analysis" / "selected-scorer.json",
        ROOT / "analysis" / "aligned_search_results.csv",
        ROOT / "analysis" / "aligned_search_summary.csv",
        ROOT / "analysis" / "final-decision.json",
        ROOT / "audit.json",
        REPORT,
    ]
    write_json(
        ROOT / "delivery-manifest.json",
        {
            "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "decision": decision,
            "audit_status": audit["status"],
            "artifact_sha256": {str(path): sha256(path) for path in artifact_paths},
        },
    )


if __name__ == "__main__":
    main()
