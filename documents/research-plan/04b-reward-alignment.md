# 任务 4B：奖励可靠性与 LR Balanced Accuracy 对齐

日期：2026-09-14。状态：**NO-GO**。本任务只使用 source-train 的 outer-development 流程，
未打开 source-test，未训练 PPO/DQN。

## 1. 结论

按运行前冻结的字典序规则，候选银行阶段选出的唯一诊断评分器是
`fold-local StandardScaler + LR Balanced Accuracy / mean_minus_0_5_sd`。它的银行可靠性门槛
**通过**。在该评分器下，已从空集重新运行 K=16/32 的 exhaustive forward greedy，并执行最多两轮
完整 best-improvement single-swap；这不是对旧 DT 候选的重排序。

最终决策为 **NO-GO**。后续唯一主奖励：**无；本任务不冻结后续主奖励**。任务 4C 的奖励准入为
**不允许**。该判断严格沿用预注册门槛，没有根据
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

| K | Spearman | 增益符号一致 | top-32 交集合计 | 平均富集 | 富集>1 的划分 | 重复排名稳定性 |
|---|---|---|---|---|---|---|
| 16 | 0.550 | 58.5% | 54 | 9.13× | 5 | 0.851 |
| 32 | 0.292 | 57.3% | 12 | 2.67× | 4 | 0.635 |


### 四种评分信号

为横向比较信号，下表固定使用预注册的 `mean−0.5×SD` 聚合。增益统计以各银行候选相对其 forward anchor 的三次配对评分计算；方差单位为百分点平方。

| 信号 | K | 候选增益均值 | 增益方差 | 符号稳定性 | 重复排名稳定性 | outer Spearman | top-32 富集 |
|---|---:|---:|---:|---:|---:|---:|---:|
| DT Accuracy | 16 | -0.4666 pp | 0.4983 pp² | 77.7% | 0.390 | 0.186 | 1.35× |
| DT Accuracy | 32 | -0.3392 pp | 0.2928 pp² | 78.3% | 0.231 | 0.038 | 1.11× |
| DT BAcc | 16 | -0.4625 pp | 0.4841 pp² | 77.7% | 0.393 | 0.187 | 1.35× |
| DT BAcc | 32 | -0.3360 pp | 0.2868 pp² | 78.7% | 0.235 | 0.038 | 1.33× |
| LR Accuracy | 16 | -0.2718 pp | 0.0129 pp² | 86.2% | 0.853 | 0.550 | 8.80× |
| LR Accuracy | 32 | -0.0329 pp | 0.0148 pp² | 83.4% | 0.639 | 0.286 | 2.89× |
| LR BAcc | 16 | -0.2803 pp | 0.0128 pp² | 87.2% | 0.851 | 0.550 | 9.13× |
| LR BAcc | 32 | -0.0334 pp | 0.0148 pp² | 84.1% | 0.635 | 0.292 | 2.67× |

### LR BAcc 的三种聚合

| 聚合 | K16 Spearman / 富集 | K32 Spearman / 富集 | K32 富集>1 划分 | 选择结果 |
|---|---:|---:|---:|---|
| `single_5fold` | 0.548 / 8.63× | 0.238 / 2.67× | 4/5 | 未选中 |
| `repeated_mean` | 0.554 / 8.46× | 0.289 / 2.44× | 3/5 | 未选中 |
| `mean_minus_0_5_sd` | 0.550 / 9.13× | 0.292 / 2.67× | 4/5 | 唯一选中 |

四种信号和三种聚合的完整结果保存在 `analysis/reliability.csv` 与
`analysis/reliability_summary.csv`。候选相对 forward anchor 的三次评分增益均值、样本方差与符号稳定性
保存在 `analysis/candidate_gain_stats.csv`。DT 数字未参与主奖励择优。

## 4. 重新搜索的 outer-development 结果

下表是新评分器自身的 forward endpoint 到最多两轮 single-swap endpoint 的配对 LR BAcc 增益；
正/平/负以 `1e-12` 判定。

| K | 平均增益 | SD | 正/平/负 | 银行门槛 | 搜索增益门槛 | 成本门槛 |
|---|---|---|---|---|---|---|
| 16 | -0.0425 pp | 0.0949 pp | 0/4/1 | True | False | True |
| 32 | +0.4149 pp | 0.2117 pp | 5/0/0 | True | True | True |

每个 K 都必须同时达到：至少 4/5 outer 划分为正、平均至少 +0.2 pp、银行 top-B/重复稳定性门槛和
成本门槛。任何一项失败即 no-go；不能用另一 K 或另一聚合协议补位。

## 5. 成本

| K | 唯一评分子集/seed | 评分拟合/seed | cache hit/seed | wall 秒/seed |
|---|---|---|---|---|
| 16 | 1799.0 | 26985.0 | 61.8 | 20.7 |
| 32 | 3599.0 | 53985.0 | 97.0 | 72.6 |

银行阶段每个候选物理执行 15 个 DT 和 15 个 LR 折拟合，同时复用预测计算 Accuracy/BAcc；这不会把
共享预测伪装成额外拟合。新搜索只执行选定 LR BAcc 评分器；outer endpoint 的 LR/DT 拟合另计，未混入
搜索预算。

## 6. 审计与限制

- Ruff 通过；4B、任务 3 依赖与统一基线定向 pytest 为 `13 passed`。全量 pytest 在收集阶段因仓库既有缺失模块阻塞，共 11 个 collection errors；详见 `verification.json`。
- resume 审计通过：70 个 case 文件 hash/mtime 不变，0 次新模型拟合。
- audit 状态：`passed`；银行评分、银行 outer LR、搜索 endpoint outer LR 的抽样独立复算最大
  绝对误差分别为 `0`、
  `0`、`0`。
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
