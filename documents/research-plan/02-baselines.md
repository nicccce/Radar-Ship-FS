# 统一强基线结果

日期：2026-09-12
状态：**Ready with caveats / 在已审范围内可用，但必须带比较限制**

## 1. 结论

本次建立了 train-only 的统一选择入口，并完成 5 个正式 seed、固定 `K={8,16,32}` 与自动特征数轨道，共 70 个冻结子集。所有方法共享 source-train 清洗映射、每 seed 的初始 train/validation 重排行为、五折 row indices、Decision Tree random state、候选评分器和最终 LR/DT 评价器。

最强结论必须分两层表达：

1. **开发阶段、按冻结 DT inner-CV 搜索目标**：`forward_greedy_single_swap` 在 K=16 最强，accuracy 为 **0.9324 ± 0.0027**，平均比同 K 的普通 forward greedy 高 **0.00323**，5/5 seed 改善。自动轨道中 `mi_ordered_accept` 最强，accuracy 为 **0.9275 ± 0.0026**，实际 K 为 **10.4 ± 2.7**。
2. **历史已观察 source-test 上的 LR 主分类器复用诊断**：All Features 的 balanced accuracy 为 **0.9139**；所有预注册选择方法都未超过它。选择方法中最高的是 MI Top-32 的 **0.9088 ± 0.0005**。这只是描述性诊断，不能据此把 K=32 或 MI Top-K 重新选为正式规则。

因此，供后续 RL 方法比较的合理双锚点是：

- 搜索目标锚点：`forward_greedy_single_swap, K=16`；
- LR 主评价锚点：`All Features`，同时保留 `MI Top-32` 作为表现最好的已选特征诊断参照。

这两个锚点回答不同问题，不能合并成一个排名。DT 次分类器也必须独立分栏；其最高复用诊断是 MI Top-8 的 balanced accuracy **0.9243 ± 0.0017**。

DFS 状态为 **`unavailable / protocol provenance incomplete`**，没有填入估计成绩，也未阻塞其他基线。

## 2. 统一入口与冻结边界

正式配置为 `configs/v16n/unified_strong_baselines.toml`，规范化配置 SHA-256 为：

`504a2fc4c1e688c0dc115768aecfa757fcfb018dd01e07068e2e33ccf9dd9f25`

预注册细节见 `documents/research-plan/baseline-protocol-addendum.md`。运行命令：

```bash
PYTHONPATH=src conda run -n dl-lab python src/run_unified_baselines.py \
  --config configs/v16n/unified_strong_baselines.toml --stage select

PYTHONPATH=src conda run -n dl-lab python src/run_unified_baselines.py \
  --config configs/v16n/unified_strong_baselines.toml --stage evaluate
```

`select` 只打开 source-train；全部 seed 和子集完成且验证通过后，正式 `evaluate` 进程载入一次 source-test。最终指标生成后，独立审计又批量载入一次同一 test 以复算 20 个关键 LR 结果；两次读取都在子集冻结后，0 次用于选择或调参。完整结果再次运行会命中 resume，选择汇总、最终汇总和环境 manifest 的前后 SHA-256 保持不变。partial seed 或 partial final 目录没有完成标记时，入口拒绝覆盖。

小规模配对 smoke 使用两个 seed、2-fold 和 `K={4,8}`，完成 22/22 结果行的不变量检查；其目录位于 `/tmp/radar_ship_unified_baselines_smoke_v1`，未进入下列科研表。

## 3. 固定 K 结果

数值为 5 个 seed 的均值 ± 样本标准差。inner-CV accuracy 是开发阶段主判断；LR/DT balanced accuracy 来自 reused source-test，只作冻结后诊断。时间和成本均为每 seed 平均值。

| 方法 | K | DT inner-CV acc | LR test BAcc | DT test BAcc | 唯一评分子集 | DT 拟合数 | 选择耗时 (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| All Features（参照） | 65 | 0.9177 ± 0.0051 | **0.9139 ± 0.0000** | 0.9089 ± 0.0027 | 1 | 5 | 1.30 |
| MI Top-K | 8 | 0.9194 ± 0.0039 | 0.8684 ± 0.0000 | **0.9243 ± 0.0017** | 1 | 5 | 0.96 |
| Forward greedy | 8 | 0.9263 ± 0.0020 | 0.7939 ± 0.0788 | 0.9156 ± 0.0062 | 492 | 2,460 | 5.97 |
| Forward greedy + 1-swap | 8 | 0.9289 ± 0.0014 | 0.8057 ± 0.0719 | 0.9177 ± 0.0078 | 1,674 | 8,370 | 28.08 |
| MI Top-K | 16 | 0.9135 ± 0.0017 | 0.8800 ± 0.0000 | 0.9044 ± 0.0040 | 1 | 5 | 1.05 |
| Forward greedy | 16 | 0.9292 ± 0.0031 | 0.8903 ± 0.0140 | 0.9145 ± 0.0027 | 920 | 4,600 | 19.04 |
| **Forward greedy + 1-swap** | **16** | **0.9324 ± 0.0027** | 0.8874 ± 0.0099 | 0.9150 ± 0.0054 | 3,237.8 | 16,189 | 100.28 |
| MI Top-K | 32 | 0.9222 ± 0.0039 | **0.9088 ± 0.0005** | 0.9087 ± 0.0044 | 1 | 5 | 1.39 |
| Forward greedy | 32 | 0.9285 ± 0.0045 | 0.9043 ± 0.0055 | 0.9099 ± 0.0063 | 1,584 | 7,920 | 57.54 |
| Forward greedy + 1-swap | 32 | 0.9317 ± 0.0032 | 0.9039 ± 0.0037 | 0.9128 ± 0.0051 | 3,995.4 | 19,977 | 226.07 |

单交换相对其 forward 初始化的配对 inner-CV 增益为：

| K | 平均增益 ± SD | seed 胜/平/负 |
|---:|---:|---:|
| 8 | +0.002565 ± 0.001091 | 5 / 0 / 0 |
| 16 | +0.003233 ± 0.001567 | 5 / 0 / 0 |
| 32 | +0.003233 ± 0.001610 | 5 / 0 / 0 |

相对 MI Top-K，单交换在 K=8、16、32 的平均配对增益分别为 +0.00950、+0.01889、+0.00950，三个 K 都是 5/5 seed 更高。

## 4. 自动特征数结果

`MI Top-K auto` 只在预注册的 `{8,16,32}` 中由 inner-CV 选 K；其余方法使用各自冻结的严格改善停止规则。

| 方法 | 实际 K | DT inner-CV acc | LR test BAcc | DT test BAcc | 唯一评分子集 | DT 拟合数 | 选择耗时 (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| MI Top-K auto grid | 22.4 ± 13.1 | 0.9240 ± 0.0019 | 0.8925 ± 0.0220 | 0.9143 ± 0.0100 | 3 | 15 | 1.62 |
| **MI ordered accept** | **10.4 ± 2.7** | **0.9275 ± 0.0026** | 0.8836 ± 0.0070 | **0.9207 ± 0.0047** | 65 | 325 | 8.39 |
| Forward greedy auto | 5.0 ± 1.0 | 0.9249 ± 0.0037 | 0.7848 ± 0.0687 | 0.9187 ± 0.0052 | 374.6 | 1,873 | 3.86 |
| Forward greedy auto + 1-swap | 5.0 ± 1.0 | 0.9255 ± 0.0037 | 0.7848 ± 0.0688 | 0.9183 ± 0.0056 | 661.0 | 3,305 | 7.50 |

逐 seed 自动 K：

- MI Top-K grid：8, 8, 32, 32, 32；
- MI ordered accept：12, 8, 13, 12, 7；
- forward greedy / 1-swap：4, 6, 6, 5, 4。

MI ordered accept 相对 MI Top-K auto grid 的配对 inner-CV 增益为 **+0.00344 ± 0.00310**，4/5 seed 更高。自动 forward 的单交换只在 seed 45 找到进一步改善，其余 seed 已在该 K 上通过一轮全邻域复核。

## 5. 性能与特征数

开发阶段的关系不是“特征越多越好”：

- MI Top-K 在 K=16 明显下降，到 K=32 再回升；
- forward greedy 从 K=8 到 K=16 改善，但 K=32 略回落；
- 单交换同样在 K=16 达到最高开发分数，K=32 没有继续改善；
- 自动 forward 通常在 4–6 个特征时遇到首个无严格加法改进，而 MI ordered accept 因允许跳过 MI 序列中的坏候选，保留了 7–13 个特征。

但 LR 主分类器的 reused-test 诊断呈不同方向：三种固定 K 选择方法通常都在 K=32 取得各自最高 LR BAcc，而 65 个 All Features 又高于 K=32。相反，DT 次轨道更偏好更小的 MI Top-8。这说明“最优特征数”依赖分类器和评价阶段，不能把 DT-CV 的子集优势直接解释成 LR 的通用优势。

## 6. 计算成本

每个首次出现的候选子集都执行 5 次冻结 DT 拟合，因此 `classifier_fit_count = 5 × unique_scored_subsets`，70/70 结果均通过该恒等式。缓存命中未记作新拟合。

主要量级：

- MI Top-K 固定 K：1 个评分子集、5 次 DT 拟合，另加 1 次 MI 拟合；
- MI ordered accept：固定 65 个评分子集、325 次 DT 拟合；
- forward greedy：从 K=8 的 492 个子集增长到 K=32 的 1,584 个；
- 单交换：前向初始化、每轮全部邻居及最后一轮无改善复核均计入；K=16 平均 16,189 次 DT 拟合，K=32 平均 19,977 次；
- 所有单交换运行都在 32 轮上限前确认局部最优，cap 命中为 0；
- 每个冻结子集的最终评价另有 1 次 LR 与 1 次 DT 拟合，不混入搜索成本。

forward 的 K=8/16/32 是同一 K=32 路径上的累计 checkpoint，表中成本表示“从头获得该 checkpoint 所需成本”，不应把三个 K 的数值相加。MI ranking 也在同 seed 内共享；表中每行仍按独立获得该结果所需的一次 MI 成本报告。wall-clock 受 8-worker 并行与候选特征数影响，只作资源说明，拟合次数是主要预算口径。

## 7. 验证与产物

正式目录：`experiments/unified_strong_baselines_v1/`。

主要产物：

- `selection/runs.csv`：70 行逐 seed 选择与成本；
- `selection/summary.csv`：选择阶段汇总；
- `selection/paired_deltas.csv`：预注册配对差值；
- `selection/seed-*/context.json`：实际 split/CV/MI random state、开发行顺序和五折行号；
- `selection/seed-*/*/result.json`：双坐标子集、fold scores、完整路径、archive 和成本；
- `final/runs.csv`、`final/summary.csv`：LR 主轨道与 DT 次轨道；
- `final/seed-*/*/metrics.json`：accuracy、balanced accuracy、macro-F1、逐类 recall、ROC-AUC 和 confusion matrix；
- `validation/selection-audit.json`、`validation/final-audit.json`：独立重算结论；
- `dfs_status.json`、`environment.json`、`code-hashes.json`：DFS 缺口、运行环境与 dirty-worktree 文件 hash。

验证结果：

- 70/70 选择子集以保存的 fold rows 和 random state 独立重算，最大绝对误差为 0；
- 20 个关键 LR 结果通过一次批量 test 加载独立重算，最大绝对误差为 0；
- 70/70 双坐标交叉校验通过；
- 固定 K、自动 K 对齐、局部搜索不劣于初始化、fold 覆盖与互斥全部通过；
- 定向 Ruff 通过，定向 pytest 为 `21 passed`；
- 完整结果的 resume 前后，selection summary、final summary 和 environment manifest 的文件 hash 不变。

## 8. 比较限制

1. **source-test 不是新独立验证。** 它在此前研究中已被反复观察，本次只能称为冻结后复用诊断。
2. **inner-CV 分数有自适应搜索偏差。** 同一组折既驱动候选选择又用于报告开发最优，搜索越广越可能对折噪声过拟合；单交换的开发优势不能按同等幅度外推。
3. **选择器与主分类器错配。** 选择评分器是 DT，最终主分类器是 LR。LR 上 All Features 胜出、而 DT 次轨道排序不同，正是这一限制的实证表现；不能混排或泛化成“子集质量”的单一结论。
4. **seed SD 不是数据集抽样不确定性。** 五个 seed 共享同一个 source-test，只改变内部折、MI 随机性、子集或 DT random state。All Features 的 LR SD 为 0 也不表示总体误差为 0。
5. **固定 K 与自动 K 不同轨。** 自动方法的实际 K 不同，不能将其 accuracy 直接解释为 matched-K 胜负。
6. **DFS 缺失。** 当前只有不可追溯或 oracle-oriented 的旧评价线索，不满足代码、环境、checkpoint、seed trajectory 和 test 隔离门槛。
7. **代码工作树在运行时为 dirty。** 基准 commit 为 `6f40c37d0f445ed9cffc81a81655d50ea543cfee`；本次新增/修改文件的 SHA-256 已写入 `code-hashes.json`，但在提交前复现仍应同时校验这些文件 hash。

下一步若要估计真正的泛化性能，应新增此前未访问的数据，或采用 repeated/nested outer CV，并把清洗、K、子集和全部适配限制在每个 outer-train 内。该工作应另建协议，不应回写本次冻结结果。
