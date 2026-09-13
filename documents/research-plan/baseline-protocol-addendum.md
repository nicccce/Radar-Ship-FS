# 统一强基线协议附录

版本：`unified-strong-baselines-v1`
预注册日期：2026-09-12
上位协议：`research-baseline-v1`（`documents/research-plan/protocol.md`）
正式配置：`configs/v16n/unified_strong_baselines.toml`
配置原文 SHA-256：`e5feaced461b71e464ae4faa257669886d6fd174400d2c92b526108911c390d5`
规范化配置 SHA-256：`504a2fc4c1e688c0dc115768aecfa757fcfb018dd01e07068e2e33ccf9dd9f25`

## 1. 目的与边界

本附录在不改进或调参 RL 网络的前提下，预注册经典强基线的固定特征数与自动特征数比较。它继承上位协议的数据版本、train-only 清洗、特征坐标、seed、内部折、选择评分器、最终分类器和 reused source-test 声明。

本附录只增加上位协议要求另行预注册的 matched-K 轨道，以及前向贪心后的 best-improvement 单交换局部搜索。正式产物写入新的 `experiments/unified_strong_baselines_v1/`，不覆盖历史实验或 `experiments/research_baseline_v1/`。

## 2. 冻结数据、随机性与评分

- 数据：`v16n_2x_noise`；source-train 3897 行，source-test 1671 行；文件 hash 沿用上位协议。
- 正式 seed：`42, 43, 44, 45, 46`。
- 每个 seed 复现“先按 0.25 切 train/validation，再按 train 后 validation 合并”的行顺序和共享 RNG 消耗。
- 每个 seed 固定一组 `StratifiedKFold(n_splits=5, shuffle=true)` 折，并用同一 `cv_tree_random_state` 实例化所有候选的 `DecisionTreeClassifier`。
- MI 方法同一 seed 共享一个记录在 context 中的 `mi_random_state`。
- 选择分数是五折 accuracy 算术平均；严格改进阈值为 `1e-12`。
- 选择阶段只打开 source-train。所有 seed 和子集完成后，评价阶段才一次性载入 source-test。
- 开发阶段的主判断是跨 seed 的 inner-CV accuracy。source-test 仅作历史已观察测试集上的复用诊断，不能反向更改 K、子集、停止条件或排名定义。

## 3. 固定 K 轨道

预注册 `K = 8, 16, 32`。

- `mi_topk`：对完整 source-train 做一次 MI 排序，取前 K。
- `forward_greedy`：从空集开始，每步对所有剩余特征评分，加入分数最高者；固定 K 轨道即使该步不改善也继续，直到恰好 K。
- `forward_greedy_single_swap`：先执行上述恰好 K 的前向贪心；随后每轮枚举全部 `K × (65-K)` 个单交换邻居，接受分数最高且严格改善的邻居，直到无改善。
- `all_features`：65 个清洗后特征的参照项，不伪装成 matched-K 方法。
- `mi_ordered_accept` 不建立人为固定 K 版本；强制接受固定 MI 前缀会与 `mi_topk` 重复，因而只进入自动轨道。

同一 K 内以跨 seed 平均 inner-CV accuracy 描述性能，不跨 K 混成一个固定 K 排名。完全同分时沿用确定性候选顺序。

## 4. 自动特征数轨道

- `mi_topk__auto_grid`：只在预注册的 `{8,16,32}` 中用 inner-CV accuracy 选 K；同分选更小 K。
- `mi_ordered_accept__automatic`：沿固定 MI 顺序逐一尝试；只有严格改善才接受；达到 32 或遍历完 65 个特征后停止。
- `forward_greedy__automatic`：每步枚举全部剩余特征；最佳加入仍不能严格改善时停止，最大 32。
- `forward_greedy_single_swap__automatic`：用上一条自动前向贪心初始化，在其实际 K 上执行 best-improvement 单交换搜索；交换不改变 K。

自动轨道允许实际 K 不同，必须把 accuracy 与实际 K 并列解释，不能宣称为 matched-K 排名。

## 5. 有界搜索与成本口径

单交换搜索最多 32 个接受/复核轮次。若达到上限仍未确认无改善，结果标记 `max_swap_rounds_reached_without_optimality_confirmation`，不得称为局部最优。

每个结果记录：

- 候选子集请求数；
- 唯一评分子集数；
- scorer cache 命中数；
- 实际分类器拟合次数（每个首次出现的候选五折各拟合一次）；
- MI 拟合次数；
- 选择总耗时及 scorer 拟合耗时；
- 初始化路径、每轮候选数、接受决定和完整 archive。

局部搜索的计数从前向初始化之前开始，因此初始化和最后一轮“无改善”复核均计入。最终 LR 与 DT 各一次拟合另记为 2 次，不混入搜索成本。

## 6. 最终分类器对齐

主轨道固定为：

`StandardScaler + LogisticRegression(C=1.0, solver="liblinear", max_iter=5000, class_weight="balanced", random_state=seed)`。

次轨道固定为 `DecisionTreeClassifier(random_state=seed)`。两者分栏报告，不择优合并。每个已冻结子集报告 accuracy、balanced accuracy、macro-F1、逐类 recall、ROC-AUC 和 confusion matrix。

若未来 DFS 使用不同训练分类器，只能新增明确的 classifier-aligned 轨道；不同分类器的数字不得进入同一排名。

## 7. DFS 与 smoke

当前 DFS provenance 不满足上位协议第 10 节，正式结构化状态固定写为：

`DFS: unavailable / protocol provenance incomplete`。

不得从现有 oracle diagnostic 脚本、人工特征列表或 source-test 扫描中补数。DFS 缺失不阻塞其他基线。

`configs/v16n/unified_strong_baselines_smoke.toml` 仅以两个 seed、2-fold、`K={4,8}` 检查配对、映射、成本计数、序列化和恢复；其任何数值不得进入科研汇总。
