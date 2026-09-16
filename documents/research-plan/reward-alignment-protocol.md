# LR Balanced Accuracy 奖励可靠性与搜索对齐预注册

版本：`reward-alignment-v1`。冻结日期：2026-09-14。任何候选银行评分或 outer-development 评价开始前冻结。

## 1. 范围、数据隔离与解释边界

本实验只使用 `v16n_2x_noise` source-train（3897 行，SHA-256
`2191fdc297aa49ce6a700df956ec44f68dc13e4c737937b31167543a203d22bb`），不得打开
source-test，也不训练或更新 PPO/DQN。正式 seed 为 42–46，主 K 固定为 16 和 32。

outer-development 划分逐行继承 `search-diagnosis-v1` 的五个 `nested_dev` context：每个 seed
以 `random_state=10000+seed` 分层留出 975 行 outer-development validation，其余 2922 行为
outer-train。清洗只在各 outer-train 拟合，再按保存的 original 1-based 列映射应用到该轮
outer validation。程序必须核对继承 context 的 hash、行互斥、覆盖和双坐标；不得重新抽取更有利的
outer 划分。五个 validation 集有重叠，是开发模型选择证据，不是五份独立数据，也不是最终 test。

## 2. 冻结候选银行

每个 `seed×K` 的共同候选银行在任何候选评分和 outer-validation 评价前物化并 hash 冻结。银行只用
对应 outer-train 信息生成，包含：

1. 任务 3 在同一 outer-train 上从空集运行 DT Accuracy forward greedy 得到的 K checkpoint；
2. 同一 outer-train 的 MI Top-K；
3. forward checkpoint 的完整 `K×(65-K)` 单交换邻域（K=16 为 784 个，K=32 为 1056 个）；
4. 32 个均匀随机 exact-K 子集；
5. 相对 forward checkpoint 恰好 2、4、8 次交换的扰动各 16 个。

随机银行使用 `numpy.default_rng(400000 + 100*seed + K)`，按上述顺序无放回生成；若重复则继续抽取，
直到每层达到冻结数量。候选按 canonical clean 0-based tuple 去重，一个候选可保留多个 strata 标签。
保存 clean 0-based 与 original 1-based 坐标并双向校验。forward 和 MI 仅作为银行分层来源；银行阶段
不以 outer validation 筛选候选。

## 3. 四种评分信号与重复规则

每个 outer-train 使用三次预注册 5 折 `StratifiedKFold(shuffle=true)`：repeat 0 精确继承任务 3
保存的五折；repeat 1、2 的 random state 分别为 `420000+seed`、`430000+seed`。四种信号为：

- `dt_accuracy`：`DecisionTreeClassifier(random_state=cv_tree_random_state)`，Accuracy；
- `dt_bacc`：同一 DT 预测，Balanced Accuracy；
- `lr_accuracy`：折内 `StandardScaler` + `LogisticRegression(C=1.0, solver=liblinear,
  max_iter=5000, class_weight=balanced, random_state=seed)`，Accuracy；
- `lr_bacc`：同一 fold-local LR pipeline 预测，Balanced Accuracy。

StandardScaler 必须在每个评分训练折内拟合，不能在 outer-train 全体或 validation 上预拟合。每个信号
保存 15 个 fold 分数和 3 个 repeat 均值。候选相对 forward anchor 的三次配对增益报告均值、样本方差、
正/平/负次数和多数符号稳定度。每个信号另报告三对 repeat 排名 Spearman 的平均值。

预注册三种标量协议：

- `single_5fold`：repeat 0 的五折均值；
- `repeated_mean`：三个 repeat 均值的算术平均；
- `mean_minus_0_5_sd`：上述均值减 `0.5 ×` 三个 repeat 均值的样本 SD。

不扫描 repeat 数、折数、惩罚权重或 LR 超参数。DT 轨道单独报告，不能与 LR 轨道择优混排。

## 4. 可靠性评价、唯一候选评分器与 top-B

outer validation 上固定拟合一次完整 outer-train 的主 LR pipeline，并以 Balanced Accuracy 为唯一主
评价；同一候选另报 LR Accuracy、DT Accuracy 和 DT BAcc。主评价模型的 scaler 只在完整 outer-train
拟合。`B=32`。每个 `seed×K×评分协议` 计算训练评分与 outer LR BAcc 的 Spearman、相对 forward
anchor 的增益符号一致率，以及训练 top-32 与 outer LR-BAcc top-32 的交集。富集倍数定义为
`(|交集|/B)/(B/N) = |交集|×N/B²`。

只有三个 `lr_bacc` 标量协议有资格成为搜索评分器。按以下运行前冻结的字典序最大规则选出恰好一个，
不得按后续搜索结果改选：

1. 两个 K 中较小的平均 top-B 富集倍数；
2. 两个 K 合并的平均 top-B 富集倍数；
3. 两个 K 中较小的平均 Spearman；
4. 两个 K 合并的平均 Spearman；
5. 重复排名稳定性；
6. 更低实际评分拟合倍数；
7. 固定同分顺序 `repeated_mean`、`mean_minus_0_5_sd`、`single_5fold`。

银行可靠性预门槛要求两个 K 各自同时满足：平均 top-B 富集倍数至少 2.0、至少 4/5 outer 划分的
富集倍数大于 1.0、三次重复的平均排名稳定性至少 0.5。未通过仍按上述规则选出一个诊断评分器完成
搜索回放，但不能据此冻结后续主奖励。

## 5. 必须重新运行的 forward + single-swap

选出的唯一评分器必须在每个 outer-train、每个 K 从空集重新运行 exact-K exhaustive forward greedy，
不得只重排银行或旧 DT 候选。随后从该 forward checkpoint 运行最多两轮完整 best-improvement
single-swap：每轮评分全部 `K×(65-K)` 邻居，仅当同一选定目标严格提高超过 `1e-12` 时接受；无改善
立即停止。两次接受后停止并标记为预算终止，不宣称局部最优。

reward、archive、候选择优和停止规则全部使用同一选定标量目标。不得加入相关性惩罚；如未来需要，
只能作为单独消融。完全同分按 canonical 候选顺序保留先到项。保存 forward endpoint、每轮最佳项、
接受 archive、最终 endpoint、双坐标、所有 repeat/fold 分数和成本。

## 6. 成本上限和成功门槛

评分协议的实际拟合倍数不得超过单次 5 折的 3 倍。每个 `seed×K` 的新搜索最多 4000 个唯一评分
子集、60000 次分类器拟合；请求、cache hit、实际拟合、wall-clock 和初始化成本全部报告。outer endpoint
评价不混入搜索评分预算。

冻结后续主奖励的实用成功门槛要求全部成立：

1. 银行可靠性预门槛在 K=16 和 K=32 都通过；
2. 对每个 K，新 single-swap endpoint 相对同评分器新 forward endpoint 的 outer LR BAcc 至少 4/5
   划分为正（`>1e-12`），且五划分平均增益至少 `0.002`（0.2 个百分点）；
3. 两个 K 的每个搜索 case 都在成本上限内，且结构、坐标、fold-local scaler 和独立复算审计通过。

仅当三项全部通过，才把第 4 节选中的唯一 `lr_bacc` 标量协议冻结为任务 4C 的主奖励。任一项失败即
输出 no-go；不得同时保留多个奖励供后续择优，也不得因查看结果而放宽门槛。

## 7. 交付与恢复

正式结果写入新目录 `experiments/reward_alignment_v1/`，不覆盖历史实验。交付包括冻结配置和 hash、
outer context、候选银行、候选重复评分、outer validation、可靠性和 top-B 表、新搜索轨迹与 endpoints、
成本、环境、audit、定向测试、Ruff 结果及 `documents/research-plan/04b-reward-alignment.md`。
完成 case 可 resume；partial case 拒绝覆盖；每个下游阶段先校验上游冻结文件 hash。
