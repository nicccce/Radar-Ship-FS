# K=32 锁定稳健性复核与静态基线定位预注册

版本：`k32-reward-robustness-v1`。冻结日期：2026-09-16。本文、正式配置、实现和 outer partition
在读取任何 4C 性能结果前冻结。

## 1. 性质、准入与解释边界

任务 4C-0 已在当前工作树复验：`pytest --collect-only -q` 收集 174 项且 0 collection errors，
`pytest -q` 为 174 passed、0 failed、0 skipped。任务 4B 的正式结论仍为整体 **NO-GO**，
`unique_main_reward=null`；K=16 已关闭。K=32 在 4B 中为 `+0.4149 pp`、5/5 为正，因此本任务是
看过 4B 后新建的 post-selection development robustness check，不是原 4B 通过、预注册确认、
独立泛化验证或新的 held-out test。

`documents/research-plan/03a-data-lineage-audit.md` 不存在，没有经审计的真实 base-scene/group ID。
因此 outer partition 唯一固定为
`StratifiedKFold(n_splits=5, shuffle=true, random_state=20260915)`。五个 validation folds 互不重叠，
每行恰好验证一次；结论只限同一 `v16n_2x_noise` source-train 上的事后开发复核。4C、4D、5A、5B
若复用这些 folds，属于同一累计模型选择链，不得重新称为新验证。source-test 读取次数固定为 0。

## 2. 数据、清洗与坐标

只读取 SHA-256 为 `2191fdc...bb` 的 3897 行 source-train。每个 outer fold 独立在 outer-train
拟合常量列删除和精确重复列删除，再把同一 original 1-based 映射应用到 outer validation。
全部子集保存 clean 0-based 与 original 1-based 坐标并双向交叉校验。outer validation 不进入清洗拟合、
inner folds、候选评分、archive、停止或方法选择。

outer partition、逐 fold 清洗映射和三次 inner folds 在任何评分前物化到正式目录并 hash；任何冻结文件
变化均拒绝运行。三次 inner 5-fold 的随机状态固定为配置中的三个基值加 outer fold 编号；每次 repeat
在相应 outer-train 中分层，训练/held-out 互斥且每行每 repeat 恰好 held out 一次。

## 3. 唯一开发评分器与搜索

K 只取 32。唯一 J 为每个评分训练折内单独拟合的
`StandardScaler + LogisticRegression(C=1.0, solver=liblinear, class_weight=balanced, max_iter=5000)`
的 Balanced Accuracy。每个 repeat 取五折均值，J 为三个 repeat 均值的
`mean - 0.5 × sample SD`。不比较 K、分类器、指标、repeat/fold 数、聚合器、惩罚权重或 LR 超参数。

每个 outer-train 从空集运行 exact-K exhaustive forward：每步检查所有未选特征，按固定候选顺序的
argmax 加入，直到 K=32，不因中途不改善停止。随后最多两轮完整 best-improvement single-swap；每轮
检查全部 `32×33=1056` 个邻居，只有 J 严格提高超过 `1e-12` 才接受，无改善立即停止。两次接受后标为
预算终止，不宣称局部最优。reward、候选择优、archive 和 stop 全部使用同一 J；本任务不训练 RL。

## 4. 固定静态定位

每个新 outer-train 内重新生成并在该 fold 的 outer validation 上用相同 LR endpoint pipeline 评价：

- All Features；
- MI-32（MI 只在 outer-train 计算）；
- LR-forward；
- LR-forward + 最多两轮完整 swap；
- 任务 2 可比的 DT-Accuracy 五折 exhaustive forward K=32；
- 任务 2 可比的 DT-forward + 完整 best-improvement single-swap，最多 32 轮并以严格改善停止。

任务 2 的旧子集和旧 outer 数值均不得搬用。Task-2 DT 轨道使用本 outer-train 的第一个冻结 inner
5-fold；它只作静态定位，不参与 4C GO-development 判定。DFS 固定写
`unavailable / protocol provenance incomplete`，禁止用人工列表或 source-test oracle 补数。

报告每个方法逐 fold 和 pooled out-of-fold LR Balanced Accuracy；主比较另报告 LR-forward 到 swap
endpoint 的逐 fold及 pooled增益。保存每步 argmax、接受/停止原因、archive、全部唯一评分子集、
请求、cache hit、分类器折拟合、endpoint 拟合、MI 拟合和 wall time。

## 5. GO-development 门槛与停止规则

下列数值在读取 4C 结果前冻结，沿用 4B 原先 K=32 分项门槛，不修改 4B 预注册：

1. swap endpoint 相对 LR-forward 至少 4/5 outer folds 为正（`>1e-12`）；
2. 五折平均 LR BAcc 增益至少 `0.002`（`+0.2 pp`）；
3. 主 LR 搜索每折不超过 4000 个唯一评分子集、60000 次分类器折拟合；
4. 数据隔离、fold-local scaler、坐标、逐步 argmax、严格停止、成本恒等式和独立数值复算全部通过。

全部满足才输出 `GO-development`，且只冻结一个“K=32 机制实验用 development scorer”。它不修改
4B 的整体 NO-GO/`unique_main_reward=null`，也不构成外部泛化。任一失败即 `NO-GO`，停止当前静态
PPO、5A 标签扩展和新的 K/reward 扫描。只有 `GO-development` 才准入任务 4D；5A 必须等待 4D 报告完成。

## 6. 交付与复核

正式目录为 `experiments/k32_reward_robustness_v1/`。交付实现、定向测试、冻结 manifest/partition、
逐 fold context/candidates/results、OOF 预测、汇总、最终决定、独立 audit、环境与完整成本；最终报告为
`documents/research-plan/04c-k32-robustness.md`。审计独立重拟合主 forward/swap endpoints 的 J 和所有
静态 endpoint 的 outer LR 数值，并复核 pooled OOF、fold 覆盖、清洗、inner scaler 边界、argmax、
停止、坐标和成本恒等式。

