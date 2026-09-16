# 任务 4C：K=32 锁定稳健性复核与静态基线定位

日期：2026-09-16  
版本：`k32-reward-robustness-v1`  
状态：**NO-GO — 不进入任务 4D**

## 1. 决策摘要

本次锁定复核未通过 GO-development 门槛。LR-forward 到最多两轮完整 best-improvement swap 的
outer LR Balanced Accuracy：

- 逐折正/平/负为 **3/2/0**，低于要求的至少 4/5 为正；
- 五折平均增益为 **+0.1009 pp**，低于要求的 `+0.2 pp`；
- pooled out-of-fold 增益为 **+0.1010 pp**；
- 成本、隔离、坐标、argmax、严格停止和独立数值复算全部通过。

因此最终决定为 **NO-GO**。不得进入任务 4D；停止当前静态 PPO、5A 标签扩展和新的 K/reward
扫描。5A 不可启动，也不存在“等待一个尚未准入的 4D 训练结果后继续”的例外。

任务 4B 的正式状态保持整体 **NO-GO**，`unique_main_reward=null`；本任务没有冻结 K=32 机制实验用
development scorer，也没有把 4B 改写为通过。该实验是看过 4B K=32 `+0.4149 pp、5/5 正` 后提出的
post-selection development robustness check，不是原 4B 预注册确认、独立泛化验证或新 held-out test。

## 2. 准入、冻结与数据边界

正式实验前在当前工作树复验：

- `pytest --collect-only -q`：174 collected，0 collection errors；
- `pytest -q`：174 executed，174 passed，0 failed，0 skipped；
- 4C-0 工程准入为 PASS。

`documents/research-plan/03a-data-lineage-audit.md` 不存在，未恢复真实 base-scene/group ID。因此没有在
row/group split 之间按结果择优，唯一 outer 方案固定为
`StratifiedKFold(n_splits=5, shuffle=true, random_state=20260915)`。五个 validation folds 互不重叠，
3897 行每行恰好验证一次。outer partition 在任何 4C 评分前物化并冻结，SHA-256 为：

`62235d2fde916ce00cb4fdea0224a08339c9ac26226c43f294074c6160e1fea6`

结论只适用于同一 `v16n_2x_noise` source-train 上的事后开发复核。若后续任务复用这些 folds，它们仍是
同一累计模型选择链，不得称为新的 held-out 验证。source-test 读取次数为 **0**，RL 训练次数为 **0**。

每折清洗只在 outer-train 拟合，得到 65 个 clean features，再把保存的 original 1-based 映射应用到
outer validation。每个 inner 评分训练折单独拟合 StandardScaler；endpoint scaler 只在完整
outer-train 拟合。审计对 36,978 条候选记录完成 clean 0-based ↔ original 1-based 交叉校验。

## 3. 锁定评分与搜索

唯一 J 为 fold-local
`StandardScaler + LogisticRegression(C=1.0, solver=liblinear, class_weight=balanced, max_iter=5000)`
的 Balanced Accuracy；三次 5-fold 各取均值，再计算 `mean - 0.5 × sample SD`。只运行 K=32，未比较
K、分类器、指标、repeat/fold 数、聚合器、惩罚或 LR 超参数。

每折从空集执行 exact-K exhaustive forward，再最多两轮完整 `32×33=1056` 邻居的 best-improvement
single-swap。reward、argmax、archive 和 stop 使用同一 J；只有严格改善超过 `1e-12` 才接受。

| outer fold | forward LR BAcc | swap LR BAcc | 增益 (pp) | 接受轮数 | 停止原因 |
|---:|---:|---:|---:|---:|:---|
| 0 | 92.3804% | 92.5045% | +0.1241 | 2 | 两轮预算结束，不声称局部最优 |
| 1 | 90.5322% | 90.9035% | +0.3713 | 2 | 两轮预算结束，不声称局部最优 |
| 2 | 90.9142% | 90.9231% | +0.0089 | 2 | 两轮预算结束，不声称局部最优 |
| 3 | 92.4654% | 92.4654% | 0.0000 | 0 | 第一轮无严格改善 |
| 4 | 91.5791% | 91.5791% | 0.0000 | 0 | 第一轮无严格改善 |
| **fold mean** | **91.5743%** | **91.6751%** | **+0.1009** | — | **3/2/0** |
| **pooled OOF** | **91.5746%** | **91.6756%** | **+0.1010** | — | — |

前三折接受两次交换只表示达到冻结的两轮预算；后两折完整检查第一轮 1056 个邻居后严格停止。

## 4. 新 outer partition 上的静态定位

除 All Features 外，所有子集均在每个新 outer-train 内重新生成，没有搬用任务 2 的旧子集或旧 outer
数值。表中主指标均为 LR Balanced Accuracy；SD 是五个开发 folds 的描述性离散度。

| 方法 | fold mean ± SD | pooled OOF | 选择规则 |
|:---|---:|---:|:---|
| All Features | **91.8579% ± 0.7875 pp** | **91.8582%** | 每折 outer-train 的全部 65 clean features |
| MI-32 | 91.5176% ± 0.9597 pp | 91.5180% | 每折 outer-train 内重新计算 MI |
| LR-forward K=32 | 91.5743% ± 0.8611 pp | 91.5746% | 锁定 J 的 exhaustive forward |
| LR-forward + swap≤2 | 91.6751% ± 0.7878 pp | 91.6756% | 锁定 J 的完整两轮 best-improvement |
| Task-2 DT-forward K=32 | 90.7281% ± 0.3909 pp | 90.7276% | 每折重建的 5-fold DT Accuracy forward |
| Task-2 DT-forward + full single-swap | 90.9260% ± 0.5189 pp | 90.9262% | 每折重建，严格停止，最多 32 轮 |

Task-2 DT single-swap 在 outer folds 0/1/2/3/4 分别接受 2/2/0/2/2 轮，并均通过完整下一轮邻域
确认无严格改善。该 DT 轨道只作静态定位，不参与 4C GO-development 判定。All Features 的 pooled OOF
仍最高，但这是同一 post-selection development partition 上的定位，不是最终方法冻结或泛化排名。

DFS 状态保持：**`unavailable / protocol provenance incomplete`**。没有使用人工列表、旧 oracle 结果或
source-test 补数。

## 5. 成本与停止审计

主 LR 搜索逐折成本：

| fold | requests | unique subsets | cache hits | inner LR fits | case wall (s) |
|---:|---:|---:|---:|---:|---:|
| 0 | 3696 | 3599 | 97 | 53,985 | 209.22 |
| 1 | 3696 | 3599 | 97 | 53,985 | 200.03 |
| 2 | 3696 | 3599 | 97 | 53,985 | 149.16 |
| 3 | 2640 | 2607 | 33 | 39,105 | 135.64 |
| 4 | 2640 | 2607 | 33 | 39,105 | 138.25 |
| **合计** | **16,368** | **16,011** | **357** | **240,165** | **832.29** |

每折均满足 4000 unique subsets 与 60,000 classifier-fold fits 上限。恒等式
`requests = unique + cache hits` 和 `LR fits = 15 × unique` 对五折全部成立。forward checkpoint 每折固定
1584 unique subsets、23,760 fits；endpoint outer LR 评价每方法每折另计 1 次拟合，六方法共 30 次，
MI 共 5 次拟合，均未混入主搜索预算。

Task-2 DT 静态轨道合计 21,648 requests、20,967 unique subsets、681 cache hits、104,835 个 DT fold
fits；其完整局部停止成本独立报告，不用于主 LR 成本门槛。wall time 是整折 LR、DT、MI 与 endpoint
评价的端到端时间；并行 worker fit seconds 不等于墙钟时间。

## 6. 独立审计与验证

`experiments/k32_reward_robustness_v1/audit.json` 状态为 `passed`：

- outer partition 覆盖 3897 行，每行 validation 恰好一次，train/validation 互斥；
- 每折清洗映射由 outer-train 独立重算一致，三次 inner repeat 均覆盖 outer-train 且折内互斥；
- 36,978 条 LR/DT 唯一候选坐标全部交叉校验；
- forward 每步完整 argmax、swap 全邻域 argmax、严格接受/停止和 round cap 全部复核；
- 主 J forward/swap endpoint 独立复算最大绝对误差 `0`；
- 六个静态 endpoint 的 outer LR 独立复算最大绝对误差 `0`；
- pooled OOF 独立复算最大绝对误差 `1.11e-16`；
- source-test 使用次数 0，数据隔离、fold-local scaler 与成本恒等式全部通过。

交付后的代码验证：

- 聚焦测试：15 passed（4C、4B scorer、feature mapping）；
- 全量测试：178 collected/executed，178 passed，0 failed，0 skipped，0 collection errors；
- Ruff：`src/run_k32_robustness.py`、`src/audit_k32_robustness.py`、
  `tests/test_k32_robustness.py` 全部通过。

## 7. 产物与复现

主要产物：

- 协议：`documents/research-plan/04c-k32-robustness-protocol.md`；
- 配置：`configs/v16n/k32_reward_robustness_v1.toml`；
- 正式目录：`experiments/k32_reward_robustness_v1/`；
- manifest/partition：`manifest.json`、`outer-partition.json`、`partition-frozen.json`；
- 每折 context、LR/DT candidates、逐步 path/archive、双坐标 endpoint 和 OOF prediction；
- 汇总：`analysis/fold_results.csv`、`pooled_results.csv`、`costs.csv`、`oof_predictions.csv`；
- 决定：`analysis/final-decision.json`；
- 独立审计：`audit.json`。

复现命令：

```bash
PYTHONPATH=src conda run -n dl-lab python src/run_k32_robustness.py --stage prepare
PYTHONPATH=src conda run -n dl-lab python src/run_k32_robustness.py --stage run --jobs 24
PYTHONPATH=src conda run -n dl-lab python src/run_k32_robustness.py --stage finalize
PYTHONPATH=src conda run -n dl-lab python src/audit_k32_robustness.py
```

## 8. 最终路线决定

**任务 4C：NO-GO。任务 4D：blocked / 不准入。**

失败原因是锁定的两项效应门槛同时未达到，不是工程或审计失败。按预先冻结的停止规则，停止当前静态
PPO、5A 标签扩展和新的 K/reward 扫描；5A 必须等待 4D 报告的条件在本分支无法满足，因为 4D 未准入。
后续若要重新评价泛化，只能在方法与问题重新立项后使用新增、此前未参与选择的数据，不能继续复用
本 outer partition 将其包装成独立确认。

