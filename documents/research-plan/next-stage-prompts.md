# 4A/4B 后续可复制任务 Prompt

版本：`post-4ab-prompts-v2`  
日期：2026-09-15  
适用仓库：`/root/feature-select/Radar-Ship-FS`

## 使用说明

任务 4A 与 4B 已完成，不要再次执行。4B 的正式状态是整体 NO-GO；下面保留的是看过结果后新提出的 K=32 分支，不能改写成原 4B 已通过。

建议顺序：

1. 任务 4C-0 与任务 3A 可并行；
2. 4C-0 完成后执行任务 4C；
3. 任务 4C 通过后执行任务 4D；
4. 任务 4D 完成后，无论其 GO/NO-GO 都可执行 5A；只有 4D 和 5A 都通过才执行 5B；
5. 方法冻结且有新增独立数据后执行任务 6；
6. 任务 7 只在 3A 恢复真实环境元数据后考虑。

每个 prompt 都假设 agent 是全新会话。路线、已知结果、门槛和共同约束统一记录在：

`/root/feature-select/Radar-Ship-FS/documents/research-plan/next-stage-roadmap.md`

---

## Prompt：任务 4C-0（应最先执行）

```text
你正在一个全新会话中工作，不具备此前对话上下文。

请在 /root/feature-select/Radar-Ship-FS 中完整执行“任务 4C-0：仓库兼容边界与全量测试修复”。

开始前必须完整阅读：

1. documents/research-plan/next-stage-roadmap.md
2. documents/research-plan/04a-action-support.md
3. documents/research-plan/04b-reward-alignment.md
4. experiments/action_support_v1/manifest.json、delivery-manifest.json 与 audit.json
5. experiments/reward_alignment_v1/manifest.json
6. experiments/reward_alignment_v1/audit.json
7. experiments/reward_alignment_v1/verification.json
8. 当前 git status、相关 git history、测试配置和 AGENTS.md（若存在）

任务性质与边界：

- 这是后续正式实验的工程可信度门槛，不做科研调参，不训练正式 PPO/DQN，不改写 4A/4B 结论。
- 4A/4B 的定向测试与独立数值审计已经通过；本任务处理的是仓库全量 pytest 无法完整收集的问题。
- 当前已知有 11 个 collection errors，涉及已移动或删除后仍被引用的 methods.configure、methods.advice、radar_ship_fs.legacy 和顶层 stage2 入口。
- 已知受阻模块包括 tests/test_basic_baselines.py、test_engine.py、test_harness.py、test_invariants.py、test_methods.py、test_stable_architecture.py、test_stage2_beta_sweep.py、test_stage2_budget_sweep.py、test_stage2_dt_test.py、test_stage2_guidance_sweep.py、test_stage2_rl.py；仍须以实际复现为准。
- active src/run_irfs.py 与 src/harness/aggregate.py 也可能存在悬空 import，因此不能简单认定全是陈旧测试。

执行要求：

1. 先复现并保存 pytest --collect-only -q 和 pytest -q 的原始结果，准确列出 collected、executed、passed、failed、skipped、collection errors。
2. 用 git history 和现行 runtime 逐项判断：恢复薄兼容层、迁移 active 调用，或在有充分证据时退役已经失效的契约。
3. 检查所有受影响的 stage2 wrapper、harness、invariants、PPO 和 methods tests。
4. 不得通过全局 skip、批量 xfail、删除仍代表现行契约的测试、降低断言或只改 PYTHONPATH 来掩盖问题。
5. 不得恢复 active stable 模块对已明确隔离的 legacy 实现的反向依赖；需要兼容时使用边界清楚、带测试的薄适配层。
6. 不修改 4A/4B 的协议、配置、manifest、实验结果和 hash。
7. 修改后依次验证：
   - pytest --collect-only -q 为 0 collection errors；
   - 全量 pytest -q 真正执行到结束，所有现行 active tests 通过；
   - 只有证明不覆盖当前 v16n、scorer、PPO、harness 或 invariants 路径的历史数据依赖测试，才可进入单独命名、明确计数的 legacy-data profile；其他失败均为 NO-GO；
   - 4A、4B、PPO、harness、invariants 的相关测试实际被执行；
   - Ruff 对修改文件通过。
8. 若全量执行后暴露数据 fixture 缺失或真正的测试失败，继续修复属于本任务范围的兼容/隔离问题；无法修复时逐项报告，不得再写“全量 pytest 已运行”来暗示全量通过。

输出：

- 实现与必要测试；
- documents/research-plan/04c0-repository-test-repair.md；
- 根因到修复的逐项映射；
- 最终收集数、执行数、passed/failed/skipped/errors；
- 是否允许进入任务 4C 的明确 PASS/NO-GO。
```

---

## Prompt：任务 4C（K=32 锁定复核）

```text
你正在一个全新会话中工作，不具备此前对话上下文。

请在 /root/feature-select/Radar-Ship-FS 中评估并执行“任务 4C：K=32 锁定稳健性复核与静态基线定位”。

开始前必须完整阅读并核查实际产物：

1. documents/research-plan/next-stage-roadmap.md
2. documents/research-plan/04c0-repository-test-repair.md
3. documents/research-plan/04a-action-support.md
4. documents/research-plan/action-support-protocol.md
5. experiments/action_support_v1/manifest.json 与 audit.json
6. documents/research-plan/04b-reward-alignment.md
7. documents/research-plan/reward-alignment-protocol.md
8. experiments/reward_alignment_v1/analysis/final-decision.json
9. experiments/reward_alignment_v1/analysis/selected-scorer.json
10. experiments/reward_alignment_v1/analysis/aligned_search_results.csv
11. experiments/reward_alignment_v1/audit.json
12. 如果存在，documents/research-plan/03a-data-lineage-audit.md
13. 当前 scorer、清洗、forward greedy、single-swap、成本计数和审计代码

先做准入检查：

- 任务 4C-0 必须已使全量 pytest 零 collection errors，并给出 PASS；否则只输出 blocked 报告，不运行正式实验。
- 保留事实：任务 4B 整体是 NO-GO，unique_main_reward=null；K=16 已关闭。
- K=32 在 4B 中为 +0.4149 pp、5/5 正，因此本任务是看过结果后新增的 post-selection development robustness check。
- 本任务不能称为原 4B 通过、预注册确认或独立泛化验证。
- 4C、4D、5A、5B 若使用同一 outer folds，属于同一条累计模型选择链；本任务不得把这些 folds 称为新的 held-out 验证。

正式评分前新建并冻结：

- documents/research-plan/04c-k32-robustness-protocol.md
- configs/v16n/k32_reward_robustness_v1.toml
- experiments/k32_reward_robustness_v1/

固定实验，不得再择优：

- K 只取 32。
- scorer 只取 fold-local StandardScaler + LogisticRegression：
  C=1.0、solver=liblinear、class_weight=balanced、max_iter=5000。
- 指标只取 LR Balanced Accuracy。
- inner 评分固定 3 次 5 折，标量固定为三个 repeat 均值的 mean - 0.5 × sample SD。
- 不比较 K、分类器、指标、repeat/fold 数、聚合器、惩罚权重或 LR 超参数。
- 每个 outer-train 从空集重跑 exact-K exhaustive forward，再执行最多两轮完整 best-improvement single-swap；每轮检查全部 K×(65-K) 邻居，严格改善才接受。
- reward、候选择优、archive 和 stop 都使用同一个 J；本任务不训练 RL。
- source-test 读取次数必须为 0。

outer 划分必须在查看结果前物化、保存并 hash：

- 建立一个五个 validation folds 互不重叠、每行恰好验证一次的新 outer partition。
- 若 3A 已审计出真实 base-scene/group ID，使用 StratifiedGroupKFold，并保证同源增强不跨 fold。
- 否则使用 StratifiedKFold(n_splits=5, shuffle=true, random_state=20260915)，并把结论限定为同一 source-train 上的事后开发复核。
- 不能根据两条轨道的表现选择 row split 或 group split；只能根据 3A 在实验前确认的元数据可用性决定。
- 清洗仅在 outer-train 拟合；每个评分 scaler 仅在 inner 训练折拟合；outer endpoint scaler 仅在完整 outer-train 拟合。

固定报告：

- LR-forward endpoint 到最多两轮 swap endpoint 的逐 fold 和 pooled out-of-fold LR BAcc 增益；
- All Features、MI-32、LR-forward、LR-forward+swap、任务 2 中可比的 K=32 静态基线；除 All Features 外，所有基线子集必须在每个新 outer-train 内重新生成，不能搬用任务 2 的旧子集或旧 outer 数值；
- 每步 argmax、接受/停止原因、唯一子集、模型拟合、cache、请求与 wall time；
- clean 0-based 与 original 1-based 坐标及交叉校验。
- DFS 只有满足既有 provenance 门槛才加入；否则写 unavailable / protocol provenance incomplete，禁止用人工列表或 source-test oracle 补数。

新扩展协议在读取任何 4C 结果前冻结以下 GO-development 数值门槛；这些数值沿用 4B 原先的 K=32 分项门槛，不构成对原 4B 预注册的修改：

1. swap endpoint 相对 LR-forward 至少 4/5 outer folds 为正；
2. 五折平均 LR BAcc 增益至少 +0.2 pp；
3. 每折不超过 4000 个唯一评分子集、60000 次分类器折拟合；
4. 数据隔离、fold-local scaler、坐标、逐步 argmax、严格停止、成本恒等式和独立数值复算全部通过。

失败则停止当前静态 PPO、5A 标签扩展和新的 K/reward 扫描。通过时只冻结一个“K=32 机制实验用 development scorer”，不得修改 4B 的整体 NO-GO，也不得声称外部泛化。

输出：

- 实现、必要测试、协议、配置、manifest、完整结果和 audit；
- documents/research-plan/04c-k32-robustness.md；
- 是否进入任务 4D 的明确 GO-development/NO-GO；5A 必须等待 4D 报告完成。
```

---

## Prompt：任务 4D（PPO 参数更新贡献）

```text
你正在一个全新会话中工作，不具备此前对话上下文。

请在 /root/feature-select/Radar-Ship-FS 中评估并执行“任务 4D：全合法动作空间下的 K=32 策略更新贡献消融”。

开始前必须完整阅读：

1. documents/research-plan/next-stage-roadmap.md
2. documents/research-plan/04c0-repository-test-repair.md
3. documents/research-plan/04a-action-support.md
4. documents/research-plan/04b-reward-alignment.md
5. documents/research-plan/04c-k32-robustness.md
6. 任务 4C 的 protocol、config、manifest、逐 fold 结果和 audit
7. 当前 PPO environment、policy、rollout、trainer、archive、checkpoint、evaluator 和 run-session 实现

准入检查：

- 4C-0 必须 PASS，4C 必须明确 GO-development。
- 缺少任一条件时，只输出 blocked/no-go 报告，不训练 PPO，也不退回旧 DT reward。
- 本任务仍是 post-selection K=32 机制研究，不能撤销 4B 整体 NO-GO。
- 4C、4D、5A、5B 若使用同一 outer folds，属于同一条累计模型选择链；本任务不得把这些 folds 称为新的 held-out 验证。

任何正式 rollout 前新建并冻结：

- documents/research-plan/04d-k32-policy-ablation-protocol.md
- configs/v16n/k32_policy_ablation_v1.toml
- experiments/k32_policy_ablation_v1/
- 方法、policy seeds、预算、请求上限、指标、聚合和 GO 门槛的 hash。

共同设置：

- K=32；每个 outer fold 从任务 4C 的 LR-forward endpoint 开始。
- 每个 episode 重置到共同起点，最多执行两次完整交换。
- 主动作 mask 使用确定性的 full_legal：
  - 删除阶段全部 32 个已选特征可见；
  - 加入阶段全部 33 个未选特征可见；
  - 单交换 32×33 个组合均直接具有支持。
- 为 full_legal 增加显式配置和测试，核对每阶段 mask 基数、合法性、确定性和坐标；不得靠候选 pool 截断。
- 质量/MI 信息若保留，只能作为实验前固定的 actor 输入或 logit prior，并在正常 PPO 与 lr=0 PPO 中完全一致。
- 4A 的 4+4 机制不进入主方法择优。
- J 固定为 4C scorer。episode 始终从 exact-K 状态开始；删除半步 reward=0 且不调用 J；完成“删除后加入”的合法完整交换、重新得到 exact-K 子集后，调用一次 J，并令 reward = 100 × [J(S_new)-J(S_previous)]。cache hit 也算一次 evaluator 请求；stop action 不重复调用 J，reward=0。gamma=1 时两次完整 swap 的累计奖励必须严格等于终点相对起点的 J 增益。
- gamma=1；correlation penalty、sparsity bonus、feature-ID、proxy shaping 全部为 0/关闭。
- reward、archive、stop、checkpoint 选择只在完整 exact-K 状态上使用同一个 J。
- outer validation 不进入训练、reward、archive、checkpoint 或动作选择。

四个主方法：

1. full-legal uniform random；
2. sampled best-improvement：第一轮无放回评分 128 个完整单交换，若有严格改善取最佳；第二轮从新状态再评分 128 个；
3. 完整 PPO 流程但 learning rate=0；
4. 正常更新 PPO。

另做确定性集成测试：在共享初始化、输入、mask 和 RNG 的条件下，“完全不调用 update”与 learning rate=0 的参数及动作输出应符合预先写明的等价预期。若不等价，查明副作用，并把 no-update 作为第五个主对照。

任务 4C 的最多两轮 exhaustive best-improvement 只作高成本参考，不是同预算对手。

预算与分析单位：

- 每个 method × outer fold × policy seed 的主搜索预算上限同为 256 次完整交换后的 J evaluator 请求；重复子集和 cache hit 同样消耗请求预算；
- checkpoints 固定在 evaluator 请求数 32/64/128/256；若按预注册规则提前停止，后续预算点保持最后 best 值；
- 唯一子集数、重复率、cache hit 与真实模型拟合数另报，不能用 4096 次请求换取 256 个唯一子集；
- 训练结束后的随机/确定性策略评价使用各策略臂完全相同、预先冻结的独立 J 请求配额；这部分计入总物理成本，且不得回流搜索 archive；
- 每个 outer fold 使用 5 个配对 policy seeds；
- 先在 fold 内聚合 policy seeds，再以 5 个 outer folds 为分析单位；不得把 25 次 run 当 25 个独立数据集样本；
- forward 初始化与监督/网络初始化成本另报，不能隐藏在 cache 中。

必须补齐真实学习诊断：

- best-so-far J 的 1–256 曲线和 checkpoints；AUC 固定为 evaluator 请求 1–256 上 100 × [best_J(b)-J_start] 的算术平均，提前停止后保持最后值；
- 达到 4C exhaustive 参考增益所需预算，未达到记 censored；
- frozen inner-best endpoint 的 outer LR BAcc；
- 当前策略随机 rollout 均值与确定性 rollout，不能只报 archive best；
- 参数更新前后 L2 距离；
- 固定 probe states 上动作分布 KL 与动作排序；
- entropy、approximate KL、clip fraction 和真实梯度范数；
- 唯一子集、重复、cache hits、总请求、评分拟合和 wall time；
- lr=0 组参数必须逐位不变，resume 必须确定性复现。
- 当前 run_session 若仍把梯度范数写成固定 0，占位记录必须修复并测试。

GO 门槛全部满足才算策略更新有效：

1. 正常 PPO 的参数与固定 probe 动作分布发生有限、非零、可重复的变化；
2. 以 1e-6 pp 为 tie epsilon，PPO AUC 相对 lr=0 PPO 与 sampled best-improvement 都至少 4/5 outer folds 为正，且相对两者中更强者的五折平均 AUC 优势至少 0.05 pp；
3. PPO endpoint 相对共同 forward 起点的 outer LR BAcc 至少 4/5 为正，平均至少 +0.2 pp；
4. PPO 相对 lr=0 PPO 及最强简单对照的 outer LR BAcc 均至少 4/5 为正，平均优势至少 +0.1 pp。

任一项失败，结论为 PPO 参数更新未显示独立贡献，停止任务 5B；任务 5A 仍可作为非 RL 路线执行。不得扫描网络结构、学习率、熵系数、候选池或预算。

输出：

- 实现、必要测试、冻结协议/配置、manifest、逐 run 结果、成本和 audit；
- documents/research-plan/04d-k32-policy-learning-ablation.md；
- 对初始化、随机采样、简单搜索与参数更新贡献的分离；
- PPO 参数更新分支的明确 GO/NO-GO，供 5A 完成后联合判定 5B 准入。
```

---

## Prompt：任务 5A（监督动作排序）

```text
你正在一个全新会话中工作，不具备此前对话上下文。

请在 /root/feature-select/Radar-Ship-FS 中评估并执行“任务 5A：K=32 监督动作排序的可迁移性”。

开始前完整阅读：

1. documents/research-plan/next-stage-roadmap.md
2. documents/research-plan/04b-reward-alignment.md
3. documents/research-plan/04c-k32-robustness.md
4. 任务 4C 的 protocol、config、manifest、评分缓存、逐 fold 结果与 audit
5. 如果已完成，documents/research-plan/04d-k32-policy-learning-ablation.md
6. 如果存在，documents/research-plan/03a-data-lineage-audit.md
7. 当前状态特征、图特征、动作编码和候选打分实现

准入条件：

- 任务 4C 必须 GO-development，任务 4D 必须已经执行完成并产出 GO 或 NO-GO 报告；否则只输出 no-go，不生成大规模标签。
- 4D 可以 GO 或 NO-GO。4D NO-GO 时，5A 是独立的非 RL 搜索路线，不能包装成 RL 结果。
- 4B 整体仍是 NO-GO；这里使用 4C 冻结的 K=32 development scorer。
- 4C、4D、5A、5B 若使用同一 outer folds，属于同一条累计模型选择链；本任务不得把这些 folds 称为新的 held-out 验证。

正式生成标签前冻结协议、模型族、数据分组、预算和门槛，使用新目录，不覆盖 4B/4C。

标签只能在相应 outer-train 内由固定 J 生成：

(state subset, removed feature, added feature, delta J, paired-delta SD)

要求：

- paired-delta SD 必须是三次配对 repeat 的 delta J sample SD，不是候选绝对 J 的 SD。
- outer validation 的 LR BAcc 不得成为标签、样本权重、模型选择或 checkpoint 信号。
- 不得把高度重叠的单条 swap 行随机拆分。
- 一个起点、完整轨迹及其全部邻居必须作为不可拆分 group。
- 至少做 leave-one-outer-context-out；这些 contexts 底层样本仍可能重叠，所以只能称 context transfer。
- 若 3A 恢复真实 scene/group ID，再增加 group-disjoint 轨道；不得按行序、相似度或聚类伪造 group。
- 复用 4B/4C 缓存时保存来源 hash，同时报告复用成本与从零标签生成成本。

本任务先只研究监督排序，不训练 PPO。预先冻结并比较：

1. 随机排序；
2. 固定 MI/质量先验；
3. 轻量线性或 pairwise ranker；
4. 一个预先冻结的小型监督动作排序模型。

禁止宽泛超参数扫描。报告：

- Spearman、pairwise accuracy、NDCG@32；
- top-32 正增益动作 recall 与 enrichment；
- 按完整 context/trajectory 分组的泛化；
- 由排序器驱动的 32/64/128/256 次 J evaluator 请求预算 best-so-far 曲线与 AUC；cache hit 也消耗请求预算，唯一子集与实际拟合数另报；
- frozen endpoint 的 outer LR BAcc，仅作停止护栏；
- 标签生成、训练、推理、在线评分、请求/cache 和 wall time。

进入 5B 的门槛全部满足：

1. held-out context top-32 enrichment 至少 4/5 大于 1，平均至少 2×；若某 context 没有任何正 delta J 动作，该 context 直接记为门槛失败，不能剔除；
2. sampled best-improvement 必须复用 4D 已冻结的 full-legal 起点、两轮各 128 个无放回邻居、严格改善规则和配对 seeds；以 1e-6 pp 为 tie epsilon，ranker AUC 减 sampled-best AUC 至少 4/5 contexts 为正，五折平均至少领先 0.05 pp；
3. frozen endpoint 的 outer LR BAcc 相对最强简单对照至少 4/5 为正，平均至少 +0.1 pp；
4. 若只在随机行拆分有效、按 context/trajectory 分组后失效，必须 NO-GO。

输出：

- documents/research-plan/05a-supervised-action-ranking-protocol.md；
- documents/research-plan/05a-supervised-action-ranking.md；
- 实现、测试、数据 manifest、成本和 audit；
- 是否允许 5B 的明确 GO/NO-GO。
```

---

## Prompt：任务 5B（监督初始化 + PPO）

```text
你正在一个全新会话中工作，不具备此前对话上下文。

请在 /root/feature-select/Radar-Ship-FS 中评估并执行“任务 5B：监督初始化后的 PPO 微调增量”。

开始前完整阅读：

1. documents/research-plan/next-stage-roadmap.md
2. documents/research-plan/04d-k32-policy-learning-ablation.md
3. documents/research-plan/05a-supervised-action-ranking.md
4. 4D 与 5A 的全部 protocol、config、manifest、逐 run 结果和 audit
5. 当前 PPO policy/trainer 与监督 ranker/初始化代码

只有 4D 和 5A 两份报告都明确 GO 才能运行。否则只输出 no-go 报告，不训练复杂模型。

- 4C、4D、5A、5B 若使用同一 outer folds，属于同一条累计模型选择链；本任务不得把这些 folds 称为新的 held-out 验证。

任何正式训练前冻结新协议、配置、seeds、预算、方法和门槛。固定 K=32，并完整继承 4D 的：

- full-legal support；
- J scorer 与 reward 定义；
- outer folds 与共同 forward 起点；
- 每 episode 最多两次交换；
- 256 次在线 J evaluator 请求预算与请求数 32/64/128/256 checkpoints；cache hit 同样消耗请求预算；
- archive/stop/checkpoint 和唯一子集/实际拟合/墙钟成本口径。

比较：

1. sampled best-improvement；
2. 5A 监督策略冻结后直接排序；
3. 从头 PPO；
4. 监督预训练策略，learning rate=0；
5. 监督预训练策略 + PPO 微调。

所有策略臂的 actor 结构、输入张量、动作头和参数量必须完全一致，只有初始化来源不同。若监督 ranker 不能直接映射到 actor，必须在查看结果前冻结转换 adapter，并加入容量匹配对照。所有方法共享数据边界、动作支持、在线评分预算和评价方式。标签生成与预训练成本必须计入端到端成本，不能只报告在线阶段。

主要问题是 PPO 微调是否在冻结监督策略之上产生增量。GO 必须同时满足：

- 以 1e-6 pp 为 tie epsilon，pretrained+PPO 的 best-so-far AUC 相对 frozen pretrained 至少 4/5 contexts 为正，且五折平均至少领先 0.05 pp；
- frozen endpoint 的 outer LR BAcc 相对 frozen pretrained 至少 4/5 contexts 为正，平均优势至少 +0.1 pp；
- 微调后的参数与固定 probe 动作分布确实发生有限、非零、可重复的变化；
- 结果通过数据隔离、预算、resume 和独立复算审计。

任一条件失败，就把有效贡献归因于监督排序，停止静态 PPO 路线。不得因为 pretrained+PPO 的绝对值最高、却没有稳定超过 frozen pretrained，而声称 RL 有贡献。

输出：

- documents/research-plan/05b-pretrained-ppo-protocol.md；
- documents/research-plan/05b-pretrained-ppo.md；
- 实现、测试、配置、manifest、完整成本与 audit；
- 监督预训练和 PPO 微调贡献的分离结论；
- 进入任务 6 的唯一最终候选。
```

---

## Prompt：任务 6（最终锁定评价）

```text
你正在一个全新会话中工作，不具备此前对话上下文。

请在 /root/feature-select/Radar-Ship-FS 中评估并执行“任务 6：最终方法冻结、强基线比较与新增数据评价”。

开始前完整阅读：

1. documents/research-plan/next-stage-roadmap.md
2. documents/research-plan/02-baselines.md
3. documents/research-plan/04c-k32-robustness.md
4. 若存在，documents/research-plan/04d-k32-policy-learning-ablation.md
5. 若存在，documents/research-plan/05a-supervised-action-ranking.md
6. 若存在，documents/research-plan/05b-pretrained-ppo.md
7. 所有对应 protocol、config、manifest 与 audit
8. documents/research-plan/protocol.md 中的 source-test 与 DFS provenance 规则

准入检查：4C 必须明确 GO-development，且 4D、5A 都必须存在明确的 GO/NO-GO 报告。缺失报告不能按 NO-GO 处理；4C 失败时，本 K=32 分支的任务 6 只输出 blocked，不进行 final 评价。

在读取任何 final 标签前，仅根据前序冻结报告生成并 hash 锁定 experiments/final_method_selection_v1/final-method-selection.json。唯一主方法按以下规则选择：

1. 5B GO：pretrained+PPO；
2. 5B 明确 NO-GO 或因 4D 未通过而未准入、且 5A GO：冻结监督排序策略；
3. 5A 明确 NO-GO 且 4D GO：scratch PPO；
4. 仅当 4C GO 且 4D、5A 均已有明确 NO-GO 报告：4C 的 LR-forward+最多两轮 exhaustive swap。

final 表可保留多个冻结基线，但不得按 final 结果更换主方法、K、checkpoint 或超参数。

最终评价数据必须是方法冻结后取得的新仿真场景、新生成批次、真实数据，或有证据证明此前完全未参与设计/选择的外部数据。现有 source-test 已被反复查看，不能重新命名为独立 test。

如果没有合格的新数据：

- 不要再次读取现有 source-test 来制造“最终确认”；
- 输出 blocked 报告、冻结方法清单和最小数据获取/生成规范；
- 将已有结论明确限定为 development evidence。

若使用新仿真批次，必须在生成前冻结 generator 版本/hash、物理参数分布、scene seeds、样本量和排除规则，只生成一个正式批次，禁止多批生成后择优。相同生成器的新批次只能称 prospective in-distribution replication；真实域泛化需要真实数据或不同来源数据。

有合格 final data 时，在打开标签前冻结协议、一次性 evaluation script 和 manifest。所有特征选择只在 final-train 内进行，final evaluation 只执行一次且结果不得回流调参。

固定比较：

1. All Features；
2. MI-32；
3. LR-forward K=32；
4. LR-forward + 最多两轮 exhaustive best-improvement swap；
5. 256 次 J evaluator 请求预算的 sampled best-improvement；
6. 任务 2 中可复现且 classifier-aligned 的强静态基线；
7. 通过门槛的 4D/5A/5B 唯一最终胜者；
8. 仅当 DFS 的训练代码、来源/许可证、commit、环境、配置、seeds、checkpoint、train-only selection trajectory 和坐标全部可追溯时加入 DFS。

不得从旧 oracle 列表或 source-test 选择 DFS penalty/subset。DFS 条件不满足时，正式表写 unavailable / protocol provenance incomplete。

主指标固定 LR Balanced Accuracy；同时报告 Accuracy、F1/AUC（若协议已冻结）、特征数、选择 Jaccard/频率、最差场景或 group、唯一评分数、模型拟合数、训练/选择/推理墙钟成本。给出配对结果及可信边界，不把 5 个 seeds 当独立数据集。

输出：

- documents/research-plan/06-final-locked-evaluation-protocol.md；
- documents/research-plan/06-final-locked-evaluation.md；
- 一次性脚本、配置、manifest、结果、成本和 audit；
- 能否支持论文级泛化主张的明确结论。
```

---

## Prompt：任务 7（环境条件化分支，条件任务）

```text
你正在一个全新会话中工作，不具备此前对话上下文。

请在 /root/feature-select/Radar-Ship-FS 中评估并执行“任务 7：环境条件化特征选择的前提验证”。

开始前完整阅读：

1. documents/research-plan/next-stage-roadmap.md
2. documents/research-plan/03a-data-lineage-audit.md
3. problem.md（位于 /root/feature-select/problem.md）
4. 当前数据生成、增强、shuffle、split 与环境元数据代码/manifest
5. documents/research-plan/reward-alignment-protocol.md 与 4B 的 fold-local LR scorer 实现
6. 若存在，任务 6 的 final-data protocol

准入必须全部满足：

- 3A 已恢复来源可靠的 base_scene_id 或等价父场景标识；
- 海况、擦地角、SCR 等环境变量来源可靠，且选择特征时实际可获得；
- 同源场景及增强副本可以保持在同一 outer fold；
- 各环境有足够样本支持训练与验证。

任一条件不满足时，只输出 blocked/no-go 和最小数据补充需求。协议还必须在读取性能结果前要求每个环境至少有 25 个独立 base scenes/类别，使五折中每个 validation fold 至少有 5 个 scenes/类别；任一 group fold 无法同时覆盖类别和环境时直接 NO-GO。不得按行序、相似度、聚类或结果反推伪造 scene/group/env。

本任务先验证“不同环境是否确实需要不同子集”，不直接训练复杂 PPO，也不依赖任务 4C 是否通过。任何评价前冻结物理环境表示或分箱、scene-group-disjoint outer split、K=32、4B 规格的 fold-local StandardScaler+LR Balanced Accuracy / 3×5 fold / mean_minus_0_5_sd scorer、总 J 请求预算、平均/最差环境指标和门槛。不得根据结果调整环境分箱。

比较：

1. 所有环境共享的全局静态 K=32；
2. 仅在各 outer-train 内得到的分环境静态 K=32；
3. 简单监督条件化选择器；
4. 若环境在决策时已知，可加入 contextual bandit 作为简单基线。

分环境方法自由度更大，主比较必须让所有方法使用相同总 J evaluator 请求预算；等环境预算只作次级结果。对训练中未见环境的回退固定为 outer-train 内得到的全局 K=32，不能在结果后改选。评价平均 LR BAcc、最差环境 BAcc、环境间方差、未见 scene/env 表现、选择稳定性和完整成本。

进入进一步条件化策略研究的门槛：

- 相对全局子集平均提高至少 +0.2 pp；
- 至少 80% outer folds 方向为正；
- 最差环境不恶化超过 0.2 pp；
- 控制总预算和模型复杂度后收益仍存在；
- 未见环境不明显崩溃。

门槛失败则停止环境条件化路线。简单监督条件化或 contextual bandit 已最好时保留简单方法。

只有任务具有真实序贯结构，例如不同特征有计算成本、按已观测特征决定下一项、需要学习何时停止，才允许提出多步 RL。若环境已知且只需一次选择子集，不要强行使用 PPO。

输出：

- documents/research-plan/07-environment-conditioned-feasibility-protocol.md；
- documents/research-plan/07-environment-conditioned-feasibility.md；
- 数据/split 审计、基线、成本和明确 GO/NO-GO。
```

---

## Prompt：任务 3A（尚未完成，可与 4C-0 并行）

```text
你正在一个全新会话中工作，不具备此前对话上下文。

请在 /root/feature-select 中执行 Radar-Ship-FS 的“任务 3A：v16n_2x_noise 数据谱系与场景分组审计”。

开始前必须完整阅读：

1. Radar-Ship-FS/documents/research-plan/next-stage-roadmap.md
2. Radar-Ship-FS/documents/research-plan/protocol.md
3. Radar-Ship-FS/documents/research-plan/03-search-diagnosis.md
4. Radar-Ship-FS/documents/v16n-data-analysis-report.md
5. 当前 dataset、数据生成代码、历史日志、manifest 和 git history

当前仓库中没有 Radar-Ship-FS/documents/research-plan/03a-data-lineage-audit.md；不要假设此任务已经完成。

追查 v16n_2x_noise train/test 的真实生成入口、上游输入、变换、噪声增强、shuffle、split、base scene、父子副本和环境参数。不要把旧 v16n 的增强倍数直接套用；不要按行序、相似度或聚类伪造 group ID；不要预设已经泄漏。

只有恢复可靠 parent/scene/env 元数据后，才做一个在结果前冻结的小型 row-level 与 group-aware split 敏感性比较。无法恢复时，列出缺失链路和最小补充材料，并把后续结论限制为行级 source-train development evidence。

输出 Radar-Ship-FS/documents/research-plan/03a-data-lineage-audit.md，并分别给出：

- 4C 是否可使用真实 group-aware outer split；
- 任务 7 是否具备环境条件化研究条件；
- 需要补充的原始 manifest、生成参数或 scene IDs。
```
