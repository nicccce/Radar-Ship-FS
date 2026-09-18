# Radar-Ship-FS 下一阶段科研路线与任务上下文

版本：`next-stage-roadmap-v3-post-4c-nogo`

日期：2026-09-17

适用仓库：`/root/feature-select/Radar-Ship-FS`

## 1. 本文件的用途

后续任务可能由彼此没有对话上下文的独立 agent 执行。每个 agent 必须先完整阅读本文件，再阅读对应任务列出的输入材料。不得假设其他任务已经完成；需要通过文件、配置、manifest 和实验产物检查前置条件。

本文件是下一阶段的路线说明，不追溯修改已经冻结的实验协议、配置 hash 或历史结果。若本文件与任务 1–3 的冻结协议冲突，历史结果仍按当时协议解释；新实验应另建协议和结果目录。

## 2. 研究目标

项目研究强化学习用于雷达目标识别的特征选择。目前的核心问题不是继续增加网络规模，而是回答三个可证伪的问题：

1. 现有动作候选池是否把有益交换排除在可达空间之外？
2. 训练期评分是否能够可靠预测最终目标分类器的开发验证收益？
3. 在相同起点、候选支持和评分预算下，PPO 参数更新是否优于零学习率、随机搜索和简单局部搜索？

原路线要求前两个问题都得到正面解决后才扩大策略学习，并要求策略更新显示独立增量后才开展监督预训练加 RL。任务 4A/4B 没有同时满足这一条件，原跨 K 路线已经停止。

任务 4C 已完成，K=32 锁定复核仍为 NO-GO：收益为 +0.1009 pp，3/2/0 正/平/负，两项效应门槛均未达到。当前静态 RL 分支停止，4D、5A、5B 和该分支的任务 6 均不准入。4C 没有训练 RL，不能把该停止决定解释为已证明 PPO 无效。当前下一步是任务 3A 数据谱系审计；另见 `04c-review-and-next-steps.md`。

## 3. 已完成工作及可信边界

### 3.1 任务 1：实验协议审计

已完成：

- 统一 clean 0-based、original 1-based 和 raw 0-based 特征坐标。
- 修复手写比较脚本的错列问题。
- 新科研配置关闭随机 feature ID 对 node feature、候选池、shaping 和目标的影响。
- PPO 增加 shared inner-CV 路径。
- 区分 MI Top-K、MI-ordered accept 和 exhaustive forward greedy。
- 将读取 source-test 后选择子集、K 或 penalty 的入口隔离为 oracle diagnostic。
- DFS 因训练代码和可追溯产物缺失，不进入正式比较。

需要勘误：

- `uniform` 表示统一奖励信号，`fixed` 表示固定状态编码器；对应方法仍创建 Q 网络并执行 DQN 更新。
- 当前实现应描述为工程中的多智能体 DQN 变体。在没有逐项复核论文前，不得称为原始 MARLFS/IRFS 的完整复现。

主要材料：

- `documents/research-plan/01-protocol-audit.md`
- `documents/research-plan/protocol.md`
- `configs/v16n/research_baseline.toml`
- `src/radar_ship_fs/experiment/runner.py`

### 3.2 任务 2：统一强基线

已完成 5 seeds、`K={8,16,32}` 和自动 K 的经典基线，共 70 个冻结子集。实现、坐标、成本计数和保存结果通过定向审计。

关键结果：

- 在 DT inner-CV Accuracy 搜索目标上，`forward_greedy_single_swap, K=16` 为最强开发锚点，平均约 `0.9324`。
- 历史已观察 source-test 上，LR 主分类器的 All Features BAcc 约为 `0.9139`；该 test 只能作复用诊断。
- 当前选择评分为 DT Accuracy，主要最终评价为 LR Balanced Accuracy，存在分类器和指标错配。
- 任务 2 正确记录了不同算法的评分成本，但没有建立相同硬预算下的搜索算法胜负。

这些强基线的冻结结果继续保留；任务 4B 随后建立了 LR Pipeline + Balanced Accuracy 对齐轨道。任何新 outer 划分中的正式比较仍须在相应 outer-train 内重建可比基线，StandardScaler 只能在每个评分训练折内拟合。

主要材料：

- `documents/research-plan/02-baselines.md`
- `documents/research-plan/baseline-protocol-addendum.md`
- `experiments/unified_strong_baselines_v1/`
- `src/run_unified_baselines.py`

### 3.3 任务 3：局部搜索与奖励诊断

已完成 original 和 nested-development 两条诊断轨道。nested-development 每个 seed 使用 2922 行 search-train 和 975 行 development-validation；当轮行号互斥，并在 search-train 内重新清洗和重建前向贪心。它属于开发证据，不是全研究流程的新独立测试。

关键结果：

- `K=16`：forward greedy 到完整单交换终点的 LR development BAcc 平均增加约 `+0.2760` 个百分点，5 个划分为 4 正、1 平、0 负，达到预注册实用门槛。
- 上述 4 个改善终点都只需 1–2 次交换，但被现有 `pool=4, max_swaps=2` 动作掩码排除。
- `K=32`：DT-CV 平均提高约 `+0.6435` 个百分点，LR development BAcc 平均约 `-0.0226` 个百分点，说明当前评分到最终目标的迁移不可靠。
- 从已经完成单交换优化的终点出发，有界双交换没有形成稳定 LR development 收益。
- 当前动作掩码从每个已检查起点出发，在最多两次交换内只有 117 个可达子集，其中包含原起点。
- 低 MI 条件贡献案例是候选覆盖机制证据，不是已证明的物理交互，也不能作为独立统计样本。

解释边界：

- `K=16` 的结果证明候选掩码存在硬覆盖缺口，没有证明 PPO 或 RL 已经有效。
- “强基线附近没有稳定剩余收益”仅适用于当前 DT-scored 路径，不能外推到按 LR BAcc 重新建立的搜索空间。
- 局部候选排序相关较弱，只能说明强起点附近的精细排序不可靠，不能说明评分器对所有子集质量层次都无信息。

主要材料：

- `documents/research-plan/03-search-diagnosis.md`
- `documents/research-plan/search-diagnosis-protocol.md`
- `documents/research-plan/search-diagnosis-ablation-addendum.md`
- `experiments/search_diagnosis_v1/`
- `src/run_search_diagnosis.py`
- `src/audit_search_reachability.py`

### 3.4 任务 4A：动作支持修复与同预算回放

任务 4A 已完成，正式结论分为两部分：

- **工程修复 PASS**：`quality_prior_pool=4 + swap_exploration_pool=4` 使先验池外的每个合法删除和加入动作在跨 episode 意义下具有严格正的纳入概率；旧配置默认探索配额为 0，逐位保持旧 mask。
- **搜索效率 NO-GO**：在每个起点 64 个新增唯一子集的冻结回放中，新 4+4 机制没有一致优于旧 mask。K=32 时 `quality_plus_global` 与旧 mask 几乎相同，`uniform_legal` 的描述性均值反而更高；这些回放使用旧 DT 目标，不能据此外推 LR 收益。
- forward 起点仍存在支持缺口，K=32 的完整单交换邻域相对旧 mask 一步支持平均高 `0.21928 pp`，5 个起点中 4 个为正；local 起点在冻结 DT 目标下没有剩余增益。
- 本任务未训练 PPO/DQN，历史 matched-context policy gap 为 NA。候选变得可达不等于策略能找到，也不证明 RL 有效。

因此，4+4 机制可以保留为兼容性的工程选项，但不得被写成已验证的高效主候选机制。后续策略归因实验应使用确定性的完整合法动作支持，使硬 mask 不再成为混杂因素；若保留质量信息，只能作为显式先验对照或 actor 输入，并与零更新臂共享。

主要材料：

- `documents/research-plan/action-support-protocol.md`
- `documents/research-plan/04a-action-support.md`
- `experiments/action_support_v1/`

### 3.5 任务 4B：LR Balanced Accuracy 奖励对齐

任务 4B 已完成，正式状态为 **NO-GO**，不得追溯改写：

- 候选银行唯一选中的诊断评分器是 fold-local `StandardScaler + LogisticRegression` 的 Balanced Accuracy，三次五折聚合为 `mean_minus_0_5_sd`；K=16 和 K=32 的银行可靠性门槛均通过。
- K=16 从空集重跑 forward 再做最多两轮完整 best-improvement swap 后，outer LR BAcc 平均变化为 `-0.0425 pp`，正/平/负为 `0/4/1`，未通过搜索收益门槛。
- K=32 对应平均变化为 `+0.4149 pp`，正/平/负为 `5/0/0`，单独通过银行、搜索收益和成本门槛。
- 预注册要求 K=16 与 K=32 同时通过，因此整体决定仍是 NO-GO，`unique_main_reward=null`，原任务 4C 的准入为 false。

K=32 的结果曾形成一个看过 4B 结果后提出的新假设，随后在任务 4C 做了锁定的 development robustness check。4C 已完成且为 NO-GO；4B 的整体结论没有改变。

主要材料：

- `documents/research-plan/reward-alignment-protocol.md`
- `documents/research-plan/04b-reward-alignment.md`
- `experiments/reward_alignment_v1/`

### 3.6 仓库验证：4C-0 已修复

任务 4C-0 已完成并 PASS。原有 11 个 collection errors 由兼容层、stage2 wrappers 及数据 loader/splitter 契约修复解决，修复后 174/174 测试通过。4C 交付的 `verification.json` 记录全量 178 executed / 178 passed / 0 failed / 0 skipped / 0 collection errors。

以上是已保存的验证记录，本次路线更新没有重新运行 pytest。不能继续将“仓库现有 11 个 collection errors”写成当前状态。详见 `04c0-repository-test-repair.md` 和 `experiments/k32_reward_robustness_v1/verification.json`。

### 3.7 任务 4C：K=32 锁定复核 NO-GO

- forward 到最多两轮完整 swap：平均 +0.1009 pp，pooled OOF +0.1010 pp，3 正、2 平、0 负。
- 没有达到平均 +0.2 pp、至少 4/5 为正的预先冻结门槛；成本、隔离、坐标、逐步 argmax 和数值复算全部通过。
- fold 3/4 在完整第一轮邻域检查后无严格 J 改善；fold 0/1/2 都接受两次交换后达到预算上限。不能从此推出全空间没有更优子集，也不能指责策略没有找到动作——本实验没有策略训练。
- All Features 为 91.8579%，LR-forward 为 91.5743%，LR-forward+swap 为 91.6751%（均为五折平均 LR BAcc）。32 维相对 65 维减少 50.77% 的特征数量，平均性能低 0.1828 pp，最差配对 fold 差为 -1.8350 pp。这只是开发描述，不是性能等价、非劣或计算成本降低的证据。
- `frozen_development_scorer=null`，`task_4d_admission=false`；4B 仍为整体 NO-GO。
- 本轮重新从保存 CSV 计算平均增益和正/平/负，与 `final-decision.json` 一致；没有重跑搜索、模型拟合或读取 source-test。

输入证据：`04c-k32-robustness.md`、`04c-k32-robustness-protocol.md`、`experiments/k32_reward_robustness_v1/analysis/{fold_results.csv,pooled_results.csv,final-decision.json}`、`audit.json`。

## 4. 全局实验约束

后续所有任务必须遵守：

1. 数据版本固定为 `v16n_2x_noise`，除非另建版本化协议。
2. source-test 已被历史研究反复查看，不能称为新独立 test，也不能用于选择方法、K、奖励、候选池、checkpoint 或超参数。
3. 新方法决策只能使用 source-train 内部数据；外层开发验证一经用于 go/no-go，就进入模型选择流程。
4. 清洗规则只能在每个训练区拟合，再应用到该轮验证区。
5. 所有结果同时保存 clean 0-based 与 original 1-based 特征坐标并交叉校验。
6. 随机 feature ID 不得进入节点特征、候选排序、shaping、reward 或 archive。
7. 比较搜索算法时共享起点、候选支持、评分器和增量评分预算。
8. 同时报告唯一评分子集数、分类器拟合次数、cache 命中、运行时间及包含初始化/预训练的总成本。
9. seed 的离散程度不是独立数据集抽样不确定性，不得据此夸大统计显著性。
10. 每个新实验必须先写协议和配置 hash，再运行；不得看过开发验证后回写规则。
11. 奖励、archive 和停止条件在机制主实验中应使用同一目标；额外相关性惩罚另作消融。
12. 只运行与任务结论必要的实验，不扩大无边界超参数扫描。
13. 任务 4C、4D、5A、5B 若复用同一 outer folds，它们共同构成累计模型选择链；后续任务不得把这些 folds 重新称为新验证或独立确认。

## 5. 4C NO-GO 后的当前执行状态

| 任务 | 当前状态 | 后续动作 |
|:---|:---|:---|
| 4C-0 | 已完成 PASS | 保留验证记录，不重复修复 |
| 4A / 4B / 4C | 已完成；4B、4C 均 NO-GO | 保留协议与结果，不改门槛、不换 splits 继续复试 |
| 4D | 未准入 | 不训练 PPO，不伪造算法 NO-GO 报告 |
| 5A / 5B | 未准入 | 不生成监督标签或训练预训练策略 |
| 原任务 6 | 本 K=32 分支 blocked | 不能把缺失的 4D/5A 报告当作 NO-GO 来选静态 winner |
| 3A | 尚未完成；当前下一步 | 审计真实 scene/group/env 谱系 |
| 7 | 尚未准入 | 等 3A 确认数据、环境可用性后再评估 |

现在只推进 3A 与已有证据整理。3A 发现可靠 group 也不会自动重开旧 4C；任何新问题都需要新的协议、明确的目标和验证边界。

若研究目标改为“在可接受性能损失下减少特征计算成本”，应先证明各特征确有可减少的计算成本，并在新的评价数据开放前确定可接受损失。当前 32 维结果只提供提出该假设的理由，不能把失败的精度增益实验改名为压缩成功。

若希望继续研究 RL，应先建立实际需要逐步决策的问题，例如按已观测特征决定下一项并学习停止，或可靠环境条件下的预算分配。任务 7 首先检查这种需求与简单基线；当前结果没有授权增加 PPO 网络、训练轮数或继续扫描 K/reward。

## 6. 任务定义：当前 3A、条件任务 7 与历史设计

**本节保留旧设计用于追溯。4C-0/4C 已完成；4D/5A/5B/6 因 4C NO-GO 而关闭，不能把旧的条件说明当作当前执行许可。**

### 任务 3A：v16n_2x_noise 数据谱系与场景分组审计

该任务尚未完成，是当前下一步。它决定是否能支持真实场景分组与环境研究，不承担重新挑选划分使旧 4C 过关的任务。

要求：

- 追查 train/test 文件的生成入口、上游输入、增强、shuffle、split、父场景与环境参数。
- 不得把旧 v16n 的 `696×4` 结论直接用于 v16n_2x_noise。
- 不得按行序、相似度或聚类伪造真实 group。
- 能恢复可靠 group 时，使用新的诊断协议和目录，最多做一项固定 All Features + LR 的 row/group split 敏感性比较，不重新运行特征搜索或解禁 4D/5A。不得读取历史 source-test；其谱系仅查生成代码与既有 manifest。
- 无法恢复时，列出最小补充材料并维持行级开发结论边界。
- 输出 `documents/research-plan/03a-data-lineage-audit.md`。

### 已完成设计：任务 4C-0 仓库兼容边界与全量测试修复

这是工程可信度门槛，不做科研调参、不运行正式 PPO、不改写 4A/4B 的报告、manifest 或结果 hash。

要求：

- 用 git history 和现行 runtime 判定每个缺口是误删兼容层、未迁移 active 调用，还是已经退役的 API。
- 处理 `methods.configure/advice`、`radar_ship_fs.legacy`、top-level stage2 wrappers 及其测试之间的悬空引用；特别检查 `src/run_irfs.py` 与 `src/harness/aggregate.py`。
- 不得用全局 skip、批量 xfail、删除仍代表现行契约的测试或降低断言获得绿色结果。
- `pytest --collect-only -q` 必须为 0 collection errors；全量 `pytest -q` 必须真正执行到结束，所有现行 active tests 必须通过。
- 只有证明不覆盖当前 v16n、scorer、PPO、harness 或 invariants 路径的历史数据依赖测试，才可进入单独命名并明确计数的 `legacy-data` profile；其他失败均为 NO-GO。
- 4A、4B、PPO、harness 和 invariants 相关测试必须实际执行；修改文件通过 Ruff。
- 输出 `documents/research-plan/04c0-repository-test-repair.md`，逐项记录根因、兼容决策、收集数、执行数、通过/失败数和剩余限制。

### 已完成设计：任务 4C K=32 锁定稳健性复核（NO-GO）

这是根据 4B 的 K=32 结果新提出的 post-selection development robustness check。原 4B 仍是整体 NO-GO。

固定范围：

- 只保留 K=32；K=16 路线关闭。
- 唯一评分器固定为 fold-local `StandardScaler + LogisticRegression(C=1.0, solver=liblinear, class_weight=balanced, max_iter=5000)` 的 Balanced Accuracy，3×5 折聚合为 `mean_minus_0_5_sd`。
- 不再比较 K、分类器、指标、fold/repeat 数、聚合器、惩罚权重或 LR 超参数。
- 每个 outer-train 从空集重跑 exact-K exhaustive forward，再运行最多两轮完整 best-improvement single-swap。
- source-test 禁止读取；outer validation 不能进入评分、搜索、停止或任何方法选择。

划分：

- 在查看 4C 结果前冻结一个新的、五个 validation fold 互不重叠的 outer partition。
- 若 3A 已恢复可信 scene/group，使用 `StratifiedGroupKFold` 并保证同源增强不跨 fold；否则使用固定随机状态的 `StratifiedKFold`，并明确该结果仍是同一 source-train 上的事后开发复核。
- 清洗只在各 outer-train 拟合；评分 scaler 只在各 inner 训练折拟合。

新扩展协议必须在读取任何 4C 结果前冻结；数值门槛沿用 4B 原先的 K=32 分项门槛，但这不构成对原 4B 预注册的修改：至少 4/5 outer folds 为正，平均 LR BAcc 增益至少 `+0.2 pp`，每折不超过 4000 个唯一评分子集和 60000 次分类器折拟合，并通过数据隔离、坐标、argmax、fold-local scaler 和独立复算审计。

同时固定报告 All Features、MI-32、LR-forward、LR-forward+最多两轮 swap、任务 2 的可比 K=32 静态基线。除 All Features 外，所有基线子集都必须在每个新的 outer-train 内重新生成；不得把任务 2 的旧子集或旧 outer 数值直接搬入新比较。DFS 只有满足既有 provenance 门槛才进入；否则写 `unavailable / protocol provenance incomplete`，不得从 oracle 列表补数。

通过只冻结一个“K=32 机制实验用 development scorer”，不改变 4B 的 `unique_main_reward=null`，也不构成外部泛化证据。输出 `documents/research-plan/04c-k32-robustness.md`。

### 已关闭设计：任务 4D 全合法动作空间下的 K=32 策略更新贡献消融

准入检查：4C-0 必须通过，4C 必须为 `GO-development`。否则只输出 blocked/no-go，不训练 PPO。

主实验固定：

- 每个 outer fold 使用 4C 的 LR-forward K=32 endpoint 作为共同起点。
- 主动作支持为确定性的 `full_legal`：删除阶段全部 32 个已选特征可见，加入阶段全部 33 个未选特征可见。不得再用候选 pool 做硬截断。
- 质量/MI 信息可以作为预先冻结的 actor 输入或 logit prior，但必须在正常 PPO 与零更新 PPO 间完全共享；4A 的 4+4 机制只作工程背景，不参与主结果择优。
- episode 始终从 exact-K 状态开始。删除半步 reward 固定为 0 且不调用 J；完成一次“删除后加入”的合法完整交换、重新得到 exact-K 子集后，才调用一次 J，并给出 `100 × [J(S_new)-J(S_previous)]`。cache hit 仍算一次 evaluator 请求；stop action 不重复调用 J，reward=0。这样 `gamma=1` 时两次完整 swap 的累计奖励严格等于终点相对起点的 J 增益。
- `gamma=1`；correlation、sparsity、feature-ID 和 proxy shaping 全部关闭。reward、archive、stop 和 checkpoint 选择只在完整 exact-K 状态上使用同一个 J；outer validation 不回流搜索。
- 每个 episode 从固定 forward 起点开始，最多两次交换。

比较四个主方法：

1. full-legal 均匀随机交换；
2. 两轮 sampled best-improvement，每轮无放回评分 128 个完整单交换邻居；
3. 完整 PPO 流程但 learning rate 为 0；
4. 正常更新 PPO。

另用确定性集成测试验证“不调用 update”与 learning rate=0 在参数和动作输出上的等价边界；若不等价，必须解释副作用并将 no-update 加入主对照。

4C 的两轮 exhaustive best-improvement 只作高成本参考，不作为同预算对手。

每个方法、outer fold、algorithm seed 的主搜索预算统一为最多 256 次完整交换后的 J evaluator 请求，cache hit 与重复子集同样消耗请求预算；在请求数 32/64/128/256 处报告曲线。提前停止后的曲线保持最后值。唯一子集数、cache hit 和实际模型拟合数另报。训练结束后的策略评价使用各策略臂完全相同、预先冻结的独立请求配额，计入总物理成本且不得回流 archive。每个 outer fold 使用 5 个配对 algorithm seeds，先在 fold 内聚合 seeds，再以 5 个 outer folds 为分析单位。

必须报告 best-so-far J 曲线及 AUC；AUC 固定定义为 evaluator 请求 1–256 上 `100 × (best_J(b)-J_start)` 的算术平均，提前停止后的曲线保持最后值。另报告到达 4C exhaustive 参考增益所需预算、冻结 inner-best endpoint 的 outer LR BAcc、当前策略随机与确定性 rollout、参数 L2 变化、固定 probe states 的动作分布 KL、entropy、approximate KL、clip fraction、真实梯度范数、动作排序变化、重复/cache/request/fits/wall time。不得沿用占位梯度值。

GO 必须同时满足：

1. PPO 参数和固定 probe 动作分布发生有限、非零且可重复的变化；
2. 以 `1e-6 pp` 为 tie epsilon，PPO AUC 相对零学习率 PPO 和 sampled best-improvement 都至少 4/5 outer folds 为正，且相对两者中更强者的五折平均 AUC 优势至少 `0.05 pp`；
3. PPO endpoint 相对 forward 起点的 outer LR BAcc 至少 4/5 为正，平均至少 `+0.2 pp`；
4. PPO 相对零学习率及最强简单对照的 outer LR BAcc 均至少 4/5 为正，平均优势至少 `+0.1 pp`。

任一项失败都不得声称策略更新有效，并停止 5B；可继续 5A 作为非 RL 路线。输出 `documents/research-plan/04d-k32-policy-learning-ablation.md`。

### 已关闭设计：任务 5A K=32 监督动作排序的可迁移性

准入检查：4C 必须通过，4D 必须已经执行完成并产出 GO 或 NO-GO 报告。4D 失败时，本任务明确属于非 RL 替代路线。

要求：

- 标签固定为 `(state subset, remove, add, ΔJ, paired-delta SD)`，其中不确定性是三次配对 repeat 的 `ΔJ` sample SD，不是候选绝对 J 的 SD；J 与 4C 完全一致，只能在相应 outer-train 内生成。
- 不得随机拆分高度重叠的 swap 记录。一个起点、轨迹及其全部邻居必须是不可拆分 group；至少做 leave-one-context-out。若 3A 恢复真实 scene group，再增加 group-disjoint 轨道。
- 先只研究动作排序，不训练 PPO。比较随机排序、固定 MI/质量先验、轻量线性或 pairwise ranker，以及一个预先冻结的小型监督模型；禁止大规模超参数扫描。
- 报告 Spearman、NDCG@32、top-32 正增益动作召回/富集，及 32/64/128/256 次 J evaluator 请求预算的真实搜索曲线；cache hit 同样消耗请求预算，唯一子集与实际拟合数另报。
- 标签生成、训练、推理、在线评分成本全部报告，并分别给出复用已有 cache 与从零生成的成本。

进入 5B 的门槛：held-out context 的 top-32 富集至少 4/5 大于 1 且平均至少 2×；若某 context 没有任何正 `ΔJ` 动作，该 context 直接记为门槛失败，不得剔除；以 `1e-6 pp` 为 tie epsilon，256 次 J 请求预算的 best-so-far AUC 在至少 4/5 contexts 超过 4D 已冻结的 sampled best-improvement，且五折平均至少领先 `0.05 pp`；冻结 endpoint 的 outer LR BAcc 相对最强简单对照至少 4/5 为正、平均至少 `+0.1 pp`。随机行拆分有效但 context/trajectory 分组后失效，应判 NO-GO。

输出 `documents/research-plan/05a-supervised-action-ranking.md`。若 4D 失败而 5A 成功，成果归为监督搜索，不包装成 RL。

### 已关闭设计：任务 5B 监督初始化后的 PPO 微调增量

准入检查：4D 与 5A 必须同时明确 GO；否则只输出 no-go，不训练复杂模型。

固定 K=32，并继承 4D 的 full-legal support、scorer、起点、两次交换、outer folds、256 次在线 J evaluator 请求预算及成本口径；cache hit 同样消耗请求预算。比较：

1. sampled best-improvement；
2. 冻结监督策略；
3. 从头 PPO；
4. 监督预训练策略、learning rate 为 0；
5. 监督预训练策略 + PPO 微调。

各策略臂的 actor 结构、输入张量、动作头和参数量必须完全一致，只有初始化来源不同；若监督 ranker 不能直接映射到 actor，必须在查看结果前冻结转换 adapter，并加入容量匹配对照。标签生成与预训练成本计入端到端成本。以 `1e-6 pp` 为 tie epsilon，微调 PPO 必须相对冻结监督策略在 best-so-far AUC 和 outer LR BAcc 上均至少 4/5 contexts 为正；五折平均 AUC 至少领先 `0.05 pp`，outer LR BAcc 至少领先 `+0.1 pp`，且参数/动作分布确实变化。否则把贡献归于监督排序并停止静态 RL。

输出 `documents/research-plan/05b-pretrained-ppo.md`。

### 已关闭设计：任务 6 当前 K=32 分支的最终锁定评价

准入检查：4C 必须明确 `GO-development`，且 4D、5A 都必须存在明确的 GO/NO-GO 报告；缺失报告不能按 NO-GO 处理。4C 失败时，本 K=32 分支的任务 6 为 blocked。

在读取任何 final 标签前，必须仅根据 4C/4D/5A/5B 的冻结报告生成并 hash 锁定 `experiments/final_method_selection_v1/final-method-selection.json`。唯一主方法按以下确定性规则选择：

1. 5B GO：选择 pretrained+PPO；
2. 5B 明确 NO-GO 或因 4D 未通过而未准入、且 5A GO：选择冻结监督排序策略；
3. 5A 明确 NO-GO 且 4D GO：选择 scratch PPO；
4. 仅当 4C GO 且 4D、5A 均已有明确 NO-GO 报告：选择 4C 的 LR-forward+最多两轮 exhaustive swap。

final 表可以同时评价多个已冻结基线，但不得根据 final 结果更换主方法、K、checkpoint 或超参数。

最终一次性比较：

- All Features、MI-32；
- LR-forward、LR-forward+两轮 exhaustive swap；
- 256 次 J evaluator 请求预算的 sampled best-improvement；
- 任务 2 中可复现且 classifier-aligned 的强静态基线；
- 通过门槛的 4D/5A/5B 最终胜者；
- 只有训练代码、来源/许可证、配置、seed、checkpoint、train-only selection trajectory 和坐标全部可追溯时才加入 DFS。

主指标固定为 LR Balanced Accuracy；同时报告特征数、选择稳定性、最差组/场景表现、唯一评分数、模型拟合数和完整墙钟成本。所有选择只用 final-train；final evaluation 只打开一次且不再调参。

现有 source-test 已被反复查看，只能作为历史诊断。最终评价需要方法冻结后取得的新仿真场景、新生成批次、真实数据或可证明此前完全未参与选择的外部数据。若生成新仿真批次，必须在生成前冻结 generator 版本/hash、物理参数分布、scene seeds、样本量和排除规则，只生成一个正式批次而不择优。相同生成器的新批次只能称 prospective in-distribution replication；真实域泛化仍需要真实数据或不同来源数据。没有新数据时，输出开发性结论和明确的数据获取计划，不制造“独立 test”。

输出 `documents/research-plan/06-final-locked-evaluation.md`。

### 任务 7：环境条件化特征选择的前提验证

只有 3A 恢复可靠的 base-scene/group/env 元数据、同源增强可保持同 fold、环境变量在推理时可获得，才允许执行；否则只输出 blocked 及最小数据补充需求。

第一阶段只验证“不同环境是否确实需要不同子集”，不直接训练复杂 RL。该分支不依赖 4C 是否通过；直接冻结 K=32 和 4B 规格的 fold-local StandardScaler+LR Balanced Accuracy、3×5 折 `mean_minus_0_5_sd` scorer。预先冻结物理环境表示、scene-group-disjoint outer split、总 J 请求预算和门槛；不得根据结果调整环境分箱。

实验协议必须在查看性能前要求每个环境至少有 25 个独立 base scenes/类别，使五折中每个 validation fold 至少有 5 个 scenes/类别；任一 group fold 无法同时覆盖类别与环境即 NO-GO。比较全局静态 K=32、训练内部的分环境静态子集、简单监督条件化选择器，并在存在真实序贯决策时加入 contextual bandit。主比较使用完全相同的总 J 请求预算；等环境预算只作次级结果。对训练未见环境的回退固定为 outer-train 内得到的全局 K=32，不能结果后更换。相对全局子集，条件化方法必须平均提高至少 `0.2 pp`、至少 80% outer folds 为正、最差环境不恶化超过 `0.2 pp`，且等总预算后仍保留收益，才允许提出条件化策略。

若环境已知且只需一次选子集，优先使用监督条件化或 contextual bandit。只有存在按已观测特征逐步决定下一特征、计算成本不同或需要学习停止时机等真实序贯结构，才开展多步 RL。

输出 `documents/research-plan/07-environment-conditioned-feasibility.md`。

## 7. 更新后的停止规则

出现以下任一情况，应按对应分支停止：

- 4C 的 K=32 锁定复核失败：停止当前 reward 下的静态 PPO、监督标签扩展和 K 扫描。
- 4D 未稳定优于零学习率 PPO 与简单局部搜索：停止把参数更新作为贡献；5A 可作为非 RL 方法继续。
- 5A 在 context/trajectory 分组后失效：不做监督预训练。
- 5B 不优于冻结监督策略：贡献归于监督排序，停止 PPO 微调。
- 3A 无法恢复真实 group/env，或任务 7 未证明环境异质性收益：停止环境条件化路线。
- 缺少新增独立数据：不做最终泛化主张。
- DFS provenance 仍缺失：正式表写 unavailable，不补写估计成绩。

无论 RL 分支是否通过，任务 4A 的支持诊断、4B 的奖励错配诊断、K=32 对齐静态搜索、特征稳定性和物理解释都可以形成科研贡献。

## 8. 通用交付要求

每个任务都应：

- 先检查适用 `AGENTS.md` 和当前 git status，保留用户已有修改。
- 使用新协议版本、配置和结果目录，不覆盖历史实验。
- 在查看正式结果前冻结假设、划分、方法、预算、门槛与配置 hash。
- 保存代码/data/config hash、实际随机状态、fold/outer 行号和环境信息。
- 同时保存 clean 0-based 与 original 1-based 特征坐标并交叉校验。
- 历史 source-test 在所有新任务中的读取次数必须始终为 0，包括任务 6；任务 6 只能读取另行取得并登记的新 final 数据。
- 报告逻辑唯一评分数、实际模型拟合数、请求/cache hit、初始化/标签/预训练成本和墙钟时间。
- 用 Ruff、定向 pytest 与适用的全量 pytest 验证，并执行独立数值复算。
- pytest 报告必须分别写明命令是否调用、收集数、实际执行数、passed/failed/skipped 和 collection errors。若停在 collection 阶段，必须写“全量套件未执行完成”，不得暗示全量通过。
- 报告完成内容、验证证据、可信边界、go/no-go 与下一任务准入状态。
