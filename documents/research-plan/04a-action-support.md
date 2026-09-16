# 任务 4A：旧动作掩码上界、候选支持修复与同预算回放

版本：`action-support-v1`  
完成日期：2026-09-13  
数据范围：仅 `v16n_2x_noise` source-train 与任务 3 冻结的 nested-development 上下文

## 决策摘要

**候选支持修复：PASS。** 生产 PPO environment 现在可以在保持原质量先验池的同时，从先验池外均匀抽取候选；任一合法删除和加入动作在 episode 间都具有严格正的纳入概率。默认探索配额为 0，旧配置的 action mask 逐位不变。

**作为固定预算搜索效率改进：未通过。** 在每个起点恰好 64 个新增唯一子集的冻结回放中，`quality_plus_global` 没有一致优于旧掩码：K=8、16 的 forward 平均终值低于旧掩码，K=32 与旧掩码几乎相同且低于均匀合法交换。该机制应作为消除结构性零支持的工程修复保留，不应宣称为已经验证的搜索性能提升。

**RL 结论：NA。** 本任务没有训练 PPO 或 DQN；任务 3 也没有在这 30 个 matched-context 起点上运行历史 PPO。因此历史 `search-policy gap` 无法识别，不能用别的 K、切分或起点替代。“现在可达”不等于“策略能找到”，本次无学习回放也不证明 RL 有效。

任务 4C 仍然不能据此启动：需要任务 4B 先给出通过审计的对齐奖励；即使 4B 通过，4C 仍须单独检验学习是否发生以及是否优于相同预算基线。

## 冻结设计

正式配置为 `configs/v16n/action_support_v1.toml`，协议为 `documents/research-plan/action-support-protocol.md`。实验固定：

- seed 42–46，K={8,16,32}，每个 `seed×K` 的 forward/local 两个任务 3 冻结起点，共 30 个起点；
- search score 为完全继承任务 3 的五折 DecisionTree accuracy；
- 旧机制：`quality_prior_pool=4`、`global_exploration_pool=0`、`max_swaps=2`；
- 新机制：质量先验 4 + 先验池外均匀无放回探索 4，状态内缓存、每次 reset 确定性重抽；
- 回放方法：`old_mask`、`quality_plus_global`、`uniform_legal`；
- 每个 `seed×K×phase×method` 使用 5 个冻结 replay seed，共 450 次回放；
- 每次回放恰好评分 64 个不含起点的新增唯一子集；重复请求和返回起点只计 cache hit，不花评分预算；
- 交换深度 1/2 各以 0.5 概率抽取，候选 mask 内均匀选动作；
- 未读取 source-test，未训练或更新 PPO/DQN，未用结果调配额、预算或 replay seed。

所有子集同时保存 clean 0-based 与 original 1-based 坐标，audit 对全部 3,510 个旧可达点完成坐标交叉检查。

## 旧掩码可达全集

对每个冻结起点直接调用生产 `FeatureSelectionEnv` 的真实 action mask，并穷举 0、1、2 次完整交换后的唯一子集。每个起点恰好得到：

| 最短交换距离 | 唯一子集数 |
|---:|---:|
| 0 | 1 |
| 1 | 16 |
| 2 | 100 |
| 合计 | 117 |

30 个起点总计 3,510 条带评分、最短距离和双坐标的可达记录。任务 3 的完整单交换邻域（含起点）分别有 K=8: 457、K=16: 785、K=32: 1,057 个点，因此旧掩码的一步 17 点支持只是完整合法单交换邻域的严格子集。

为避免混淆搜索半径，本报告分开定义：

- `action_support_gap_1swap`：完整单交换邻域（含起点）最优分数减旧掩码一步直接支持（含起点）最优分数；
- `old_mask_ceiling_2swap`：旧掩码在 0–2 次交换内 117 个唯一点的最高分；
- `search_policy_gap_at_64`：旧两交换上界减旧掩码 64 分回放实际最好分数。

## Action-support gap 与旧掩码上界

下表的增益和缺口均为**百分点（pp）**；SD 只是 5 个重叠 outer split 的描述性离散度。

| K | phase | 单交换支持缺口 mean ± SD | 最大缺口 | 正缺口起点 | 旧掩码两交换上界相对起点 mean / max |
|---:|:---|---:|---:|---:|---:|
| 8 | forward | 0.00692 ± 0.01544 | 0.03454 | 2/5 | 0.10935 / 0.37613 |
| 8 | local | 0 | 0 | 0/5 | 0 / 0 |
| 16 | forward | 0.10256 ± 0.08738 | 0.20554 | 4/5 | 0.05476 / 0.23949 |
| 16 | local | 0 | 0 | 0/5 | 0 / 0 |
| 32 | forward | 0.21928 ± 0.12493 | 0.30828 | 4/5 | 0.07524 / 0.23943 |
| 32 | local | 0 | 0 | 0/5 | 0 / 0 |

这分解出两个不同事实：forward 起点，尤其 K=16/32，存在旧掩码根本不给直接支持的更优单交换；而 local 起点在冻结 DT 目标下没有剩余正增益。后者不能解释为候选机制普遍充分，只说明这些特定 local 起点已经达到所检查邻域的冻结目标上界。

## 可复现的 search-policy gap

历史 PPO 的 matched-context gap 固定报告为 `NA_no_matched_historical_policy`。作为本任务的无学习诊断，旧掩码回放在 64 个唯一评分点后与其两交换上界的差为：

| K | phase | gap@64 mean ± SD (pp) | 最大 (pp) | 到达上界的回放 |
|---:|:---|---:|---:|---:|
| 8 | forward | 0.00414 ± 0.01146 | 0.03454 | 22/25 |
| 8 | local | 0 | 0 | 25/25 |
| 16 | forward | 0.02603 ± 0.07281 | 0.23949 | 22/25 |
| 16 | local | 0 | 0 | 25/25 |
| 32 | forward | 0.00820 ± 0.02838 | 0.10251 | 23/25 |
| 32 | local | 0 | 0 | 25/25 |

该量只回答冻结随机提议器在固定预算下是否找到旧掩码上界，不是 PPO 的学习表现。

## 候选支持修复

新字段 `swap_exploration_pool` 已贯通实验配置、PPO 配置、run session 和 environment。设某一步合法动作集合去掉质量先验前 4 个后还剩 R 个动作，探索配额为 q=4：

- 质量先验内动作纳入概率为 1；
- 先验外任一动作纳入概率为 `min(q,R)/R`，R>0 时严格大于 0；
- 合法交换由删除、加入两段动作组成，两段条件概率均为正，故完整交换支持概率也严格为正。

同一 episode 的同一状态使用缓存候选，避免 action mask 在状态未变时漂移；reset 用实验 seed 与 reset 序号确定性重抽，保证可复现且让先验外动作跨 episode 获得支持。`swap_exploration_pool=0` 时退化为原实现。

定向测试覆盖旧掩码逐位兼容、状态内稳定、跨 reset 可复现变化、每个合法交换的正边际支持，以及负探索配额拒绝。

## 同预算回放结果

下表为每个面板 25 个 `seed×replay-seed` 的最终 best-so-far 相对起点增益，单位 pp；三种方法的逻辑评分预算都严格为 64 个唯一子集。

| K | phase | old mask | quality + global | uniform legal |
|---:|:---|---:|---:|---:|
| 8 | forward | 0.10521 ± 0.14461 | 0.06694 ± 0.11603 | 0.01912 ± 0.04631 |
| 16 | forward | 0.02873 ± 0.06600 | 0.01369 ± 0.04310 | 0.00818 ± 0.01783 |
| 32 | forward | 0.06704 ± 0.08913 | 0.06704 ± 0.08637 | 0.09990 ± 0.11517 |
| 8/16/32 | local | 0 | 0 | 0 |

配对终值差进一步显示：

| K forward | 新−旧 mean (pp), W/T/L | 新−均匀 mean (pp), W/T/L |
|---:|---:|---:|
| 8 | -0.03827, 2/17/6 | +0.04782, 7/16/2 |
| 16 | -0.01505, 3/16/6 | +0.00551, 2/18/5 |
| 32 | +0.000005, 7/12/6 | -0.03286, 7/6/12 |

由于 seed 的 outer split 重叠，且 replay seed 共享同一数据，本任务不把这些行当作独立数据集样本，不做显著性推断。可支持的结论只有：扩大支持消除了结构性零概率，但在 64 分预算下没有带来一致的提议效率优势。

![Task 4A 同预算 best-so-far 曲线](../../experiments/action_support_v1/analysis/budget_curves.png)

图中每点为 25 个 `seed×replay-seed` 的平均 best-so-far 增益；所有曲线使用共同的 pp 纵轴。完整逐预算数值保存在 `analysis/budget_curves.csv`。

## 任务 3 覆盖回归

K=16 四条改善终点对应的 6 条接受边，以及 23 个低 MI 但净/条件 DT 与 LR 贡献均为正的案例，仅作覆盖回归。新机制对 33 条记录全部给出严格正支持概率：

- 改善终点：4/4；
- 接受边：6/6；
- 低 MI 条件贡献案例：23/23。

它们没有参与探索配额、预算或 replay seed 的选择，也没有被计作“策略找到”的证据。

## 成本、审计与复现

450 次回放共产生 28,800 个逻辑新增唯一评分点（每次 64）和 41,024 次请求，其中 12,224 次为重复/cache hit。跨方法及旧任务 3 分数复用后，本任务实际新增评分 11,068 个唯一子集、55,340 次 DT 折拟合；另复用 19,693 个任务 3 已冻结分数。总墙钟时间 136.43 秒。逻辑预算不因 cache 复用而改变。

最终 audit 状态为 `passed`：

- manifest 文件 hash 全部通过；
- 30 个起点、每起点 117 个可达点、3,510 条坐标与缺口公式全部复核；
- 450 次回放均恰好 64 个新增唯一评分点，28,800 行预算曲线单调性通过；
- 缺口公式最大数值误差 `2.28e-16`；
- 每个数据 seed 独立重拟合一个抽样子集，共 25 次折拟合，最大评分误差为 0；
- source-test 使用次数 0，RL 训练次数 0。
- Ruff 静态检查通过；覆盖本次生产路径的聚焦回归套件 30/30 通过。
- 仓库全量 pytest 在收集阶段被既有缺失模块（`radar_ship_fs.legacy`、`methods.advice/configure` 和 stage2 入口）阻断；扩展可收集切片另有 73 个测试通过，其余依赖缺失的 `data/sim_ship_cr_v10.*`。因此不把聚焦套件标作全量通过。

主要复现命令：

```bash
PYTHONPATH=src conda run -n dl-lab python src/run_action_support.py --stage run --jobs 24
PYTHONPATH=src conda run -n dl-lab python src/summarize_action_support.py
PYTHONPATH=src conda run -n dl-lab python src/audit_action_support.py
```

主要交付物：

- 冻结配置：`configs/v16n/action_support_v1.toml`
- 冻结协议：`documents/research-plan/action-support-protocol.md`
- 正式 manifest：`experiments/action_support_v1/manifest.json`
- 完成标记：`experiments/action_support_v1/complete.json`
- 旧可达全集：`experiments/action_support_v1/analysis/old_reachable_subsets.csv`
- 起点级缺口：`experiments/action_support_v1/analysis/support_gaps.csv`
- 回放终值与请求成本：`experiments/action_support_v1/analysis/replay_runs.csv`
- 完整预算曲线：`experiments/action_support_v1/analysis/budget_curves.csv`
- 汇总表与图：`experiments/action_support_v1/analysis/`
- 独立审计：`experiments/action_support_v1/audit.json`

## 局限与后续门槛

1. 支持是跨 episode 的边际支持，不保证单个 episode 同时暴露所有动作，也不保证训练策略给这些动作足够概率。
2. 固定 4+4 候选是预注册的诊断机制，不是通过本结果调出的最优配额。
3. 回放优化的是冻结 DT 搜索分数；没有用 LR development 结果选择机制，因而不能外推为 LR 改善。
4. local 起点的零增益是这些冻结起点和该评分函数下的结果，不是一般性的搜索不可改进证明。
5. 若任务 4B 通过，后续 RL 实验必须继续报告候选覆盖、同预算无学习基线、学习曲线和 matched-context policy gap，不能把本任务的 reachability PASS 当作 RL PASS。
