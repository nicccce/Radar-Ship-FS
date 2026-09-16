# 旧动作掩码上界、候选支持修复与同预算回放预注册

版本：`action-support-v1`。日期：2026-09-13。正式结果写入前冻结。

## 范围和隔离

本实验只使用 `v16n_2x_noise` source-train 与任务 3 已冻结的 nested-development 搜索上下文，不读取 source-test，不训练或更新 PPO/DQN。使用 seed 42–46、K={8,16,32}，每个 `seed×K` 固定 forward 与 local 两个起点，共 30 个起点。清洗、outer 行号、五折行号、DT random state、特征坐标和搜索分数定义完全继承 `search-diagnosis-v1`。

## 旧掩码上界和缺口

旧机制固定 `quality_prior_pool=4, global_exploration_pool=0, max_swaps=2`。从每个起点调用生产 `FeatureSelectionEnv` 的真实 action mask，穷举 0、1、2 次完整交换后所有唯一终点，保存最短交换距离和双坐标并评分。

“完整邻域”沿用任务 3 的术语，专指起点的完整单交换邻域。为避免混淆半径，报告三项而不把不同半径强行合并：

1. `action_support_gap_1swap = 完整单交换邻域（含起点）最优分数 - 旧掩码一步直接支持（含起点）最优分数`；
2. `old_mask_ceiling_2swap = 旧掩码在 0–2 次交换内全部可达终点的最优分数`；
3. `search_policy_gap_at_64 = old_mask_ceiling_2swap - 旧掩码同预算回放实际最好分数`。

任务 3 没有在这 30 个 nested-development 起点训练或运行历史 PPO，不能构造 matched-context 的“历史策略实际最好分数”。因此该历史量固定报告为 NA，不能用其他数据切分、K 或起点的旧 PPO 结果代替。`search_policy_gap_at_64` 是本任务可复现的无学习回放量，不冒充历史 PPO 缺口。

## 新候选支持

新机制在每个删除/加入状态保留质量排序前 4 个候选，并从其余合法候选中均匀无放回抽取至多 4 个全局探索候选。探索集合在同一 episode 的同一状态上缓存不变，每次 reset 使用 `ExperimentConfig.seed` 与 reset 序号确定性地产生新抽样。质量先验内动作的掩码纳入概率为 1；其余任一合法动作的纳入概率为 `min(4,R)/R>0`，其中 R 是质量先验外合法动作数。因此任一合法交换的两段条件概率乘积严格为正。旧配置不提供该字段时默认为 0，逐位保持旧掩码。

## 同预算回放

三种无学习提议机制共享每个起点和 DT 五折评分器：

- `old_mask`：旧质量池 4；
- `quality_plus_global`：质量池 4 + 全局探索配额 4；
- `uniform_legal`：所有合法删除和加入等概率。

每个 `seed×K×起点×方法` 使用 replay seed 41001–41005。每个 episode 的交换深度用共享预注册分布在 1 和 2 中等概率抽取；每段动作在当前候选 mask 内均匀抽取。每次运行必须取得恰好 64 个不含起点的新增唯一终点；重复请求与回到起点只计请求/cache hit，不消耗预算。保存预算 1–64 上的 best-so-far。方法逻辑成本固定为 64 个唯一子集、320 次 DT 折拟合；跨方法复用冻结分数或全局 cache 只降低实际重算成本，不改变公平预算。

不使用开发验证分数选择候选机制、配额、预算或 replay seed。预算曲线只描述冻结 DT 搜索目标上的提议效率，不证明 LR development 收益，更不证明 RL 有效。

## 覆盖回归与交付

任务 3 中 K=16 四条改善终点（六条接受边）及 23 个满足净/条件 DT 与 LR 均为正的低 MI 条件贡献交换，只检查新机制的理论支持概率是否严格为正；它们不进入配额选择、回放 seed 选择或成功率判定。

交付包括生产实现、定向测试、冻结配置及 hash、30 起点旧可达全集、缺口表、预算曲线、成本、运行环境、audit 和 `04a-action-support.md`。所有子集同时保存 clean 0-based 与 original 1-based 坐标。seed 离散程度仅作描述，不作为独立数据集抽样不确定性。
