# 局部搜索空间与奖励可靠性诊断预注册

版本 `search-diagnosis-v1`，2026-09-13；先写入本文件和程序 hash，再执行评分。
继承 research-baseline-v1 与 unified-strong-baselines-v1 的数据、K={8,16,32}、seed=42..46、五折 DT accuracy、严格阈值 1e-12、最多32轮单交换、固定 LR/DT 分类器和 J=accuracy−0.02×mean_abs_correlation。新实验不改写既有协议或产物，不训练 RL，不打开 source-test（包括不为 hash 而读取它）。

## 数据隔离与起点

1. original 轨道：逐 seed 复现旧 context，从已有 forward_greedy_single_swap 固定 K 子集开始，枚举完整单交换并复算起点一致性。此轨道已使用全部 source-train，不能用其重新划分出来的 holdout 声称独立验证；只作旧搜索空间复核。
2. nested_dev 轨道：每个 seed 以 random_state=10000+seed 对 source-train 做分层 75% search-train / 25% development-validation 划分。清洗（常量及精确重复列）只拟合 search-train，并保存全局原始行号及 original 1-based 特征映射。对 search-train 再调用冻结 build_seed_context，复现 RNG 消耗、行重排和五折评分。重建到 K=32 的完整 forward greedy 路径，保存 K=8/16/32 起点，再按冻结 best-improvement 全邻域规则爬到单交换局部最优。起点及每个完整单交换邻域均归档。
3. 全部 seed 的候选搜索完成且 search-complete manifest 写入后，才运行开发验证阶段；该阶段只拟合 search-train，验证全部已冻结邻域候选（不以搜索分数筛掉候选），并评价起点和终点。验证分数不得反馈本轮搜索。初始化加法候选保存搜索分数与成本，不纳入交换排序分析。
4. development-validation 从用于本次 A/B/C/D 决策起即属于模型选择流程，不是最终独立 test。重复划分有样本重叠；跨 seed 稳定性是描述性复现，不提供独立样本显著性检验。
5. DFS 只在满足原协议 provenance 门槛时加入；既有 unavailable 状态不允许使用 oracle 列表补充。

## 有界两次交换

只有完整单交换邻域无严格改善时启用。以该邻域分数最高的前4个候选作为桥（允许下降/平局，稳定枚举顺序破同分）。每个桥通过独立固定 RNG(seed×100+K)，从尚未改变的起点特征中删除1个、从原起点外且未加入的特征中加入1个，均匀无放回抽取最多128种第二交换；最终相对起点恰好2出2进。每起点至多512次第二步请求，去重缓存，绝不因开发验证结果扩大预算。记录桥分数、损失、两条路径次序及覆盖率。该预算不能证明全局双交换最优，也不能排除更深路径。

## 指标与预先指定决策

- 搜索主分数 DT五折 accuracy；同时记录 J 和冗余，但只用 accuracy 搜索，不调 J 的权重。
- 开发验证主指标固定 StandardScaler+LR(C=1,liblinear,max_iter=5000,class_weight=balanced,random_state=seed) 的 balanced accuracy；次指标固定 DT(random_state=seed) 的 accuracy 与 balanced accuracy，分别报告。
- 每起点分别计算全部单交换候选的 Spearman、Kendall tau-b；三分类增益符号一致性（±1e-12）；搜索正增益候选验证为正比例；搜索 top10% 对验证正增益 top10%（最多ceil(0.1N)）的召回。验证无正增益时召回记 NA，不能记100%。J排序同样报告。
- 改进规则预先固定：best_search 邻居只有严格超过起点才移动；best_J 规则只作冻结奖励敏感性诊断；forward→单交换终点、局部最优→有界双交换终点跨5划分配对。
- 实用稳定收益门槛：同K、同规则 LR开发验证平均增益≥0.002，且至少4/5划分为正；同时披露DT结果与所有负例。0.002是本诊断的预注册实用阈值，不是统计显著性。
- A：稳定收益存在且 PPO 掩码排除有效交换，支持改搜索；掩码可达不等于已有策略能找到，未训练新策略不能证明优化失败。
- B：搜索平均改善，但主验证未达到稳定收益门槛，且排序/符号一致性弱，优先奖励可靠性/分类器对齐诊断，暂停扩大RL搜索。
- C：完整一步无收益而有界双交换达到稳定验证收益门槛，才支持优先多步搜索。
- D：强基线附近上述稳定收益均未成立，尤其搜索本身也缺少明显改善；有限预算不证明无更深收益。
- 给出证据最强类别及次要现象，不把“多训练”作为默认下一步。

## PPO覆盖与产物

直接使用现有 build_feature_graph 和 FeatureSelectionEnv 掩码，feature ID项关闭、pool=4、max_swaps=2。质量为 max-scaled |corr(Xj,y)| 与 MI 的平均值；删除最低4项、加入最高4项，并排除刚删除项。报告每个有效交换能否进入删除/加入池，以及任意两种执行次序能否完成双交换。低MI定义为按MI降序排名>ceil(d/2)，不把它等同于已证明生物/物理交互；联合贡献只表示条件于当前子集的交换增益。K=8/16为同规则matched-K反事实，实际冻结PPO主任务K=32。

新产物目录 experiments/search_diagnosis_v1/。保存协议/代码/data hash、环境、所有split/fold行号、train-only清洗、双坐标、候选来源、桥、accuracy/J/fold scores、相对起点增益、候选逐项实际拟合数与耗时、cache事件、验证分数、汇总和审计。不覆盖历史实验。禁止用最终test选择搜索/奖励参数。
