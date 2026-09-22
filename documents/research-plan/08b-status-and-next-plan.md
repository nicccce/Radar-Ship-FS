# 08B：08A 情况总结与下一步探索

日期：2026-09-22。状态：已读取08A实际产物，完成保存结果复算；尚未启动08B训练。继续按用户要求，以创新尝试和识别准确率阳性为先，机制证明与完整确认后移。

## 1. 当前情况：有可追的信号，先加强学习

08A完成12个基础搜索case及4个horizon16修订case。两折内先平均seed、再平均fold，主endpoint结果如下：

| 方法 | 开发 BAcc | 相对 G1 | 相对 All-SVC |
|---|---:|---:|---:|
| All-SVC | 92.6477% | +0.3370 pp | — |
| **Single-PPO** | **92.6271%** | **+0.3164 pp** | **−0.0206 pp** |
| Pair-PPO | 92.5386% | +0.2280 pp | −0.1091 pp |
| G1：完整forward+一轮swap | 92.3107% | — | −0.3370 pp |
| Random-Pair | 92.2303% | −0.0804 pp | −0.4174 pp |
| MI32 | 92.1645% | −0.1462 pp | −0.4832 pp |
| Pair-PPO，horizon16 | 92.1622% | −0.1485 pp | −0.4855 pp |
| All-LR（不同分类器，另作系统参照） | 91.8638% | 不作同模型选择比较 | 不作同模型选择比较 |

**最有价值的进展是：RL搜索原型已出现超过贪心的开发信号，Single-PPO均值接近All-SVC。** 目前应围绕如何提高策略学到的动作质量继续尝试，不能把Pair这个最初设想当作必须保留的答案。

- Pair有一个相对G1 **+1.0454 pp** 的局部阳性，平均也高于Random-Pair **+0.3083 pp**；值得继续研究，不必先证明机制。
- Single平均比Pair高 **0.0884 pp**，最高单case为 **93.8136%**。最高值用于定位机会，不能代替全case均值。
- horizon16平均比基础Pair低 **0.3764 pp**，本次搁置这个修订。它同时减少了episode/重启数和每次更新的批大小，所以只能说“这套延长路径方案更差”，不能证明所有长路径都无效。
- All-SVC平均仍最高；相对All-LR的提升包含分类器变化，不能全归功于RL选择。

依据：[08A报告](/root/feature-select/Radar-Ship-FS/documents/research-plan/08a-block-rewrite-results.md)、[原始结果表](/root/feature-select/Radar-Ship-FS/experiments/block_rewrite_ppo_v1/results.csv)、[本轮复算表](/root/feature-select/Radar-Ship-FS/experiments/block_rewrite_followup_analysis_v1/method_summary.csv)。

## 2. 一个直接影响下一步设计的新观察

我复算了已保存PPO日志中的条件动作熵。K=32、总列数65时，均匀动作的平均head熵为：

- Single：`(ln32+ln33)/2 = 3.481122`。
- Pair：`(ln32+ln31+ln33+ln32)/4 = 3.465492`。

基础Single最后一次更新记录的平均熵为 **3.479798–3.480961**，Pair为 **3.464857–3.465358**；所有这些更新的clip_fraction最大值均为0。梯度与参数更新记录非零。**这提示动作偏好还很弱，值得尝试更直接地学习已发现的改进动作。** 它不是“完全没学习”的证明，也不是最终策略在所有状态上的独立测量。

再从4个基础Single训练case中，按“这条路径后续能刷新此前最好J，且达到MI32水平”提取候选动作监督，每case已有 **150–280条** 合格transition，其中 **43–99条** 的即时动作会降分、但后续能实现改进。这给出很具体的尝试机会：让策略反复学习通向好子集的路径，而不只奖励即时涨分。

以上只读已有日志，没有新分类器拟合。可复算产物：[脚本](/root/feature-select/Radar-Ship-FS/experiments/block_rewrite_followup_analysis_v1/summarize.py)、[熵统计](/root/feature-select/Radar-Ship-FS/experiments/block_rewrite_followup_analysis_v1/policy_entropy.csv)、[回放样本计数](/root/feature-select/Radar-Ship-FS/experiments/block_rewrite_followup_analysis_v1/replay_opportunity.csv)、[输入hash清单](/root/feature-select/Radar-Ship-FS/experiments/block_rewrite_followup_analysis_v1/input-manifest.json)。计数是去重前的transition实例，不是独立任务数。

08A记录78113次classifier fits、搜索/基线累计墙钟2084.3秒，约34.7分钟；不是包含开发过程的总耗时。已有测试记录为15 passed，独立复核了一个正式endpoint，本轮未重跑这些测试。[成本](/root/feature-select/Radar-Ship-FS/experiments/block_rewrite_ppo_v1/costs.json)、[验证记录](/root/feature-select/Radar-Ship-FS/experiments/block_rewrite_ppo_v1/verification.json)。

## 3. 指定下一步：改进轨迹回放辅助 Single-PPO

**唯一主实验：`improvement-replay-ppo-v1`。**

**假设：当前瓶颈之一是好路径只被短暂利用，策略仍接近均匀采样；在Single-PPO后增加对“后续能刷新最好J”的动作序列的加权模仿，能提高选择质量，并争取超过All-SVC。**

使用Single作为骨架，因为它是本轮表现最好的学习臂。保留K32、H8、共同SVC及原重启方式，只增加一个“改进轨迹回放”组件；先不混合1/2交换、换分类器或改K。固定Pair和horizon16暂时退到参照位置。

这个尝试有既有思想基础：[Self-Imitation Learning](https://proceedings.mlr.press/v80/oh18b.html)研究重复利用过去的好决策，[AWR原文](https://arxiv.org/abs/1910.00177)研究优势加权的策略回归。这里是受它们启发的任务适配，不声称复现原算法或发明自模仿；值得探索的具体点是**直接用已计算的CV改进路径构造动作监督，包括必要的暂时降分动作**。先看准确率，再研究能否形成有区分度的方法贡献。

### 3.1 一次更新怎么做

对于episode的状态`S_t`、动作`a_t`、分数`J_t`，定义：

```text
b_t = max(J_0, ..., J_t)                 # 动作前已知最好分数
u_t = max(J_(t+1), ..., J_H)             # 已完成该条训练轨迹的未来最好分数
g_t = u_t - b_t
```

仅保留`g_t>1e-12`且`u_t>=J(MI32)`的transition。它可以包含一次即时降分，只要后续真的实现突破。回放中存原状态、原有序单交换动作、g/u；全部来自该case自己已结束的训练轨迹。

维护最多512条去重记录：优先u高，再g高，最后按确定性键排序。权重`w=clip(100*g,0,2)`。每个8-episode批次先做原PPO更新，再做16个batch64的回放梯度步：

`L_replay = -sum(w * log pi_theta(a|s)) / sum(w)`。

仍用Adam lr=3e-4、梯度裁剪0.5；回放只反传动作log-prob，不加critic损失，不把这些旧动作当新的on-policy PPO样本。使用原优化器，动作head与共享encoder获得梯度；不另加entropy项。回放为空时跳过并记录。当前采集batch用完即丢弃其PPO ratio数据，下一批从更新后的策略重新采样。

不加载08A旧checkpoint/旧轨迹作为训练输入。08A只用于产生假设、复用开发划分与静态参照；新实验从新seed重新积累自己的改进记忆。这样实现最直接，也不需要先做数据集式预训练。

### 3.2 小矩阵与预算

| 臂 | 更新方式 | 回答的问题 |
|---|---|---|
| Plain Single-PPO | 原PPO，重新跑新seed | 准确率提升的主对照 |
| **Replay-PPO** | PPO后加改进轨迹回放 | 唯一主方法 |
| Replay-only | 相同结构/搜索预算，仅做同样回放，不做PPO更新 | 改进是否主要由回放式策略学习提供；本轮不要求完整归因 |

固定fold0/1，三个新算法seed=`[2026092501,2026092502,2026092503]`，共18 cases。新seed只是算法重复，不是新数据。每case仍128训练episodes+32冻结episodes、H8、1440次J请求。三个臂同一fold/seed使用相同模型初始权重与预生成随机起点，各自维护archive和回放。

直接引用08A相同数据/评分配置下的G1、MI32、All-SVC、All-LR作为静态参照，不重跑整套基线。新Plain-PPO必须实际训练，不能把08A不同seed均值当它的新配对结果。

基础预算：18×1440×3=**77760次SVC折拟合上界**。可依据曲线追加一次`replay_steps=64`变体，仅在回放量充分却动作偏好仍很弱时尝试；覆盖同样6个case，从头训练，其余保持不变，再增加≤25920 fits。

**整体上限110000次classifier fits、计算4小时；目标一个工作日完成实现与结果。** 按08A的8worker+GPU实测，预计计算约1–3小时，回放训练会增加开销，必须以本轮smoke修正估计。额外神经网络更新计入时间；不是宣称训练成本相同或效率优势。资源不足优先完成成组三臂，不为补齐审计延误试验。

### 3.3 看什么结果就继续

主看`Replay-PPO−Plain-PPO`的开发outer BAcc，同时列相对All-SVC、G1和Replay-only的差值。+0.30～0.50 pp仍是值得追的量级，不是准入硬门槛；局部明显阳性可继续，但所有seed结果都要保留。

动作熵、回放样本log-prob、内层J与冻结rollout表现帮助选下一处修改，**不能用“熵下降了”替代准确率提升**。不要求先完成因果归因或五折验证才继续探索。

- 准确率提升：先补其余开发fold/更多seed，再考虑将学到的回放机制迁回Pair或加入动作粒度选择，当前不并行开第二主线。
- Replay-only也很好：记录为好轨迹监督有价值，继续研究PPO怎样补充它，不强行为RL独占功劳。
- 记住动作但outer准确率不升：重点转向“固定32维是否限制识别”，停止继续增加回放梯度步。
- 本版及一次有理由修订都无可用信号：唯一备选是同一SVC下的可变大小add/delete/STOP（16–48维），保留已表现较好的识别器，避免同时切回LR造成两处变化。它更新了08-v3的LR备选，理由是本轮SVC已显示更高系统准确率。

## 4. 实施范围与最少交付

复用已有SVCScorer、条件化Single模型、轨迹收集、archive和评价函数。新增回放组件与独立driver/config，旧08A产物保持不变。重点防止把历史动作错误塞进PPO ratio或把新method错误路由到Pair；不用做全仓审计。

保存每case配置/seed、checkpoint、曲线、回放数量/损失、最终子集、BAcc、计算成本及针对改动的测试结果。保持source-test不用，训练回报不看outer，开发结果可指导一次有记录的改动。完整STG、零更新矩阵和新场景确认继续后移。

[下一轮完整执行Prompt](/root/feature-select/Radar-Ship-FS/documents/research-plan/08b-first-experiment-prompt.md)已给出落地范围。下一任务直接做这个试验，不再安排一轮“仅增加重复以证明值得尝试”的前置任务。
