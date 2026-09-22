# 08B：改进轨迹回放辅助 PPO 实验结果

本轮为已看过的 source-train 开发 fold0/1；不是新 test。历史 source-test 未打开。

已完成 18 cases；未完成 []。主 endpoint 按所有已评分子集最高 inner J 选择，冻结 endpoint 仅限冻结32条轨迹；outer 不参与选择。

| 版本/方法 | 主 BAcc | 冻结 BAcc | Accuracy | Recall− | Recall+ |
|---|---:|---:|---:|---:|---:|
| base/plain_single_ppo | 92.4985% | 92.5141% | 92.5427% | 93.7638% | 91.2332% |
| base/replay_only | 92.4994% | 92.5054% | 92.5641% | 94.3416% | 90.6572% |
| base/replay_ppo | 92.9862% | 92.2292% | 93.0342% | 94.3410% | 91.6314% |

先在每fold内平均3 seeds，再平均两个fold。静态参照来自08A相同context/model，文件路径及hash见 static_references.json；未重新拟合基线。

- all_svc: 92.6477%
- g1: 92.3107%
- mi32: 92.1645%
- all_lr: 91.8638%

| 版本 | Replay-PPO 相对 | 平均差 pp |
|---|---|---:|
| base | all_lr | +1.1224 |
| base | all_svc | +0.3385 |
| base | g1 | +0.6755 |
| base | mi32 | +0.8217 |
| base | plain_single_ppo | +0.4877 |
| base | replay_only | +0.4868 |

| 版本 | fold | seed | Plain | Replay-PPO | Replay-only | Replay−Plain pp |
|---|---:|---:|---:|---:|---:|---:|
| base | 0 | 2026092501 | 93.9292% | 93.5741% | 93.8051% | -0.3551 |
| base | 0 | 2026092502 | 93.3002% | 93.8308% | 93.4158% | +0.5305 |
| base | 0 | 2026092503 | 93.4500% | 93.6981% | 93.5655% | +0.2481 |
| base | 1 | 2026092501 | 91.2432% | 91.8343% | 90.9495% | +0.5912 |
| base | 1 | 2026092502 | 91.9765% | 92.7560% | 91.5684% | +0.7794 |
| base | 1 | 2026092503 | 91.0917% | 92.2240% | 91.6921% | +1.1323 |

学习行为、结论及下一处修改见本报告末尾的实测解读。

本轮成本：65328 classifier fits（含smoke、端点评价、独立复核及图依赖树拟合）；25938 J请求；case累计墙钟 1668.0s。

代码复用08A评分、收集与重启；三臂都是Single/K32/H8，torch_seed=seed+100*fold+23。回放仅从本case训练轨迹纳入，PPO之后单独优化；冻结不纳入。Plain也统计候选回放机会但执行0个回放步。原始记录、曲线、子集、checkpoint、RNG、probe、成本及配置位于 experiments/improvement_replay_ppo_v1。

**实测结论。** 16步回放辅助PPO是本轮最有希望的版本。完整6个Replay-PPO case的主开发BAcc为 **92.9862%**，比新Plain提高 **+0.4877 pp**；5/6个配对为正，范围 -0.3551～+1.1323 pp。这达到了本轮值得追的提升量级，但仍是已看过开发划分上的探索结果。

冻结endpoint的Replay-PPO均值为92.2292%，Plain为92.5141%，配对平均差-0.2849 pp。主endpoint和冻结endpoint均用inner J挑选；这里的outer数值仅用于开发评价。主endpoint的阳性没有延续到冻结endpoint，因此可继续的是回放辅助搜索方案，尚不能宣称最终策略单独rollout已改善。扩大验证时同时跟踪两个endpoint。

折内先平均三个seed，主endpoint如下；两个fold等权平均得到上面的总体结果。

| fold | Plain BAcc | Replay-PPO BAcc | Replay-only BAcc | Replay−Plain pp |
|---|---:|---:|---:|---:|
| 0 | 93.5598% | 93.7010% | 93.5954% | +0.1412 |
| 1 | 91.4371% | 92.2714% | 91.4033% | +0.8343 |

All-LR仅作不同分类器的系统参照；相对它的差值不能全部归因于回放或RL。

**策略是否更会利用改进动作。** 以下NLL变化是在每批PPO结束、纳入同一批回放样本之后，对同一个回放池比较回放更新前后；正数表示保留动作平均联合概率提高。不同方法最终回放池不同，不能仅比较最终NLL宣称谁的动作更好。

| 方法 | 最终池数量范围 | 池内即时降分动作范围 | probe最终head熵范围 | 同池每批NLL平均下降 | NLL下降批数 |
|---|---:|---:|---:|---:|---:|
| plain_single_ppo | 130–275 | 39–99 | 3.4804–3.4808 | 0.00000 | 0/96 |
| replay_ppo | 171–336 | 48–114 | 3.1132–3.3793 | 0.02842 | 95/96 |
| replay_only | 145–299 | 43–110 | 3.1440–3.3916 | 0.03913 | 96/96 |

均匀Single动作的平均head熵为3.481122。Replay-PPO已明显形成动作偏好，并提高了其保留改进动作的概率；这不等价于证明每个状态的动作质量提高。首批固定16个原动作并不全是改进动作，其概率变化必须如实保留：

| fold | seed | 首批原动作平均概率：训练前 | 训练后 | 倍率 |
|---|---:|---:|---:|---:|
| 0 | 2026092501 | 0.0009469 | 0.0009778 | 1.033 |
| 0 | 2026092502 | 0.0009462 | 0.0060501 | 6.394 |
| 0 | 2026092503 | 0.0009466 | 0.0007757 | 0.819 |
| 1 | 2026092501 | 0.0009468 | 0.0008673 | 0.916 |
| 1 | 2026092502 | 0.0009471 | 0.0010415 | 1.100 |
| 1 | 2026092503 | 0.0009465 | 0.0007310 | 0.772 |

Replay-only主BAcc为92.4994%，比Replay-PPO低0.4868 pp，相对Plain仅+0.0009 pp，基本持平。本轮没有出现“只靠回放就胜过PPO+回放”的均值现象；不做更强的机制归因。

**一次修订决定。** 不追加64步：基础16步已带来开发准确率阳性，且动作偏好明显改变，不符合“池充分但策略仍接近均匀”的加步数动机。决定及数值依据保存于refine-decision.json，配置中refine.enabled=false。未启动Pair/H16、模型/K扫描或可变维数路线。

**下一处最值得改什么。** 优先扩大当前16步Replay-PPO的开发fold/seed覆盖，保留同一SVC、K32和回放定义。先确认这条准确率信号的覆盖面，再分析具体阳性机制；当前不优先继续压低entropy，也不启动16–48维add/delete/STOP备选。

**验证与可继续产物。** 本轮定向pytest实际12 passed（7项新回放测试+5项原block测试），新改Python的ruff通过。真实smoke为2条Single/H8轨迹、51次SVC拟合，完成1次PPO更新和16步真实回放及checkpoint恢复；未伪造轨迹。最初发现torch1.11不支持weights_only参数，修复后重新通过加载与测试。正式端点分数/映射复核见verification.json。18个case均为1440次J请求，6组初始参数hash一致，见case-invariants.json。

每case有主/冻结endpoint及两套坐标、trajectories.jsonl、subsets.jsonl、learning_curves.csv、ppo_updates.csv、replay_updates.csv、replay_pool.json、probe.json和checkpoint.pt。checkpoint含模型、原Adam、archive、进度、回放池和torch/CUDA/elite/replay RNG；再次执行run会跳过已完成case，未完成case可按批恢复。具体RNG约定见rng-conventions.json；全部表格来自本轮真实结果。

