# 08A：block-rewrite-ppo-v1 开发探索结果

日期：2026-09-22。版本：`accuracy-exploration-v3`。本报告只使用 source-train 的固定
outer fold 0/1，是看过结果的开发探索，不是独立泛化证明；source-test 读取次数为 0。

## 做了什么

实现了条件化 `r1→r2→a1→a2` 宏动作 Pair-PPO。合法动作即执行，允许当前 J 下降；
联合 log-prob 进入一次 PPO ratio，四个 head 的 entropy 取平均。状态包含 selected mask、进度、
当前 J 和 episode-best J。对照为同框架 Single-PPO、等时长 Random-Pair、完整 forward 加一轮
1056 single-swap 的 G1，以及 MI32、All-SVC、All-LR。prior、feature-ID、proxy shaping、
相关性/稀疏奖励均关闭。

## 准确率结果

最高 outer BAcc 为 **93.8136%**，来自
`base/single_ppo`（fold=0，
seed=2026092401）。基础 Pair-PPO 四个 case 的平均 outer BAcc 为
**92.5386%**，冻结 rollout endpoint 平均为
**92.4348%**。Pair-PPO 的单 case 最高值为
**93.5569%**（
`base`，fold=0，
seed=2026092401）。

| fold | seed | Pair-PPO BAcc | Pair−G1 |
|---:|---:|---:|---:|
| 0 | 2026092401 | 93.5569% | +0.1326 pp |
| 0 | 2026092402 | 93.0436% | -0.3808 pp |
| 1 | 2026092401 | 92.2425% | +1.0454 pp |
| 1 | 2026092402 | 91.3116% | +0.1145 pp |

| Pair 相对对照（4 case 均值） | BAcc 差 |
|:---|---:|
| all_svc | -0.1091 pp |
| g1 | +0.2280 pp |
| mi32 | +0.3742 pp |
| random_pair | +0.3083 pp |
| single_ppo | -0.0884 pp |

Pair−G1 原始四个 case 差值（BAcc）为
`[0.1326, -0.3808, 1.0454, 0.1145]` pp；fold 内先平均两个 seed 的结果见
`paired_differences.csv`。是否存在单 case ≥+0.50 pp 的局部阳性：
**True**。

| 两折基线均值 | outer BAcc |
|:---|---:|
| g1 | 92.3107% |
| mi32 | 92.1645% |
| all_svc | 92.6477% |
| all_lr | 91.8638% |

## 一次修订

选择：horizon16; 状态：complete。理由：at least one complete Pair-PPO case beat fold-matched G1 and late training still contained positive rewards; extend only the path horizon


| fold | seed | horizon16 BAcc | horizon16−G1 |
|---:|---:|---:|---:|
| 0 | 2026092401 | 93.0778% | -0.3465 pp |
| 0 | 2026092402 | 92.6542% | -0.7701 pp |
| 1 | 2026092401 | 91.6921% | +0.4950 pp |
| 1 | 2026092402 | 91.2247% | +0.0276 pp |

horizon16 四个 case 平均 BAcc 为 **92.1622%**；
相同 fold/seed 下比基础 Pair 平均 **-0.3764 pp**，
比 G1 平均 **-0.1485 pp**。修订没有放大基础版局部阳性。

## 训练与可信度检查

每个基础搜索 case 发出 1440 次 J 请求；Pair/Single 各完成 16 次更新，Random 使用相同重启、
archive 和冻结 rollout。正式 endpoint 的独立 3-fold SVC 重算与 outer 重算状态为
**passed**。累计记录的物理 classifier fits 为 **78113**，
硬上限 90000；搜索与基线记录墙钟为 **2084.3s**。
全部子集保存 clean 0-based 与 original 1-based 两套坐标。

## 判断与下一处改动

基础 Pair 平均超过 G1（+0.2280 pp）、Random-Pair（+0.3083 pp）和 MI32（+0.3742 pp），
且有一个 Pair−G1 为 +1.0454 pp 的局部阳性；但它平均低于 Single-PPO（−0.0884 pp）和
All-SVC（−0.1091 pp），seed 波动明显。horizon16 又低于基础 Pair 和 G1，因此不继续推广该修订。
当前最诚实的结论是“成对搜索与学习存在值得复查的局部信号，但 Pair-PPO 尚未稳定赢过简单对照”。
下一次若继续，应只围绕基础版局部阳性增加重复/定位学习贡献；若该信号不能重复，则暂停这个块动作
版本，转向既定的 LR 可变大小 add/delete/STOP 路线，而不是继续扫描更多块动作超参。

完整逐 case 结果、训练曲线、checkpoint、轨迹、请求/拟合成本和验证记录位于
`experiments/block_rewrite_ppo_v1/`。
