# 任务 7：环境条件化特征选择前提验证协议

版本：`task-07-prerequisite-gate-v1`

日期：2026-09-21

状态：**BLOCKED / NO-GO（数据准入失败；禁止读取性能结果）**

## 1. 决策与适用边界

本协议只执行任务 7 的数据准入检查。当前 `v16n_2x_noise` 没有可核验的
`base_scene_id`、增强父子关系或可连接的海况、擦地角、SCR 元数据，现行 radar loader 也返回
`groups=None`。因此任务 7 在任何性能结果读取、环境分箱、outer split、特征搜索或模型训练之前即被
阻断。

本次决定是**数据前提 NO-GO**，不是全局静态、分环境静态、监督条件化、contextual bandit 或 PPO
的性能结论。不得把“未运行”写成算法失败。

本次强制执行：

- 历史 source-test 特征/标签读取次数为 0；
- 任务 7 性能结果及基线结果文件读取次数为 0；路线与 3A/4B 协议文档中的既有摘要只用于判定准入；
- J evaluator 请求数、唯一评分子集数和分类器拟合数均为 0；
- 不创建环境分箱、伪 scene/group、outer folds、候选子集或实验结果目录；
- 不按行序、相似度、聚类、文件邻近关系或既有结果反推 scene/group/env。

## 2. 性能读取前的原子准入门槛

以下条件必须同时满足；任一项失败立即保持 `BLOCKED / NO-GO`：

1. 最终数据每一行可通过签名 ledger 连接到来源可靠的 `base_scene_id` 或等价父场景标识；
2. 所有增强副本可连接到 `augmentation_parent_id`，并能验证同一 base scene 及其全部增强副本从不跨
   source partition 或 outer fold；
3. 海况、擦地角和 SCR 具有冻结的物理定义、单位、来源、采集时点及 `base_scene_id` 外键；
4. 用于条件化的每个环境变量在**选择任何特征之前**真实可获得，不依赖待选特征、标签或事后估计；
5. 冻结环境表示或分箱后，每个 `environment × class` 单元至少有 25 个**独立 base scenes**；增强行
   不增加这个计数；
6. 预生成的 scene-group-disjoint 五折中，每个 validation fold 的每个
   `environment × class` 单元至少有 5 个独立 base scenes；
7. 每个 group fold 同时覆盖全部冻结类别和环境。只要有一个 fold 无法覆盖，直接 `NO-GO`，不得通过
   改分箱、删环境或重抽更有利的 folds 补救；
8. ledger、environment catalog、生成 manifest、最终数据和 fold manifest 均有 hash，并通过确定性
   完整性、唯一性、外键覆盖和 group-overlap 检查。

计数和 fold 可行性必须在任何性能列、候选子集评分或 outer-validation 指标被读取之前完成并冻结。

## 3. 准入通过后才允许冻结的评价合同

本节是后续准入合同，不授权在当前数据上执行实验。准入通过后，必须在读取性能前一次性冻结：

- 物理环境表示/分箱及其边界；不得根据结果调整；
- scene-group-disjoint 五折 outer split；
- `K=32`；
- 4B 规格的 fold-local `StandardScaler + LogisticRegression(C=1.0, solver=liblinear,
  class_weight=balanced, max_iter=5000)`，指标为 Balanced Accuracy，3×5 folds，聚合器为
  `mean_minus_0_5_sd`；
- 所有方法完全相同的**总 J evaluator 请求预算**；等环境预算只能作为次级结果；cache hit 与重复请求
  仍计入请求预算；
- 全局静态 K=32、仅由各 outer-train 得到的分环境静态 K=32、简单监督条件化选择器，以及仅在环境
  决策时确实已知时可加入的 contextual bandit；
- 未见环境固定回退到对应 outer-train 内得到的全局 K=32，不得在结果后更换；
- 平均 LR BAcc、最差环境 BAcc、环境间方差、未见 scene/env 表现、选择稳定性，以及初始化、训练、
  推理、J 请求、唯一子集、模型拟合和墙钟时间的完整成本；
- GO 门槛：相对全局子集平均至少 `+0.2 pp`、至少 80% outer folds 为正、最差环境恶化不超过
  `0.2 pp`、控制总预算和模型复杂度后收益仍存在、未见环境不明显崩溃。

门槛失败即停止环境条件化路线。若简单监督条件化或 contextual bandit 最好，则保留简单方法。只有问题
具有真实序贯结构（特征成本不同、依据已观测特征选择下一项、或需要学习停止）时才允许提出多步 RL；
环境已知且只做一次子集选择时不得强行使用 PPO。

## 4. 当前准入审计

| 原子条件 | 当前状态 | 证据 |
|:---|:---|:---|
| 可靠 `base_scene_id` | 失败 | SVM 无 `qid`/注释；目标 manifest 无行级 scene 键 |
| 可靠增强父子关系 | 失败 | 无目标版本 generator、row ledger、shuffle permutation 或 split indices |
| 可靠环境字段 | 失败 | 目标数据/manifest 无海况、擦地角、SCR；旧 v15 解析不能归因到目标版本 |
| 选择前环境可获得 | 未证明，按失败处理 | 现行 selector 输入和部署说明均未提供这些字段 |
| 同源不跨 fold | 不可验证，按失败处理 | `groups=None`，没有合法 group 可供 split |
| 每环境每类至少 25 scenes | 不可计算，按失败处理 | scene 与 environment 外键均缺失 |
| 五折每格至少 5 scenes 且覆盖类/环境 | 不可构造，按失败处理 | 不得从现有行推造 group/env |

结论：**BLOCKED / NO-GO；不得运行四类基线或任何条件化/RL 实验。**

## 5. 解除阻断的最小数据补充

只接受以下最小可验证数据包；仅有汇总计数、文件名约定或近邻匹配不够：

1. **目标版本生成 manifest**：实际生成入口、代码 commit/SHA-256、输入文件及 hash、完整配置/CLI、
   物理参数分布、scene seeds、噪声模型与强度、copy count、标准化 fit scope、shuffle 和 split 规则；
2. **全输出 row-lineage ledger**：每个最终 train/test 行至少含
   `dataset_version, output_split, output_row_index, original_row_id, base_scene_id,
   augmentation_parent_id, augmentation_copy_id, augmentation_type`，且能直接对齐最终文件；
3. **scene environment catalog**：以 `base_scene_id` 为唯一键，至少含 class/target、海况、擦地角、
   SCR、字段定义、单位、原始文件或 scene seed；
4. **选择时可用性合同**：逐个环境变量说明在特征选择前由何传感器/系统提供、时间语义、缺失值策略，
   并证明不由待选择特征或标签事后计算；
5. **验证脚本与冻结 fold manifest**：校验主键唯一性、ledger/catalog 外键全覆盖、parent/base-scene
   不跨 partition/fold、每个 `environment × class` 至少 25 个独立 scenes、每个 validation fold 每格
   至少 5 个 scenes，并记录所有输入/输出 hash。

若历史数据无法补齐，最小可接受替代是从保留 scene/env 键的新 generator 重新发布一个版本化数据集；
必须在生成前冻结上述内容，并只生成一个正式批次。
