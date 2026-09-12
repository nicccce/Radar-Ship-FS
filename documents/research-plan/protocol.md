# 下一阶段冻结实验协议

版本：`research-baseline-v1`
冻结日期：2026-09-11
主配置：`configs/v16n/research_baseline.toml`
规范化配置 SHA-256：`6de4eb01b4b2cf7915e230cc06cacfa808f79a5f8fb0f79bfbf5eb5c14a0bf88`

## 1. 适用范围与禁止事项

本协议用于 Radar-Ship-FS、迁移后的 GNN-PPO-FS 以及未来可复现的 DFS 比较。任何偏离项必须在运行前生成新的协议版本，不能在查看评价结果后回写本协议。

正式比较禁止：

- 使用 source-test 选择特征子集、K、penalty、checkpoint、seed、分类器或超参数；
- 将随机 feature ID 用作 node feature、候选优先级、reward、shaping 或 archive 条件；
- 混用 clean 0-based、original 1-based 和 raw 0-based 坐标；
- 将不同数据版本、不同主分类器或不同 K 的既有数字放入同一主表而不重新运行；
- 将历史上已反复查看的 source-test 描述为新的独立验证。

## 2. 冻结数据版本

数据集：`v16n_2x_noise`。下列路径相对 `/root/feature-select`。

| split | 路径 | 行数 | SHA-256 |
|---|---|---:|---|
| source-train | `dataset/sim_ship_cr_v16n_2x_noise.train.svm` | 3897 | `2191fdc297aa49ce6a700df956ec44f68dc13e4c737937b31167543a203d22bb` |
| reused source-test | `dataset/sim_ship_cr_v16n_2x_noise.test.svm` | 1671 | `6817c73da48cdc32fe0262e35c05532c0c4552cdbeb1349db9a6e16f0f014e72` |

原始空间为 75 个特征；清洗后为 65 个。常量列、重复列和完整 clean↔original 映射以数据加载器生成的 metadata 以及 `FeatureIndexMap` 校验为准。清洗规则只能在 source-train 上拟合，然后原样应用到 source-test。

每个结果必须记录：数据 hash、原始维数、清洗后维数、常量列、重复列映射、`clean_indices_0based`、`original_feature_ids_1based` 以及两者交叉校验结果。

## 3. 随机种子与划分

正式种子：`42, 43, 44, 45, 46`。

每个 seed 先按冻结的 `validation_fraction=0.25` 将 source-train 确定性分成临时 train/validation，再按 train 后 validation 的顺序合并；因此全部 3897 行仍进入选择，但该字段会影响行顺序和共享 RNG 状态。随后从 `SeededRng(seed)` 确定性抽取 CV/tree random state，建立一次固定 `StratifiedKFold(n_splits=5, shuffle=true)`。manifest 必须保存实际 random state 与 fold row indices。同一 seed 下 DQN、PPO、MI-ordered accept、forward greedy 和其他需要子集评分的方法复用这个 context、fold indices 和决策树 random state，不能各自重新抽样。`shared_inner_cv` 不再创建 PPO 专属 20% archive holdout。

source-test 不属于选择划分。它只在该 seed 的子集及所有超参数冻结后评价一次，并标记为 reused diagnostic。

## 4. 特征数 K

- 最大特征预算：`K_max = 32`。
- MI Top-K：固定返回恰好 32 个可用特征。
- PPO 主运行：目标为恰好 32 个特征；初始化和动作修复不得通过随机 ID 决定成员。
- DQN、MI-ordered accept、forward greedy：允许因无改进或 archive 规则返回少于 32 个特征，但必须报告实际 K。
- 主表不得把不同实际 K 隐藏在共同的 “K=32” 标题下。至少同时给出 accuracy、实际 K、唯一 scorer 调用数。
- 若需要严格 matched-K 分析，另建预注册协议；不得看过 source-test 后补齐或截断。

## 5. 选择评分器与奖励

统一选择评分器：

```text
DecisionTreeClassifier(random_state=cv_tree_random_state)
cv_tree_random_state = 从 SeededRng(seed) 确定性抽取并写入 manifest
metric = accuracy
folds = 第 3 节冻结的同一组 source-train 5-fold indices
subset score = 五折 accuracy 算术平均
```

统一 RL 数据目标：

```text
J(S) = CV_accuracy(S) - 0.02 * mean_abs_correlation(S)
```

超出 `K_max` 的候选无资格进入 archive；不得通过 source-test 决定 over-budget penalty。若方法内部需要 dense shaping，shaping 只能由 source-train 上的冻结统计量和上述 scorer 派生，并在产物中逐项记录权重。

PPO 的科研配置冻结为：

```text
feature_id_node_feature = false
feature_id_reward_weight = 0
sparsity_bonus = 0
evaluation_protocol = shared_inner_cv
```

历史配置不修改；历史结果必须标为 legacy protocol。

## 6. 方法定义

- `mi_topk`：单变量 mutual information 排序，直接取前 K；不称为 greedy。
- `mi_ordered_accept`：沿一次固定 MI 排序依次尝试，只有加入特征使统一 inner-CV accuracy 严格提高才接受；不称为 exhaustive forward greedy。
- `forward_greedy`：每一步对所有剩余特征分别调用统一 scorer，选择最佳严格改进，直至 K 或无改进。
- `marlfs`：当前实现若仍是 uniform importance，仅可标为 uniform/MARLFS-style control，不得声称已训练 MARLFS。
- `full_irfs_fixed`：全特征决策树 importance 的固定排序基线。
- `gnn_ppo`：必须使用本协议的数据、映射、folds、scorer、K 和 archive；独立 GNN-PPO-FS 工程的默认结果不自动满足本协议。
- `dfs`：只有满足第 10 节 provenance 门槛后才进入主表。

## 7. 候选归档规则

候选资格：只使用 source-train 产生和评分，`1 <= |S| <= 32`，坐标校验通过，无重复特征。

统一择优顺序：

1. inner-CV accuracy 更高者优先；
2. accuracy 完全相等时，特征更少者优先；
3. 仍相等时保留先到候选，不使用 source-test 破同分。

必须记录：算法访问候选总数、唯一子集数、实际 scorer 调用数、cache 命中数、进入 archive 的候选序列及来源。PPO 与 DQN 若提交候选的频率不同，应作为搜索机制差异披露，不能假称归档机会完全相同。

## 8. 最终评价器

主分类器冻结为：

```text
Pipeline([
  StandardScaler(),
  LogisticRegression(
    C=1.0,
    solver="liblinear",
    max_iter=5000,
    class_weight="balanced",
    random_state=seed
  )
])
```

主指标：source-test balanced accuracy。原始 accuracy、macro-F1、per-class recall、ROC-AUC 和 confusion matrix 作为固定次指标报告，但不得用这些 source-test 指标反向选择子集或 seed。

固定 `DecisionTreeClassifier(random_state=seed)` 作为次分类器/机制敏感性结果，必须与 LR 主结果分栏，不得择优报道。

所有最终模型只在完整 source-train 上拟合一次，然后评价冻结的 source-test。任何新的分类器或超参数比较必须在 source-train 内完成并另行预注册。

## 9. 计算预算口径

主配置预算：

- DQN：每 seed `250` environment steps；另报唯一子集数和 DT-CV scorer 调用数。
- PPO：每 seed `64` episodes，动作/交换上限以主配置为准；另报唯一子集数、DT-CV scorer 调用数和策略更新次数。
- MI Top-K：一次 MI 排序；报告 MI 拟合次数。
- MI-ordered accept：至多按 65 个排序特征各评分一次，达到 K 或遍历结束停止。
- forward greedy：逐步枚举剩余特征，报告每一步候选数和累计 scorer 调用数。
- fixed importance：报告全特征 importance 模型拟合次数。

运行时间和硬件信息必须记录，但 wall-clock 只作资源说明，不作为唯一公平预算。缓存命中不能伪装成新的模型评价。

`research_smoke.toml` 仅验证装载、选择、归档、序列化、映射和最终评价链路；其结果没有科研解释力，不能进入结果表。

## 10. DFS 纳入门槛

DFS 在进入正式比较前必须提供并归档：

1. 可执行训练代码、依赖环境、代码 commit 和许可证/来源；
2. 训练与评价数据路径及 SHA-256；
3. 输入坐标声明和可自动验证的 clean/original/raw 映射；
4. reward/scorer、分类器、划分、K、penalty 和停止条件；
5. 所有 seed、checkpoint 和候选轨迹；
6. 不读取 source-test 的 checkpoint、penalty 和最终子集选择规则；
7. 按本协议运行产生的新产物，而非人工转录列表。

任一项缺失时，主表写 `DFS: unavailable / protocol provenance incomplete`，不填估计值、不从 oracle 脚本挑最好值。

## 11. 结果目录与不可覆盖要求

新正式产物写入独立目录 `experiments/research_baseline_v1/`。主配置固定 `resume=true`：相同身份的完整结果直接跳过，不同 config/algorithm 身份被拒绝；正式运行禁止用 CLI `--no-resume` 覆盖同身份产物。若有意建立新协议，必须创建带新版本/manifest/hash 的新 run 目录。历史 `experiments/`、`tmp/` 和独立 GNN-PPO-FS 的既有结果保持原样。

每次正式运行保存：配置原文与 hash、代码 commit/dirty 状态、Python/依赖版本、硬件、数据 manifest、fold indices、方法预算计数、完整候选 archive、最终选择、评价输出和失败日志。

## 12. 验证声明模板

允许的表述：

> 子集仅由 source-train 的固定 inner-CV 协议选择；冻结后在历史上已观察的 source-test 上进行复用诊断评价。

不允许的表述：

> 该 source-test 是全新的独立验证集。

只有此前未被研究团队或自动调参流程访问的新数据，才能支持后一类独立验证声明。否则采用 nested/repeated outer CV 估计泛化性能，并把所有适配限制在 outer-train 内。
