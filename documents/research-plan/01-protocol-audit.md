# Radar-Ship-FS / GNN-PPO-FS / DFS 协议审计

日期：2026-09-11
状态：代码审计与小规模验证基线

## 1. 审计结论

既有结果不能作为三种方法已经完成公平比较的证据。主要原因不是模型字段命名，而是实际执行路径之间存在数据版本、训练划分、评分器、候选归档、特征坐标和 test 使用方式的差异。

本次修复建立了以下边界：

1. 所有特征选择结果同时记录清洗后 0-based 索引和原始 1-based feature ID，并由同一个映射模块生成和校验。
2. 新科研主配置关闭随机 feature ID 奖励及其 node feature、候选池和 shaping 贡献；历史配置保持不变以便复现。
3. 统一入口中的 PPO 可以使用与 DQN 相同的 source-train、相同交叉验证折和同一决策树 probe；旧的 80/20 holdout 归档模式仍保留为 legacy 模式。
4. MI Top-K、MI 顺序接受改进和逐步枚举全部剩余特征的 forward greedy 被拆成三个独立方法。
5. 任何读取 source test 后选择子集、K、DFS penalty 或 checkpoint 的程序都被标为 oracle diagnostic，并要求显式确认参数；它们不得产生正式比较结果。
6. 当前工作区没有可访问的 DFS 训练代码和可溯源训练产物，因此本次不声明 DFS 成绩。

## 2. 数据与特征坐标证据

审计时在 `/root/feature-select` 及其子目录未发现 `AGENTS.md`，因此没有额外的仓库级代理约束可应用。

审计对象 `v16n_2x_noise` 的实际文件如下（路径相对 `/root/feature-select`）：

| 文件 | 行数 | SHA-256 |
|---|---:|---|
| `dataset/sim_ship_cr_v16n_2x_noise.train.svm` | 3897 | `2191fdc297aa49ce6a700df956ec44f68dc13e4c737937b31167543a203d22bb` |
| `dataset/sim_ship_cr_v16n_2x_noise.test.svm` | 1671 | `6817c73da48cdc32fe0262e35c05532c0c4552cdbeb1349db9a6e16f0f014e72` |

每行含 75 个原始特征和 1 个标签。数据加载器只用 source-train 拟合清洗规则，再将规则应用到 source-test。常量列原始 1-based ID 为 `13, 43`；重复列关系为：

```text
5→2, 6→3, 8→3, 15→14, 27→23, 46→42, 61→60, 71→69
```

清洗后共有 65 个特征。clean 0-based 索引按下列原始 1-based ID 顺序定义：

```text
[1, 2, 3, 4, 7, 9, 10, 11, 12, 14, 16, 17, 18, 19, 20, 21,
 22, 23, 24, 25, 26, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
 38, 39, 40, 41, 42, 44, 45, 47, 48, 49, 50, 51, 52, 53, 54,
 55, 56, 57, 58, 59, 60, 62, 63, 64, 65, 66, 67, 68, 69, 70,
 72, 73, 74, 75]
```

坐标定义冻结为：

- `clean_indices_0based`：清洗后 65 列矩阵的列索引。
- `original_feature_ids_1based`：原始 75 特征空间的论文/数据 feature ID。
- `raw_indices_0based = original_feature_ids_1based - 1`：未经清洗的 75 列 NumPy 矩阵索引。

### 已发现的错列

| 入口 | 原行为 | 风险 | 修复 |
|---|---|---|---|
| `test_final_ppo_val.py` | 将原始 1-based ID 直接用作原始矩阵列索引 | 整体偏移一列，ID 75 还可能越界 | 明确转成 raw 0-based |
| `test_dfs_cheat.py` | 原始 1-based ID 或 clean 0-based 索引直接索引原始矩阵 | 两种坐标均可能取错列 | 按字段语义分别映射 |
| `test_ppo_cheat_18.py` | clean 0-based 索引直接索引 75 列原始矩阵 | 删除列之后发生错列 | clean→original→raw 映射 |
| `test_ppo_fair_18.py` | 同上 | 同上 | 同上 |

新增 `radar_ship_fs.feature_mapping.FeatureIndexMap` 作为唯一转换入口。新产物生成时同时写入两套坐标；读取已有产物时，如果两字段并存则逐元素交叉校验，拒绝重复、越界、指向已删除列或互相不一致的选择。

## 3. 实际执行协议，而非字段名推断

### 3.1 Radar-Ship-FS DQN / MARLFS / fixed-IRFS

- 输入：`v16n_2x_noise` 的完整 source-train，清洗规则仅由 source-train 拟合。
- 状态评分：固定 `StratifiedKFold` 上的 `DecisionTreeClassifier` 平均 accuracy。
- 优化目标：`J = accuracy - beta * mean_abs_correlation`，并可选超预算惩罚。
- DQN 候选：实际访问的子集；归档按 accuracy 优先，accuracy 完全相等时选择更少特征。
- MARLFS 基线：生成均匀重要性向量，不等价于训练过的 MARLFS 代理。
- fixed-IRFS：完整特征上拟合决策树，以 feature importance 作为固定排序。
- 最终评价：在完整 source-train 上分别拟合固定 `StandardScaler + LogisticRegression(C=1.0, solver=liblinear, max_iter=5000, class_weight=balanced)` 主分类器和决策树次分类器，在 source-test 上仅评价已冻结子集。

因此，不能把 selector 类名或结果 JSON 中的通用字段理解为同一学习算法、同一目标或同一归档策略。

### 3.2 Radar-Ship-FS 统一入口 PPO

旧路径的实际行为是：

- 将 source-train 再切成 80% PPO-train / 20% PPO-val；
- graph 和训练 reward 使用 80% 部分上的固定 K-fold 决策树；
- archive 使用单个 20% holdout 的决策树 accuracy；
- reward、候选池和 shaping 可以包含随机 feature ID；
- 这并不是注释或独立工程描述中的 repeated 5×5 CV。

新 `evaluation_protocol = "shared_inner_cv"` 路径改为：

- 使用完整 source-train；
- PPO reward、候选评分和 archive 共用 ExperimentRunner 创建的同一个交叉验证 probe 与固定 folds；
- archive 与 DQN 同为 accuracy 优先、完全相等时更少特征；
- source-test 不参与策略更新、候选生成、候选池、归档或 penalty/K 选择。

仍需在论文中披露：DQN 归档所有访问过的候选，而当前 PPO 向统一归档提交每批/回合形成的候选，这两者的搜索预算口径不能只用“episode/step”名称比较，必须同时报告唯一子集数和 scorer 调用数。

### 3.3 独立 GNN-PPO-FS 工程

审计到的默认配置与 Radar 新主配置不相同：

- 默认数据是 `v16n`，不是 `v16n_2x_noise`；
- 使用完整 source-train 和 repeated 5×5 CV audit；
- 默认 K=32；
- 最终分类器为决策树；
- 已有 zero-learning-rate 运行只能说明在相应初始化/候选机制下得到结果，不能归因为 PPO 学习。

因此独立 GNN-PPO-FS 的既有数字不能直接进入 `v16n_2x_noise` 公平比较表。若迁移结果，必须通过冻结协议重新运行并记录数据 hash、映射、scorer、K、预算和候选来源。

### 3.4 三种容易混淆的 MI / greedy 方法

| 方法 | 实际定义 | 子集评分次数 |
|---|---|---:|
| MI Top-K | 单变量 MI 排序后直接取前 K，不根据子集分数接受/拒绝 | 排序本身，不逐子集优化 |
| MI ordered accept | 只沿固定 MI 顺序考察；加入当前特征使 inner-CV accuracy 改进时才接受 | 至多考察每个排序特征一次 |
| exhaustive forward greedy | 每一步枚举所有尚未选择的特征，选择使 inner-CV accuracy 改进最大的一个 | 约为 `p + (p-1) + ...`，受 K/无改进提前停止影响 |

旧 `MIGreedySelector` 属于第二种，不是第三种。本次保留旧名称作为历史别名，并新增明确名称 `mi_ordered_accept` 和真正的 `forward_greedy`。

### 3.5 DFS

当前工作区的 `cxc-dfs/` 只有评价脚本 `eval_v16n_v3.py`，该脚本硬编码引用不可访问的 `/data/cxc/dfs-master`。Radar 仓库当前版本和可见历史中均未找到 DFS 训练实现或可验证训练产物。

现有 `Radar-Ship-FS/tmp/test_dfs_lr_v16n.py` 与 `tmp/dfs_lr_v16n_results.json` 只是对人工转录的 raw 0-based 特征列表做 LR 评价，并出现 `f22=f26` 的重复列迹象。它们不能回答 DFS 的训练数据、reward/scorer、penalty 选择、随机种子、checkpoint 或候选生成来源。

结论：DFS 暂时标为 `not reproducible / not eligible for formal comparison`，不得补写或推测成绩。纳入条件见 `protocol.md`。

## 4. 随机 feature ID 泄漏式奖励

历史 PPO 默认可将随机 feature ID 用于三处：node feature、候选池打分和 reward shaping/终局目标。这种随机编号没有数据语义，且可能改变候选可达性；只把最终 reward 字段设成 0 并不足够。

新科研主配置采取四层禁用：

```toml
feature_id_node_feature = false
feature_id_reward_weight = 0.0
sparsity_bonus = 0.0
archive_min_cv_gain = 0.0
```

实现同时保证 weight 为 0 时不计算、不添加随机 ID 候选池分数，不进入 shaping，也不进入 terminal objective。回归测试验证不同 feature-ID seed 下候选分数、proxy reward 和 objective 保持不变，并验证图节点可以完全不包含随机 ID channel。

原历史配置和产物未覆盖，仅增加醒目的 historical 标注；需要复现旧实验时仍可使用旧配置。

## 5. Source-test oracle 隔离

下列程序读取 source-test 后再选择研究对象，定义上只能是 oracle / sensitivity diagnostic：

- `test_dfs_cheat.py`：按 source-test 选择 DFS trajectory/子集；
- `test_ppo_cheat_18.py`：按 source-test 选择 PPO 候选；
- `test_greedy_all_n.py`：按 source-test 扫描 K；
- `cxc-dfs/eval_v16n_v3.py`：one-vs-rest 路径可按 source-test 选择 DFS artifact 或 penalty。

这些入口现在要求显式传入 `--acknowledge-source-test-oracle`，且集中记录在 `/root/feature-select/oracle_diagnostics/README.md`。其输出不得写入正式比较目录，不得用于选择正式子集、K、penalty、checkpoint、seed 或模型。

`run_domain_gcn_screen_eval.py` 仍允许对已冻结子集做 source-test 评价，但产物明确标记 `reused_source_test_diagnostic_only=true`、`historical_source_test_already_observed=true`、`eligible_as_fresh_independent_validation=false`。

## 6. “独立验证”的声明限制

本仓库开发过程中已反复查看 `dataset/sim_ship_cr_v16n_2x_noise.test.svm`。即使某次新运行没有在代码路径中读取 test 来选子集，该 test 也不能重新宣称为“全新的独立验证集”。人的迭代决策已经可能通过历史观察适配该 test。

后续可采用的准确表述是：

> 在历史上已观察的固定 source-test 上，对事先冻结且不由本次 test 选择的子集进行复用测试诊断。

真正的新独立验证需要此前未访问的新数据；若无法取得，应使用完整 nested CV / repeated outer CV，并把所有选择、K、penalty 和模型调参限制在 outer-train 内。

## 7. 修复与回归测试清单

主要新增或修改内容：

- `src/radar_ship_fs/feature_mapping.py`：显式坐标转换和产物一致性校验。
- `src/methods/forward_greedy.py`：逐步枚举全部剩余特征的 forward greedy。
- `src/methods/mi_greedy.py`：明确为 MI-ordered accept，保留历史别名。
- `src/radar_ship_fs/ppo/{ppo_graph.py,ppo_env.py,run_session.py}`：关闭随机 ID 的全部影响，新增 shared-inner-CV 路径，移除隐式候选转储。
- `src/radar_ship_fs/experiment/{artifact.py,config.py,runner.py}`：统一映射、协议字段和方法注册。 新 manifest 还保存实际 CV/tree random state 和逐折原始行号。
- `configs/v16n/research_baseline.toml`：新的科研主配置。
- `configs/v16n/research_smoke.toml`：仅用于小规模链路验证，不用于科研结论。
- `tests/test_feature_mapping.py`：边界、双向映射、错配、删除列和重复选择测试。
- `tests/test_greedy_protocols.py`：三种方法的语义差异和枚举行为测试。
- `tests/test_ppo_feature_ids.py`：随机 ID 零权重不变性测试。
- `tests/test_stable_config.py`：科研主配置冻结字段测试。

验证结果：本次相关文件 Ruff 检查通过，定向 pytest 为 `35 passed`。真实 `v16n_2x_noise` 上的最小 smoke 配置已跑通 MI Top-K、MARLFS-style control 和 GNN-PPO 的装载、选择、归档与序列化，再由最终评价入口读取冻结子集完成 LR/DT source-test 复用诊断。smoke 输出位于 `/tmp/radar_ship_research_smoke_v1`，不属于科研结果，也未覆盖历史目录。全仓库 Ruff 另有 13 个位于未改动旧文件的既存 import/unused-import 问题。全量 pytest 在收集阶段还因当前工作树缺少旧测试引用的 `radar_ship_fs.legacy`、`methods.advice`、`methods.configure` 和若干 `run_stage2_*` 模块而产生 11 个既存 import errors；因此不能声称全量 suite 通过。

此外，已只读扫描 `experiments/v16n_2x_noise` 的 20 个既有 `selection.json`：全部 clean/original 双坐标字段通过新映射器交叉校验。该结论只说明这些字段彼此一致，不会追溯性地使旧 PPO 的数据划分、随机 ID reward 或 archive 协议变成新协议。

## 8. 剩余限制

1. DFS 训练代码、训练产物、commit、环境和数据 provenance 缺失。
2. 独立 GNN-PPO-FS 的历史结果使用不同数据版本和评价器，需要按冻结协议重跑后才能比较。
3. source-test 已被历史反复观察，只能作为复用诊断。
4. 搜索算法的预算需报告 scorer 调用数和唯一子集数；单报 episodes、environment steps 或运行时间不公平。
5. 不同方法可能返回少于 K 的子集；正式表必须同时报告实际 K，不能只按配置上限分组。
6. 新科研配置是下一阶段的协议起点，不追溯性地改变任何旧产物的含义。
7. 当前工作树缺失一组 legacy/stage2 模块，使旧全量测试无法完成收集；在补齐或移除失效测试前，回归置信度以本次定向测试、独立 GNN-PPO 测试和 smoke 链路为限。
