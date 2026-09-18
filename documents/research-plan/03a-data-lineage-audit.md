# 任务 3A：`v16n_2x_noise` 数据谱系与场景分组审计

日期：2026-09-17
状态：**谱系不完整；真实 group-aware 验证不可执行；任务 7 数据前提不具备**

## 1. 结论摘要

本次审计没有恢复出可核验的 `original_row_id`、`base_scene_id`、
`augmentation_parent_id`、增强副本号、海况、擦地角或 SCR。因而：

- 不生成 source-train 行到真实 group/env 的映射文件；
- 不按行序、距离、相似度或聚类构造伪 group；
- 不建立 row/group split 敏感性实验协议，也不运行性能诊断；
- 不能判断同源增强是否跨 source-train/source-test 或跨既有 CV fold，亦不能据此声称已经发生或没有发生泄漏；
- 真实 scene/group-disjoint 验证当前不可行，任务 7 的数据准入条件未满足。

可以核验的最细粒度只到：`v16n_2x_noise` source-train 文件中的当前行位置、标签和 75 维特征值。
这些行位置不是生成前的 original-row ID，也不是 base-scene ID。

任务 4C 的结论保持不变：4C 仍为 **NO-GO**（平均 `+0.1009 pp`，正/平/负
`3/2/0`，两项效应门槛均未达到）。本审计没有更换 split、没有重跑 4C、没有训练 PPO/DQN，
也不会因为以后补齐谱系而自动重启旧静态 RL 分支。4D、5A、5B 和原任务 6 仍不准入。

## 2. 审计范围与读取边界

本次完整阅读了任务指定的路线、4C 复核、冻结协议和旧 v16n 数据报告，并检查了：

- 当前数据加载与 split 代码；
- 工作区内现存的生成/增强代码；
- `v16n_2x_noise` 既有 experiment manifest、日志和 Git 历史；
- source-train 的文件结构、行数、标签、清洗 metadata、精确重复和标准化数值签名；
- 旧生成链中场景文件名的解析方式，以及场景字段是否被写入下游数据。

历史 source-test 特征和标签读取次数为 **0**。本次没有计算 source-test hash，也没有通过 loader
打开 source-test。文中 source-test 的文件名、行数、类别计数和既有 SHA-256 均来自已保存的
`experiments/v16n_2x_noise/seed-42/marlfs/manifest.json` 与冻结协议，而不是本次读取数据文件所得。

审计开始时仓库为 `main...origin/main [ahead 1]`，已有修改为：

- `documents/research-plan/next-stage-prompts.md`（modified）；
- `documents/research-plan/next-stage-roadmap.md`（modified）；
- `documents/research-plan/04c-review-and-next-steps.md`（untracked）。

这些已有修改均未覆盖或改写。本文件在开始时不存在，因此是新增交付。

## 3. 数据集与当前可见 grain

### 3.1 文件级身份

| 对象 | 可核验信息 | 来源 |
|:---|:---|:---|
| source-train | 3897 行；75 个原始特征；标签 `-1: 2016, +1: 1881`；SHA-256 `2191fdc297aa49ce6a700df956ec44f68dc13e4c737937b31167543a203d22bb` | 本次 source-train-only loader 与独立文件 hash |
| reused source-test | 1671 行；标签 `-1: 864, +1: 807`；既有 SHA-256 `6817c73da48cdc32fe0262e35c05532c0c4552cdbeb1349db9a6e16f0f014e72` | 既有 manifest/冻结协议；本次未打开文件 |
| LIBSVM 结构 | 特征索引 `0..74`；无 `qid:`；无注释或附加列 | 本次仅检查 source-train 文本结构 |
| 清洗后矩阵 | `(3897, 65)`；常量 original 1-based ID 为 `13, 43`；8 个精确重复映射见下文 | 现行 `load_radar_ship_source_train` 在 source-train 上拟合 |

现行 loader 得到的重复映射（original 1-based）为：

```text
5→2, 6→3, 8→3, 15→14, 27→23, 46→42, 61→60, 71→69
```

source-train 的 73 个非常量列具有训练集 Z-score 导出签名：最大绝对列均值约
`2.44e-7`，最大 `|std-1|` 约 `1.50e-7`。这证明当前文件是训练分区拟合标准化后的导出，
但不能单独证明标准化前的输入、增强方法、父子关系或 split seed。

### 3.2 当前行位置不是原始场景标识

source-train 的 3897 行均为不同的完整 LIBSVM 文本行，解析后的矩阵也没有精确重复行。把旧
`sim_ship_cr_v16n.train.svm` 与当前 source-train 比较时，文本级、解析数值级以及小数点后 12 位
精确匹配均为 0 行。该结果只说明无法用精确相等恢复旧 source-train 行；噪声和重新标准化都可能消除
精确相等，因此它不证明两个版本无父子关系。

4C 的 `validation_fold_by_original_row` 和任务 3 context 中的 `*_original_rows` 实际表示当前
source-train 文件的 0-based 行位置。它们能复现实验 split，却不能提升为生成前 original-row、
augmentation parent 或 base scene。继续沿用 `original_row` 这一历史字段名时必须注明这个语义。

## 4. 生成关系追查

### 4.1 实际 `v16n_2x_noise` 生成入口：unavailable

工作区中只有以下两个同名前缀文件：

```text
sim_ship_cr_v16n_2x_noise.train.svm
sim_ship_cr_v16n_2x_noise.test.svm
```

没有发现同名 augmentation/lineage manifest、生成脚本、配置、命令日志、shuffle permutation、
split indices 或 row ledger。数据目录也不是 Git 仓库；两个 SVM 文件本身不在
`Radar-Ship-FS` Git 历史中。

Git 提交 `5f1cfe5a0bcf15c5d138503b934767fd94c4be87`（2026-08-13）只是把
`src/stage2_rl_config.py` 的版本从 `v16n` 改为 `v16n_2x_noise` 并归档旧实验脚本，没有加入或说明
数据生成代码。该提交晚于两个数据文件的文件时间戳，也不能充当生成 provenance。

`dataset/test.ipynb` 只加载已经存在的 `v16n_2x_noise` train/test 并做清洗与相关分析；它不写出
数据。其 IPython 历史中同样只保留加载/分析单元，没有增强、保存或 split 命令。

### 4.2 现存增强代码不能归因为实际生成器

`dataset/augment_features.py`（SHA-256
`e769061ba79e022adaf857c98e34f481edb069219859a7c36970a99082ace038`）包含
`FeatureNoiseAugmenter.generate_augmented_dataset(..., augment_factor=2)`。该函数本身只返回
“输入原样 + 一份扰动副本”，不负责读取 v16n、全局/分区 shuffle、train/test split、标准化或写出
`v16n_2x_noise` 文件。工作区没有调用它写出目标文件的入口或 manifest，也没有冻结其
`noise_scale`、`swap_prob`、seed 和 `blend_swap` 实参。因此它只能视为候选 helper，不能作为实际
生成关系证据。

旧的 `buildv15sim/build_sim_v15.py` 和 `generate_v15_3.py` 形成了另一个、版本明确的旧链路：

1. `build_sim_v15.py` 从文件名解析 `ang`、`wave`、`scr`、极化通道和 `sid`，并构造
   `type/ship/ang_wave_scr_scene_sid` 场景键；
2. 完成四极化配对和特征提取后，只把 `X/y` 写入 LIBSVM，场景键和环境字段没有进入输出；
3. `generate_v15_3.py` 读取只含 `X_train/y_train/X_test/y_test` 的 `v15_raw.npz`，在 train/test
   分区内分别生成每个原始行的 3 份增强副本、各自 shuffle，再用增强后的 train 拟合 Z-score；
4. `sim_ship_cr_v15.3.augmentation.json` 保存了 seed、噪声 profile、copy count 与分区摘要，但没有
   每行 parent/base-scene 映射。

这条旧链路不能直接套用于 `v16n_2x_noise`：目标版本不同；`v15.3` 与 `v16n` source-train 即使形状
相同，行序标签和数值也不相同；旧脚本的上游 `/data/cxc/dfs-master/dataset_v15` 当前不存在；
而 `v16n_2x_noise` 也没有引用旧 manifest 的证据。

### 4.3 行数关系只支持候选假设，不构成谱系证明

旧 v16n 已保存统计为 2784 行、标签 `-1: 1440, +1: 1344`；当前
`v16n_2x_noise` 的既有 manifest 总计 5568 行、标签 `-1: 2880, +1: 2688`，恰为 2 倍。
当前 train/test 计数又与对这 5568 行做 70/30 分层划分的计数一致。

因此，“先把某个 2784 行总体扩为 2 倍，再做 70/30 stratified split”是**计数一致的候选解释**。
但缺少生成代码、输入 hash、split seed、shuffle permutation 和 row ledger，不能确认：

- 2784 行输入是否就是保存的旧 v16n；
- 原始行是否也被扰动；
- 噪声是在 raw、Z-score 还是信号层加入；
- parent/child 是否可能落入不同输出分区；
- `2x` 是一份 parent 加一份 child，还是两份独立扰动样本；
- 计数一致是否只是另一个流程产生的相同摘要。

本审计不会把这个候选解释升级成事实，也不会据此预设泄漏。

## 5. 字段字典与证据来源

| 字段 | 状态 | 当前可写值/含义 | 证据与限制 |
|:---|:---|:---|:---|
| `dataset_version` | available | `v16n_2x_noise` | 配置、文件名、manifest 一致 |
| `output_split` | available（文件级） | `source_train` 或既有 `source_test` | 由文件边界给出；不等于生成前 partition |
| `source_train_row_index_0based` | available | `0..3896` | 当前文件位置，可复现实验 split；不是 original-row ID |
| `label` | available | `-1/+1` | source-train LIBSVM 首字段 |
| `original_feature_id_1based` | available（列级） | `1..75`，清洗后 65 列 | loader metadata；与行谱系无关 |
| `original_row_id` | **unavailable** | — | SVM 无 qid/comment；无 pre-shuffle ledger；旧 train 精确匹配为 0 |
| `base_scene_id` | **unavailable** | — | 场景键在旧基础构建代码中曾存在，但未写入 SVM/NPZ，且无证据连接到目标版本 |
| `augmentation_parent_id` | **unavailable** | — | 无 row ledger、生成入口或可复现参数 |
| `augmentation_copy_id` | **unavailable** | — | 文件名中的 `2x` 不能替代每行 copy ID |
| `augmentation_type` | **unavailable** | — | 现存多个噪声 helper/旧 generator；没有目标版本归因证据 |
| `augmentation_parameters` | **unavailable** | — | noise scale、swap probability、profile、seed 均未冻结 |
| `pre_shuffle_row_index` | **unavailable** | — | 无 permutation |
| `split_seed / split_indices` | **unavailable** | — | 只能看到最终文件边界和摘要计数 |
| `sea_state` / `wave` | **unavailable** | — | 旧文件名解析有 `wave`，但字段被丢弃且未连接到目标行 |
| `grazing_angle` | **unavailable** | — | 旧文件名解析有 `ang`，但字段被丢弃且未连接到目标行；单位也未在目标 manifest 冻结 |
| `SCR` | **unavailable** | — | 旧文件名解析有 `scr`，但字段被丢弃且未连接到目标行；单位/定义未在目标 manifest 冻结 |
| `group_id_for_validation` | **unavailable** | — | 不允许从行序、相似度或聚类代造 |

因此没有生成 `source-train-row-to-group-env.csv` 或等价映射。当前唯一可验证的行表会只包含
`source_train_row_index_0based` 与 `label`，它不回答真实 group/env 问题，生成该文件只会造成“已恢复
映射”的错误印象。

## 6. 数据质量发现与风险

### Finding A：行级谱系主键缺失

- **证据：** source-train SVM 无 qid/注释；manifest 只有文件级 hash、行数、类别数和列清洗；
  所有 3897 行均不同；没有 target-version row ledger。
- **影响：** 无法验证增强父子完整性、orphan child、多 parent、每 parent copy count 或 group-disjoint split。
- **严重度：High；置信度：High。**
- **最小修复：** 提供覆盖 train/test 的签名 row-lineage ledger，并与冻结生成 manifest/hash 对齐。

### Finding B：环境字段在目标数据中不可连接

- **证据：** 旧基础生成代码从原始路径解析 `ang/wave/scr/scene_sid`，但只写 `X/y`；目标 SVM 与
  manifest 不含这些列，也没有外部 scene catalog 到目标行的 join key。
- **影响：** 无法定义预冻结环境分层、检查环境覆盖或评价最差环境；任务 7 不能准入。
- **严重度：High；置信度：High。**
- **最小修复：** 提供稳定 `base_scene_id` 和以该键连接的 environment catalog，并给出字段定义、单位与
  获取时点。

### Finding C：增强与最终 split 的先后关系未证实

- **证据：** 总计数与“2 倍后 70/30 分层重划分”一致，但实际 generator、seed 和 permutation 缺失。
- **影响：** 不能测量或排除同源副本跨 source split/CV fold；风险存在但事实状态未知。
- **严重度：High；置信度：High（对‘不可判定’），Low（对任何具体生成假设）。**
- **最小修复：** 生成 manifest + pre/post-shuffle row ledger + split indices；仅有总计数不够。

### Finding D：当前实验中的环境可用性不足

- **证据：** loader 对 radar-ship 返回 `groups=None`，选择器得到特征矩阵、标签、列名和文件级
  metadata；没有海况、擦地角或 SCR 的显式输入。
- **影响：** 在当前实际特征选择流程中，环境变量**没有作为已知显式变量提供**。真实部署/推理时这些
  量能否在选择前获得，现有代码和文档均未建立，因此标为**未确认**，不能假定可用。
- **严重度：High（针对任务 7 准入）；置信度：High（当前流程缺失），Unknown（部署可用性）。**

## 7. 真实 group-aware 验证与任务 7 判定

| 判定项 | 结果 | 理由 |
|:---|:---|:---|
| source-train 内同源增强可保持同 fold | 不可验证 | parent/base scene unavailable |
| `StratifiedGroupKFold` 的真实 group | 不可构造 | 禁止用行序、相似度或聚类替代 provenance |
| row/group split 小型敏感性诊断 | **未运行** | 前置条件失败；不创建协议或结果目录 |
| 环境分层覆盖/每类每环境 base-scene 数 | 不可计算 | env 与 base scene 均 unavailable |
| 环境变量在现有选择时显式已知 | 否 | selector 输入无环境字段 |
| 环境变量在真实推理前可获得 | 未确认 | 无采集接口、时间语义或部署说明 |
| 任务 7 数据准入 | **NO-GO / blocked by data prerequisites** | 未满足 base-scene/group/env 与推理可用性条件 |

这里的 NO-GO 只针对任务 7 的**数据前提**，不是环境条件化方法或 RL 算法的性能 NO-GO。

## 8. 最小补充材料

若要把 3A 从“不可恢复”推进到可执行 group-aware 诊断，最小充分交付为：

1. **目标版本生成 manifest**：明确实际入口、代码 commit/SHA-256、输入文件及 hash、完整 CLI/config、
   noise 模型与强度、随机 seed、copy count、标准化 fit scope、shuffle 和 split 规则。
2. **全输出 row-lineage ledger**：每个 train/test 输出行至少包含
   `dataset_version, output_split, output_row_index, original_row_id, base_scene_id,
   augmentation_parent_id, augmentation_copy_id, augmentation_type`。ledger 必须在 shuffle/split 后可直接
   对齐最终文件，而不是只有汇总计数。
3. **场景环境 catalog**：以 `base_scene_id` 为唯一键，至少包含标签/目标类型、海况（或 wave 的明确定义
   与单位）、擦地角（定义与单位）、SCR（定义与单位）、原始文件/scene seed。
4. **可验证连接**：manifest 记录 ledger/catalog 的 hash，并提供确定性的生成或验证脚本；如最终文件经过
   重写，需有不依赖近邻匹配的稳定 row ID 或 per-row fingerprint。
5. **推理可用性说明**：逐项说明海况、擦地角、SCR 在做特征选择前是否可测、由哪个传感器/系统提供、
   是否由待选择特征事后估计。缺少这一项时，任务 7 仍不得假定环境已知。

如果无法找回历史 generator，最小可接受替代不是从现有特征反推 group，而是从保留原始场景键的新
generator 重新发布一个版本化数据集；在生成前冻结上述 manifest、scene seeds、参数分布与 split-by-
base-scene 规则，并只生成一次正式批次。

## 9. 证据清单与可复核 hash

| 证据 | SHA-256 / Git identity | 作用 |
|:---|:---|:---|
| `sim_ship_cr_v16n_2x_noise.train.svm` | `2191fdc297aa49ce6a700df956ec44f68dc13e4c737937b31167543a203d22bb` | 本次允许读取的目标 source-train |
| `dataset/augment_features.py` | `e769061ba79e022adaf857c98e34f481edb069219859a7c36970a99082ace038` | 候选 helper；无目标版本调用证据 |
| `dataset/test.ipynb` | `ad1e1fe332b13d5fe4fad7491996dde67c57d4e3fd9a6a464e6908dbb61dad2b` | 生成后加载/分析，不是生成入口 |
| `buildv15sim/build_sim_v15.py` | `fdb6acfc61b591b45e9dc5d458d5cdf0208a45f08a4abd1470d8b76cd44844de` | 旧场景键/env 解析与字段丢失证据 |
| `buildv15sim/generate_v15_3.py` | `ba28451300e8d44c828ca3bda11f3a9637f6e0ed57f771476dbddf05877237a3` | 旧版本增强/shuffle/标准化对照，不归因于目标版本 |
| `experiments/v16n_2x_noise/seed-42/marlfs/manifest.json` | `49de7bbf895f796a7df3e27a6a237fe498b21bfa956cd0d507fb82c6dbe07813` | 目标文件摘要、类别数与历史 hash；无行谱系 |
| 当前仓库 HEAD | `5ea42e571d978040b4b94ed7d34842cc18859b34` | 审计代码/文档快照；工作树非 clean |

## 10. 最终决策

**任务 3A 结论：数据文件可识别，真实行谱系与场景/环境分组不可恢复。**

- source-train 当前行号可以复现实验，但不能作为真实 group；
- 没有证据支持生成可靠 row→group/env 映射，因此映射文件不生成；
- 没有可靠 group，故不拟造分组实验、不查看新的性能结果；
- 任务 7 继续 blocked，直至补齐 base-scene/group/env、同源不跨 fold 的可验证关系和推理时环境可用性；
- 旧 4C 继续 NO-GO；旧静态 RL、4D、5A、5B 和原任务 6 不自动重启。
