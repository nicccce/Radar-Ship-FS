# Experiment Log

本日志记录 Radar-Ship-FS 的代码来源、实验命令、数据版本、随机种子、输出位置和阶段性结论。每次实验完成后请追加一条记录，保证结果可以复查。

## 固定实验约定

- 代码基底：`Yigal-Meshulam/interactive-rl-feature-selection`
- 导入 commit：`f777b4d3e8dd4b89869efd94f28afb7128fa7617`
- 训练期 RL feedback：保留上游 `DecisionTreeProbe`
- 最终主评价：已新增独立 `StandardScaler + LogisticRegression` scorer
- 已完成基础基线：All Features、固定 `k=27` 的 Mutual Information KBest
- 后续 RL 矩阵：MARLFS、Full-IRFS-fixed、Full-IRFS-trained-GCN；已有连续可微 baseline 不重复实现
- 雷达数据切分：官方测试文件始终完整保留；无须验证的基础方法使用完整源训练文件，RL 方法只在源训练文件内部划分训练/奖励验证集

## 记录模板

| 日期 | 阶段 | 数据版本 | 命令 | 随机种子 | 状态编码 | 输出位置 | 结论 / 问题 |
|---|---|---|---|---|---|---|---|
| YYYY-MM-DD |  |  |  |  |  |  |  |

## 2026-07-19 初始化

| 日期 | 阶段 | 数据版本 | 命令 | 随机种子 | 状态编码 | 输出位置 | 结论 / 问题 |
|---|---|---|---|---|---|---|---|
| 2026-07-19 | 阶段 0：工程初始化 | N/A | `git pull upstream main` | N/A | N/A | N/A | 已将 Yigal 的 IRFS/MARLFS 复现代码导入 `Radar-Ship-FS`，保留 `origin` 指向本项目仓库，新增数据、实验、结果、日志和配置目录说明。 |

## 待办

- 已完成：在 WDBC 上运行 `python src/run_irfs.py --dataset wdbc --seeds 42 --state-encoder fixed`，确认完整流程和产物链可运行。
- 已完成：接入雷达舰船 SVM-light 数据适配层，并保证清理规则仅由源训练文件拟合。
- 已完成：新增与训练期 Decision Tree probe 分离的最终 Logistic Regression scorer，并用于 All Features / MI-KBest。
- 待完成：将 RL 方法接入统一最终 LR，并运行 MARLFS、Full-IRFS-fixed、Full-IRFS-trained-GCN 多种子实验。

## 2026-07-19 dl-lab 依赖与补丁记录

### conda 环境

- 环境名称：`dl-lab`
- Python 解释器：`/root/miniconda/envs/dl-lab/bin/python`
- 处理策略：复用该环境中已有的 CUDA torch，只安装缺失依赖，避免重新下载/替换 torch。

### dl-lab 当前依赖版本

| 包 | 版本 |
|---|---|
| Python | 3.10.20 |
| torch | 1.11.0+cu113 |
| numpy | 1.26.4 |
| pandas | 2.3.3 |
| scikit-learn | 1.7.2 |
| scipy | 1.15.3 |
| pytest | 9.1.1 |
| ruff | 0.15.22 |
| mrmr-selection | 0.2.8 |
| polars | 1.42.1 |
| category-encoders | 2.8.1 |
| statsmodels | 0.14.6 |

### 安装方式

为避免重装 `dl-lab` 中已有的 CUDA torch，本次没有执行完整 `requirements.lock` 安装，而是只补齐缺失运行依赖：

```bash
conda run -n dl-lab python -m pip install pandas scikit-learn pytest ruff mrmr-selection
```

### 代码补丁

- `src/engine/policy.py`：为 torch 1.11 增加权重初始化兼容路径。新版 torch 支持 `generator=` 参数；旧版 torch 不支持时，代码会临时保存 CPU RNG 状态、按特征 seed 初始化、再恢复 RNG 状态，以保持确定性和不污染全局 RNG 的约束。
- `src/methods/l1.py`：显式设置 `LogisticRegression(penalty="l1", solver="liblinear")`。上游代码文档写的是 L1/LASSO，但实际参数漏写 `penalty="l1"`，在 scikit-learn 中会退回默认 L2，导致 L1 baseline 不再按惩罚强度改变子集大小。
- `pyproject.toml` / `README.md`：将项目 Python 门槛调整为 `>=3.10`，并补充 `dl-lab` 复用已有 torch 的安装说明。

### 验证记录

| 日期 | 阶段 | 数据版本 | 命令 | 随机种子 | 状态编码 | 输出位置 | 结论 / 问题 |
|---|---|---|---|---|---|---|---|
| 2026-07-19 | 阶段 0：入口验证 | WDBC builtin | `conda run -n dl-lab python src/run_irfs.py --help` | N/A | N/A | N/A | 通过，CLI 参数可正常加载。 |
| 2026-07-19 | 阶段 0：测试验证 | N/A | `conda run -n dl-lab python -m pytest -q` | N/A | N/A | N/A | 通过，`71 passed in 48.11s`。 |
| 2026-07-19 | 阶段 0：静态检查 | N/A | `conda run -n dl-lab python -m ruff check .` | N/A | N/A | N/A | 通过，`All checks passed!`。 |
| 2026-07-19 | 阶段 0：格式检查 | N/A | `conda run -n dl-lab python -m ruff format --check .` | N/A | N/A | N/A | 初次检查提示 `src/data/loader.py` 需格式化；执行 `ruff format src/data/loader.py` 后复查通过。 |
| 2026-07-19 | 阶段 0：训练流程 | WDBC builtin | `conda run -n dl-lab python src/run_irfs.py --dataset wdbc --seeds 42 --state-encoder fixed` | 42 | fixed | N/A | 用户确认不用跑训练，命令已手动中断；未生成实验结果文件。 |

## 2026-07-19 阶段 0 完整运行结果

### 运行配置

```bash
conda run --no-capture-output -n dl-lab \
  python src/run_irfs.py --dataset wdbc --seeds 42 --state-encoder fixed
```

- 数据：scikit-learn 内置 WDBC，569 个样本、30 个特征、2 个类别；
- 划分：训练集 364、验证集 91、测试集 114，分层随机切分；
- 随机种子：42；
- RL 探索预算：每个强化方法 250 步；
- 状态编码：`fixed`，作用于 `no_trainer` 和 `full_irfs`；`marlfs` 按设计使用最小 relevance/redundancy 状态；
- 运行设备：CPU。DQN、联合学习器和可训练 GCN 在当前源码中均显式固定为 CPU，训练期的 scikit-learn 决策树也运行在 CPU；
- 本轮评分器：统一 `DecisionTreeProbe`。路线计划中的最终 `StandardScaler + LogisticRegression` scorer 尚未接入，因此本表仅用于阶段 0 可运行性验收。

### 单种子结果

| 方法 | 类型 | 选择特征数 | 验证 Accuracy | 测试 Accuracy |
|---|---|---:|---:|---:|
| `relevance_topk` | 传统 Filter | 15 | 0.9451 | 0.9035 |
| `dt_rfe` | 传统 Wrapper | 15 | 0.9121 | 0.9737 |
| `mrmr` | 传统 Filter | 15 | 0.9121 | 0.9474 |
| `l1` | 传统 Embedded | 13 | 0.9231 | 0.9561 |
| `marlfs` | 多智能体 RL baseline | 14 | 0.9670 | 0.9298 |
| `no_trainer` | 无 Trainer 的 IRFS 消融 | 22 | 0.9451 | 0.9474 |
| `full_irfs` | Hybrid Teaching IRFS | 22 | 0.9451 | 0.9561 |

### 产物核验

- `experiments/wdbc/seed-42/selection.json`：包含有效配置、7 种方法的非空 `selected` 子集、验证分数和 RL 逐步轨迹；
- `experiments/wdbc/seed-42/test.json`：包含 7 种方法的子集规模、验证分数和独立测试分数；
- `experiments/wdbc/aggregate.json`：包含 seed 42 的逐方法聚合统计。

进程正常退出，阶段 0 的“仓库可运行且选择—验证—测试—聚合产物链完整”验收通过。以上只有一个随机种子，聚合标准差为 0，不能用于判断方法优劣。`dt_rfe` 的验证/测试差距来自小样本切分、未剪枝决策树的高方差，以及零重要性平局时按最低特征编号淘汰的实现偏置；正式结论必须基于多随机种子和统一 LR 最终评价。

## 2026-07-20 阶段 1：雷达数据接入与统一最终 LR

### 主要代码变更

| 文件 | 当前职责 |
|---|---|
| `src/data/loader.py` | 注册 `dataset="radar_ship"`，读取两份 SVM-light 文件；只在源训练文件上识别常数列和完全重复列；返回官方测试行下标及数据指纹、特征映射等元数据。 |
| `src/data/splitter.py` | 识别 `predefined_test_indices`，完整保留源测试文件；需要验证集时只切分源训练文件，并把 loader 元数据传给各分区。 |
| `src/harness/lr_final.py` | 对已经冻结的特征子集执行统一 `StandardScaler + LogisticRegression`，计算训练/测试 Accuracy 及辅助指标；不包含特征选择逻辑。 |
| `src/run_basic_baselines.py` | 代码内固定实验配置，使用完整源训练文件运行 All Features 和 MI-KBest，再调用 `lr_final` 统一评价并保存逐种子/聚合产物。 |
| `tests/test_data.py` | 验证清理规则只由训练数据拟合，以及官方测试文件不会混入训练/验证分区。 |
| `tests/test_basic_baselines.py` | 验证无验证集划分、MI 固定规模与同种子可复现、最终 LR 二分类指标输出。 |

### 数据加载与清理结果

- 源文件：`sim_ship_cr_v10.train.svm`、`sim_ship_cr_v10.test.svm`，读取时显式指定原始维数 75；
- 训练文件 SHA-256：`3e8387c59bb0064d1f165f9c17a26b5710b4f52286b00353368af8808fc764d2`；
- 测试文件 SHA-256：`9cb64e671f266aa12c1f0364c4016aff6474b0206787504210a56492dd78edf8`；
- 源训练集：1843 行，标签分布 `-1: 477`、`+1: 1366`；
- 源测试集：461 行，标签分布 `-1: 99`、`+1: 362`；
- 删除 18 个训练集常数特征：`[2, 3, 5, 6, 7, 8, 10, 12, 13, 47, 48, 52, 57, 58, 60, 62, 70, 75]`；
- 删除 3 个训练集完全重复特征：`27 -> 23`、`38 -> 37`、`71 -> 69`；
- 最终得到 `X.shape=(2304, 54)`、`X.dtype=float32`、`y.dtype=int64`；
- 最终保留的原始特征编号：`[1, 4, 9, 11, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 39, 40, 41, 42, 43, 44, 45, 46, 49, 50, 51, 53, 54, 55, 56, 59, 61, 63, 64, 65, 66, 67, 68, 69, 72, 73, 74]`。

常数和重复列均只根据源训练文件判断，再把同一个列掩码应用到源测试文件。loader 为统一接口临时纵向拼接两部分数据，但用 `predefined_test_indices=1843..2303` 保留原始边界；这不是把测试数据并入训练。

### data.loader 使用说明

从项目根目录运行时：

```python
from config import load_config
from data.loader import load

config = load_config(
    {
        "dataset": "radar_ship",
        "data_dir": "../dataset",
    }
)
dataset = load(config)

X = dataset.X
y = dataset.y
official_test_indices = dataset.predefined_test_indices
original_feature_ids = dataset.metadata["final_feature_ids"]
```

`dataset.X` 的列号是清理后的 0 起始下标；报告特征时应通过 `metadata["final_feature_ids"]` 映射回原始 SVM-light 的 1 起始编号。loader 不执行标准化，标准化由最终 LR 的 Pipeline 在训练数据上拟合。

基础方法不需要奖励验证集，`run_basic_baselines.py` 会将 `predefined_test_indices` 之外的 1843 行全部用于选择和最终拟合。RL 方法需要奖励验证集时使用：

```python
from data.splitter import make_split
from rng import init_rng

config = load_config(
    {
        "dataset": "radar_ship",
        "data_dir": "../dataset",
        "validation_fraction": 0.25,
    }
)
dataset = load(config)
split = make_split(dataset, config, init_rng(config.seeds[0]))
```

该配置得到训练 1382 行、奖励验证 461 行、官方最终测试 461 行；最后一部分始终等于完整源测试文件。

### lr_final 使用说明

`score_frozen_subset_with_lr` 的输入 `subset` 必须是清理后矩阵的 0 起始列号，且必须在调用前完成选择并冻结。示例：

```python
from harness.lr_final import score_frozen_subset_with_lr

metrics = score_frozen_subset_with_lr(
    X_train,
    y_train,
    X_test,
    y_test,
    subset=selected_clean_indices,
    C=1.0,
    solver="liblinear",
    max_iter=5000,
    class_weight="balanced",
    random_state=42,
)
```

内部 Pipeline 只在 `X_train[:, subset]` 上拟合 `StandardScaler` 和 LR，再用相同变换评价 `X_test[:, subset]`。返回字段为 `train_accuracy`、`test_accuracy`、`balanced_accuracy`、`precision`、`recall`、`f1`、`f1_macro`、`roc_auc`、`confusion_matrix` 和 `positive_label`；本项目主指标仍是普通测试 Accuracy，其余指标用于类别不平衡和阈值表现诊断。

当前 `run_basic_baselines.py` 已直接使用该 scorer。上游 `run_irfs.py` 的 RL 训练反馈仍保留 `DecisionTreeProbe`，且尚未接入该最终 LR：后续 RL 实验应先用训练/验证完成选择，冻结子集后合并训练和验证数据重新拟合最终 LR，再评价官方测试集。不得使用测试集选择 `k`、`C`、缩放方式或 RL checkpoint。

## 2026-07-20 雷达舰船基础 LR baseline

### 协议

- 入口：`conda run --no-capture-output -n dl-lab python src/run_basic_baselines.py`；
- 实验参数只在 `src/run_basic_baselines.py` 顶部设置，本入口不接收命令行实验参数；
- 数据：源训练文件 1843 行、源测试文件 461 行，清理后 54 维；
- 验证集：不使用。All Features 无选择超参数；KBest 的 `k=27` 在运行前固定，因此完整源训练文件同时用于 MI 排名和最终 LR 拟合；
- 选择器：All Features 固定返回全部 54 列；KBest 使用 `mutual_info_classif`，仅在 1843 行源训练集上计算排名；
- 最终评分器：所有方法统一使用 `StandardScaler + LogisticRegression`；
- LR：`C=1.0`、`solver="liblinear"`、`max_iter=5000`、`class_weight="balanced"`；
- 主指标：普通测试 Accuracy；Balanced Accuracy、F1、Macro-F1、ROC-AUC 和混淆矩阵作为辅助诊断；
- 随机种子：42、43、44、45、46；在这次正式 baseline 执行中，测试集只做冻结子集的最终报告，不用于选 `k` 或 LR 参数。

### 聚合结果

| 方法 | 特征数 | 压缩率 | 训练 Accuracy | 测试 Accuracy | Balanced Accuracy | F1 | Macro-F1 | ROC-AUC | 选择 Jaccard |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All Features + LR | 54 | 0% | 0.9609 ± 0.0000 | 0.9479 ± 0.0000 | 0.9669 ± 0.0000 | 0.9657 ± 0.0000 | 0.9288 ± 0.0000 | 0.9971 ± 0.0000 | 1.0000 ± 0.0000 |
| KBest/MI + LR | 27 | 50% | 0.9352 ± 0.0005 | 0.9262 ± 0.0000 | 0.9464 ± 0.0016 | 0.9510 ± 0.0001 | 0.9011 ± 0.0003 | 0.9892 ± 0.0001 | 0.9714 ± 0.0369 |

All Features 在测试集上正确 437/461 条；KBest/MI 正确 427/461 条。KBest 用一半特征换取约 2.17 个百分点普通 Accuracy 的下降。All Features 的训练/测试 Accuracy 差约 1.30 个百分点；没有出现测试明显高于训练的异常。

All Features 的正类 `+1` Precision 为 1.0000、Recall 为 0.9337。KBest/MI 的五种子平均 Precision 为 0.9946、Recall 为 0.9110。由于测试集类别分布不均衡，Balanced Accuracy 和 Macro-F1 只作为辅助指标，不替代主指标普通 Accuracy。

### 特征集合、混淆矩阵与稳定性

- All Features 的原始特征编号就是清理后保留的全部 54 个编号，五个种子的子集和预测完全相同；
- seed 42、43、44、46 的 MI 子集相同：`[1, 4, 9, 11, 14, 15, 16, 17, 18, 19, 20, 23, 25, 26, 28, 29, 31, 34, 49, 50, 51, 53, 54, 55, 56, 59, 69]`；
- seed 45 在 MI 排名边界处用原始特征 32 替换了特征 29，因此两类 MI 子集的 Jaccard 为 `26/28=0.9286`，全部种子两两比较的平均 Jaccard 为 0.9714；
- 混淆矩阵的行/列顺序均为 `[-1, +1]`；
- All Features：`[[99, 0], [24, 338]]`；
- KBest/MI seed 42、43、44、46：`[[97, 2], [32, 330]]`；
- KBest/MI seed 45：`[[98, 1], [33, 329]]`。虽然错误类别构成略有变化，但总错误数仍为 34，因此五个种子的普通测试 Accuracy 相同。

### 产物

- `experiments/radar_ship_basic_lr/seed-<seed>/results.json`：逐种子配置、MI 分数、原始特征编号、完整指标和耗时；
- `experiments/radar_ship_basic_lr/aggregate.json` / `aggregate.csv`：跨种子聚合；
- `results/tables/radar_ship_basic_lr_per_seed.csv`：后续画图用逐种子表；
- `results/tables/radar_ship_basic_lr_aggregate.csv`：主表。

### 事后结果审计（不是调参结果）

为解释 All Features 94.79% 与外部结果的差异，在正式 baseline 完成后做了只读诊断：

- 源训练集与源测试集之间的完全重复样本数为 0，同标签重复数为 0、冲突标签重复数为 0；当前没有发现“相同行泄漏”证据；
- 使用当前 `StandardScaler + class_weight="balanced"` 协议时，All Features 对 `C` 非常敏感：

| C | 测试 Accuracy |
|---:|---:|
| 0.001 | 0.7093 |
| 0.01 | 0.8243 |
| 0.1 | 0.9002 |
| 1.0 | 0.9479 |
| 10 | 0.9783 |
| 100 | 0.9870 |

同一数据上，`StandardScaler + class_weight=None + C=1` 得到 0.9631，`MinMaxScaler + class_weight="balanced" + C=1` 得到 0.8764。这说明分类器正则化、缩放方式和类别权重足以造成数个百分点差异，不能在协议不同的情况下直接比较 Accuracy。

正式 baseline 的 `C=1.0` 在上述审计前已经固定，表中的 0.9479 仍是原先预设配置的结果；但事后审计已经多次查看官方测试集，所以这些 `C`/预处理对照只能解释敏感性，不能据此选择后续参数或宣称 0.9870 是调优结果。后续任何 `C`、缩放方式或类别权重选择都必须在源训练集内部交叉验证，官方测试集只用于锁定协议后的报告。

### 验证记录

| 日期 | 检查 | 命令 | 结果 |
|---|---|---|---|
| 2026-07-20 | 全量测试 | `conda run --no-capture-output -n dl-lab python -m pytest -q` | 通过，`76 passed in 85.78s`。 |
| 2026-07-20 | 基础 baseline 单测 | `conda run --no-capture-output -n dl-lab python -m pytest tests/test_basic_baselines.py -q` | 通过，`3 passed in 1.21s`。 |
| 2026-07-20 | 静态检查 | `conda run --no-capture-output -n dl-lab python -m ruff check .` | 通过，`All checks passed!`。 |
| 2026-07-20 | 格式检查 | `conda run --no-capture-output -n dl-lab python -m ruff format --check .` | 通过，全部文件格式符合要求。 |

其中数据测试明确覆盖“测试集独有变化不能挽救训练集常数列”和“完整官方测试文件保持隔离”；baseline 测试覆盖“无验证集时全部非测试行返回训练”“固定 k 的 MI 同种子可复现”和“统一 LR 能返回合法二分类指标”。
