# Experiment Log

本日志记录 Radar-Ship-FS 的代码来源、实验命令、数据版本、随机种子、输出位置和阶段性结论。每次实验完成后请追加一条记录，保证结果可以复查。

## 固定实验约定

- 代码基底：`Yigal-Meshulam/interactive-rl-feature-selection`
- 导入 commit：`f777b4d3e8dd4b89869efd94f28afb7128fa7617`
- 训练期 RL feedback：1843 行 development 内固定分层 5 折 Decision Tree 平均准确率
- 最终主评价：全部 development 拟合 Decision Tree，外层 461 行 test 评价
- 已完成基础基线：All Features、固定 `k=27` 的 Mutual Information KBest
- 已完成 RL 矩阵：MARLFS、Full-IRFS-fixed、Full-IRFS-trained-GCN；已有连续可微 baseline 不重复实现
- 当前阶段 2 划分：合并两份源文件的 2304 行，每个 seed 外层分层随机划分 1843 development / 461 test；RL 在 development 内做 5 折选择

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
- 已完成：合并两份雷达数据后重新划分，并完成三种 RL 的内部 5 折 DT 选择和外层 held-out DT test。

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

## 2026-07-21 阶段 2：内部 5 折 DT 奖励的 RL 特征选择与最终 DT/LR test

### 操作与实验协议

1. `radar_ship` 加载器读取并合并 `sim_ship_cr_v10.train.svm` 和 `sim_ship_cr_v10.test.svm`，共 2304 行。
2. 每个 seed 对 2304 行做一次外层分层随机划分：1843 行 development、461 行 test，即约 `80% / 20%`。
3. 不再单独保留 reward validation。RL 只在 1843 行 development 内做固定分层 5 折：每个候选特征集训练 5 棵 Decision Tree，以 5 个留出折 Accuracy 的均值作为选择分数。
4. 同一 seed 的三种 RL 方法复用相同的 development、test 和 5 个内部折；选择阶段不释放 test。
5. 每种 RL 运行 250 步；Hybrid Teaching 的零基区间为 relevance `[0,83)`、DT-importance `[83,166)`、无 trainer `[166,250)`。
6. 最终子集先比较内部 5 折 DT 平均 Accuracy；在 `1e-12` 容差内同分时，选择特征数更少的子集。
7. RL 完成后保存筛选出的特征编号。独立 DT 入口读取这些编号，不重新运行 RL；用全部 1843 行 development 拟合新 Decision Tree，只在 461 行 test 上评价。
8. 最终 DT 同时评价 All Features-54、KBest-27、MARLFS、Full-IRFS-fixed、Full-IRFS-trained-GCN，以及与每个 seed 的 Full-IRFS-fixed 同规模的 KBest。KBest 的 Mutual Information 只在 development 上拟合。

固定配置：

- 环境：`dl-lab`，CPU；
- seeds：42、43、44、45、46；
- RL 方法：MARLFS、Full-IRFS-fixed、Full-IRFS-trained-GCN；
- 每方法每 seed：250 步，共 15 次选择、3750 个 RL step；
- 内部交叉验证：`StratifiedKFold(n_splits=5, shuffle=True)`，随机状态来自该 seed 的统一 RNG；
- KBest：Mutual Information，`k=27`；
- 参数只在 `src/stage2_rl_config.py` 设置，不通过命令行切换；
- 最终主评价使用 DT；另用相同的已保存特征补充独立 LR test，LR 不参与选择或调参。

执行命令：

```bash
conda run --no-capture-output -n dl-lab python src/run_stage2_rl_selection.py
conda run --no-capture-output -n dl-lab python src/run_stage2_dt_test.py
conda run --no-capture-output -n dl-lab python src/run_stage2_rl_final_lr.py
```

### RL 内部 5 折选择结果

每个单元格为“筛选出的特征数 / 最佳内部 5 折 DT Accuracy”。

| Seed | MARLFS | Full-IRFS-fixed | Full-IRFS-trained-GCN |
|---:|---:|---:|---:|
| 42 | 25 / 0.9729 | 36 / 0.9685 | 36 / 0.9707 |
| 43 | 29 / 0.9729 | 30 / 0.9712 | 34 / 0.9712 |
| 44 | 23 / 0.9685 | 41 / 0.9691 | 35 / 0.9712 |
| 45 | 28 / 0.9707 | 32 / 0.9685 | 38 / 0.9685 |
| 46 | 31 / 0.9750 | 35 / 0.9756 | 39 / 0.9772 |

| 方法 | 特征数 | 压缩率 | 最佳内部 5 折 DT Accuracy | 最佳子集首次出现步数 | 末 25 步 Accuracy | 选择 Jaccard | 搜索耗时/seed |
|---|---:|---:|---:|---:|---:|---:|---:|
| MARLFS | 27.2 ± 3.2 | 49.63% | 0.9720 ± 0.0025 | 128.6 | 0.9538 | 0.3544 | 129.0 s |
| Full-IRFS-fixed | 34.8 ± 4.2 | 35.56% | 0.9706 ± 0.0030 | 161.0 | 0.9562 | 0.5100 | 139.8 s |
| Full-IRFS-trained-GCN | 36.4 ± 2.1 | 32.59% | 0.9718 ± 0.0032 | 112.6 | 0.9516 | 0.5564 | 1094.6 s |

15 次选择记录的总耗时约 6817.1 秒。每次选择均保存 250 步轨迹；每步包含 5 折 Accuracy、均值、标准差、最小/最大值、running best、特征数、相邻子集 Jaccard、最终子集 Jaccard 和累计耗时。

### 使用前面筛选出的特征进行最终 DT test

每个单元格为“特征数 / test Accuracy”。每组 DT 都用该 seed 的 1843 行 development 拟合，并在同一 461 行 test 上评分；训练 Accuracy 均为 1.0000。

| Seed | All-54 | KBest-27 | MARLFS | Full-IRFS-fixed | Full-IRFS-trained-GCN | 同规模 KBest |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 54 / 0.9718 | 27 / 0.9718 | 25 / 0.9653 | 36 / 0.9805 | 36 / 0.9740 | 36 / 0.9740 |
| 43 | 54 / 0.9523 | 27 / 0.9783 | 29 / 0.9653 | 30 / 0.9696 | 34 / 0.9761 | 30 / 0.9783 |
| 44 | 54 / 0.9653 | 27 / 0.9675 | 23 / 0.9631 | 41 / 0.9653 | 35 / 0.9610 | 41 / 0.9610 |
| 45 | 54 / 0.9566 | 27 / 0.9610 | 28 / 0.9523 | 32 / 0.9501 | 38 / 0.9588 | 32 / 0.9588 |
| 46 | 54 / 0.9631 | 27 / 0.9610 | 31 / 0.9631 | 35 / 0.9631 | 39 / 0.9675 | 35 / 0.9653 |

| 方法 | 平均特征数 | DT Test Accuracy | 相对 Full-IRFS-fixed | 对 fixed 胜/平/负 |
|---|---:|---:|---:|---:|
| All Features | 54.0 | 0.9618 ± 0.0076 | -0.0039 | 1 / 2 / 2 |
| KBest/MI, k=27 | 27.0 | 0.9679 ± 0.0074 | +0.0022 | 3 / 0 / 2 |
| MARLFS 前面筛选出的特征 | 27.2 ± 3.2 | 0.9618 ± 0.0054 | -0.0039 | 1 / 1 / 3 |
| Full-IRFS-fixed 前面筛选出的特征 | 34.8 ± 4.2 | 0.9657 ± 0.0110 | 0 | 0 / 5 / 0 |
| Full-IRFS-trained-GCN 前面筛选出的特征 | 36.4 ± 2.1 | 0.9675 ± 0.0077 | +0.0017 | 3 / 0 / 2 |
| KBest/MI, k 匹配 fixed | 34.8 ± 4.2 | 0.9675 ± 0.0084 | +0.0017 | 3 / 0 / 2 |

### 使用前面筛选出的特征进行独立 LR test

`run_stage2_rl_final_lr.py` 只读取当前 `selection.json` 中的特征编号，不重新运行 RL，也不修改特征。每个 seed 使用全部 1843 行 development 拟合 `StandardScaler + LogisticRegression(C=1.0, solver="liblinear", max_iter=5000, class_weight="balanced")`，在对应的 461 行 test 上评分；参数由代码预先固定，未用 test 调参。

下表中 All Features 和 KBest-27 是相同外层 development/test 划分下已有的基础 LR 对照；MARLFS、Full-IRFS-fixed 和 Full-IRFS-trained-GCN 使用本次内部 5 折 DT 前面筛选出的特征。每个单元格为“特征数 / LR test Accuracy”。

| Seed | All-54 | KBest-27 | MARLFS | Full-IRFS-fixed | Full-IRFS-trained-GCN |
|---:|---:|---:|---:|---:|---:|
| 42 | 54 / 0.9653 | 27 / 0.9523 | 25 / 0.8850 | 36 / 0.9089 | 36 / 0.9371 |
| 43 | 54 / 0.9566 | 27 / 0.9393 | 29 / 0.9154 | 30 / 0.9328 | 34 / 0.9501 |
| 44 | 54 / 0.9458 | 27 / 0.9328 | 23 / 0.9219 | 41 / 0.9371 | 35 / 0.9393 |
| 45 | 54 / 0.9479 | 27 / 0.9349 | 28 / 0.9241 | 32 / 0.9436 | 38 / 0.9328 |
| 46 | 54 / 0.9479 | 27 / 0.9241 | 31 / 0.9132 | 35 / 0.9436 | 39 / 0.9501 |

| 方法 | 平均特征数 | LR Test Accuracy | Balanced Accuracy | F1 | Macro-F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| All Features | 54.0 | 0.9527 ± 0.0082 | 0.9621 | 0.9677 | 0.9398 | 0.9948 |
| KBest/MI, k=27 | 27.0 | 0.9367 ± 0.0103 | 0.9537 | 0.9561 | 0.9212 | 0.9915 |
| MARLFS 前面筛选出的特征 | 27.2 ± 3.2 | 0.9119 ± 0.0157 | 0.9280 | 0.9384 | 0.8918 | 0.9790 |
| Full-IRFS-fixed 前面筛选出的特征 | 34.8 ± 4.2 | 0.9332 ± 0.0143 | 0.9508 | 0.9536 | 0.9172 | 0.9891 |
| Full-IRFS-trained-GCN 前面筛选出的特征 | 36.4 ± 2.1 | 0.9419 ± 0.0079 | 0.9549 | 0.9600 | 0.9269 | 0.9920 |

### 保存与检查

- `experiments/radar_ship_stage2_rl_selection/`：15 个 `selection.json`、15 份逐步轨迹 CSV、3750 行全轨迹和 750 行跨 seed 轨迹聚合；
- `experiments/radar_ship_stage2_dt_test/`：5 个 seed 的逐项 DT test 结果、逐 seed 表和聚合表；
- `results/tables/radar_ship_stage2_rl_selection_*.csv`：RL 选择逐 seed 与聚合表；
- `results/tables/radar_ship_stage2_rl_trajectory_aggregate.csv`：后续训练稳定性绘图表；
- `results/tables/radar_ship_stage2_dt_test_*.csv`：最终 DT test 逐 seed 与聚合表；
- `experiments/radar_ship_stage2_rl_final_lr/`：当前选择协议下的 15 个独立 LR JSON、逐 seed 表和聚合表；
- `results/tables/radar_ship_stage2_rl_final_lr_*.csv`：补充 LR 的逐 seed 与聚合绘图表。

检查数据：

- 15 个选择 JSON 均含 250 步，且 `held_out_random_test_accessed=false`、`official_test_accessed=false`、`lr_final_called=false`；
- 每个 seed 的 5 个内部折中，拟合行与留出行互斥；5 个留出折各覆盖 development 中的每行一次；
- 最终 DT 使用的三组 RL 特征编号与对应 `selection.json` 逐项一致；
- 15 个 LR JSON 的特征编号与对应 `selection.json` 逐项一致，均为 1843 行拟合、461 行 test；
- 每个 seed 的 1843 行 development 与 461 行 test 互斥并覆盖全部 2304 行；
- 基础 LR 与阶段 2 定向测试：`12 passed in 1.72s`；
- 全量测试：`85 passed in 46.82s`；
- Ruff：`All checks passed!`，`71 files already formatted`。
## 2026-07-21 阶段 2：Full-IRFS-fixed 的 β 扫描

### 配置与隔离协议

- β：0、0.02、0.1、0.5；
- seeds：42、43、44、45（相对主实验少 seed 46）；
- 方法：只跑 Full-IRFS-fixed，fixed 状态编码，每组 250 步、内部 5 折 DT；
- 同一 seed 的四档 β 复用相同 development/test 划分、内部折和 post-split RNG 起点；
- 16 次选择全部完成且逐项通过签名、轨迹长度和 test-sealed 校验后，独立入口才释放 outer test；
- 一键命令：conda run --no-capture-output -n dl-lab python src/run_stage2_beta_sweep.py；
- 16 次选择总耗时约 2187.8 秒（36.5 分钟），随后 DT 验证正常完成。

### 内部 5 折选择汇总

| β | 平均特征数 | 最佳内部 5 折 DT Accuracy | 最终子集平均绝对相关性 | 选择 Jaccard |
|---:|---:|---:|---:|---:|
| 0 | 36.75 ± 4.65 | 0.9706 ± 0.0024 | 0.2461 | 0.5468 |
| 0.02 | 34.50 ± 2.38 | 0.9710 ± 0.0029 | 0.2243 | 0.5179 |
| 0.1 | 35.50 ± 3.51 | 0.9702 ± 0.0018 | 0.2395 | 0.5215 |
| 0.5 | 33.00 ± 2.16 | 0.9712 ± 0.0023 | 0.2262 | 0.4950 |

β 增大后，最终保存子集的相关性并不单调下降。这并非参数未生效：β 会改变 RL 的逐步奖励和搜索轨迹，但最终产物仍按最高内部 5 折 DT Accuracy 选轨迹点，而不是按 Accuracy−β×相关性选择。

### outer-test DT 结果

每个单元格为 test Accuracy；MI-27 与四档 β 使用同一 seed 的 development/test。

| Seed | MI-27 | β=0 | β=0.02 | β=0.1 | β=0.5 |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.9718 | 0.9675 | 0.9631 | 0.9718 | 0.9718 |
| 43 | 0.9783 | 0.9675 | 0.9740 | 0.9761 | 0.9761 |
| 44 | 0.9675 | 0.9761 | 0.9740 | 0.9610 | 0.9610 |
| 45 | 0.9610 | 0.9523 | 0.9588 | 0.9588 | 0.9631 |

| 方法 | 平均特征数 | DT Test Accuracy | 相对 MI-27 | 对 MI-27 胜/平/负 |
|---|---:|---:|---:|---:|
| MI-27 | 27.0 | 0.9696 ± 0.0073 | 0 | 0 / 4 / 0 |
| β=0 | 36.75 | 0.9658 ± 0.0099 | -0.0038 | 1 / 0 / 3 |
| β=0.02 | 34.50 | 0.9675 ± 0.0077 | -0.0022 | 1 / 0 / 3 |
| β=0.1 | 35.50 | 0.9669 ± 0.0084 | -0.0027 | 0 / 1 / 3 |
| β=0.5 | 33.00 | 0.9680 ± 0.0072 | -0.0016 | 1 / 1 / 2 |

结论：本轮四档均未在 4-seed 平均 DT test Accuracy 上击败 MI-27；β=0.5 最接近，但仍少 0.0016，且平均使用 33 个特征，多于 MI-27。仅调 β 不能实现“碾压 top-k”；它主要改变搜索轨迹，未解决训练奖励、最终子集选择规则与目标比较对象之间的对齐问题。4 个 seed 的样本量也不足以支撑显著性结论。

### 产物与检查

- experiments/radar_ship_stage2_beta_sweep_selection/：16 个 selection.json、16 份 trajectory.csv、全轨迹和聚合表；
- experiments/radar_ship_stage2_beta_sweep_dt_test/：4 个 seed 的最终 DT 结果与聚合表；
- results/tables/radar_ship_stage2_beta_sweep_selection_*.csv：选择阶段绘图表；
- results/tables/radar_ship_stage2_beta_sweep_dt_test_*.csv：DT 最终比较表；
- 定向测试：12 passed in 1.59s；
- 新增脚本 Ruff：All checks passed。


## 2026-07-22 阶段 2：Full-IRFS-trained-GCN 的 β 扫描

### 配置与耗时

- β：0、0.02、0.1、0.5；
- seeds：42、43、44、45；
- 方法：Full-IRFS-trained-GCN，每组 250 步、内部 5 折 DT；
- 同一 seed 的四档 β 复用相同 development/test、内部折和 post-split RNG 起点；
- 16 次选择全部完成并通过签名、轨迹长度和 test-sealed 校验后，才执行 outer-test DT；
- 一键命令：conda run --no-capture-output -n dl-lab python src/run_stage2_beta_sweep_trained_gcn.py；
- 16 次选择合计约 16584.3 秒，即 4.61 小时；平均每组约 1036.5 秒。

### 内部 5 折选择汇总

| β | 平均特征数 | 最佳内部 5 折 DT Accuracy | 最终子集平均绝对相关性 | 选择 Jaccard | 耗时/seed |
|---:|---:|---:|---:|---:|---:|
| 0 | 36.50 ± 2.38 | 0.9706 ± 0.0036 | 0.2417 | 0.5821 | 1034.8 s |
| 0.02 | 37.00 ± 3.92 | 0.9711 ± 0.0029 | 0.2452 | 0.5375 | 1036.8 s |
| 0.1 | 38.25 ± 1.50 | 0.9704 ± 0.0017 | 0.2361 | 0.5564 | 1037.2 s |
| 0.5 | 37.00 ± 1.63 | 0.9711 ± 0.0026 | 0.2412 | 0.5875 | 1037.3 s |

相关性和特征数均未随 β 单调下降。trained-GCN 会改变状态表征和搜索轨迹，但最终子集仍按最高 inner-CV DT Accuracy 保存，因此 β 不是可靠的子集规模控制器。

### outer-test DT 结果

| Seed | MI-27 | β=0 | β=0.02 | β=0.1 | β=0.5 |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.9718 | 0.9631 | 0.9740 | 0.9718 | 0.9740 |
| 43 | 0.9783 | 0.9696 | 0.9740 | 0.9718 | 0.9696 |
| 44 | 0.9675 | 0.9631 | 0.9544 | 0.9696 | 0.9610 |
| 45 | 0.9610 | 0.9675 | 0.9588 | 0.9588 | 0.9566 |

| 方法 | 平均特征数 | DT Test Accuracy | 相对 MI-27 | 对 MI-27 胜/平/负 |
|---|---:|---:|---:|---:|
| MI-27 | 27.0 | 0.9696 ± 0.0073 | 0 | 0 / 4 / 0 |
| β=0 | 36.50 | 0.9658 ± 0.0033 | -0.0038 | 1 / 0 / 3 |
| β=0.02 | 37.00 | 0.9653 ± 0.0102 | -0.0043 | 1 / 0 / 3 |
| β=0.1 | 38.25 | 0.9680 ± 0.0062 | -0.0016 | 1 / 1 / 2 |
| β=0.5 | 37.00 | 0.9653 ± 0.0079 | -0.0043 | 1 / 0 / 3 |

结论：trained-GCN 的四档 β 仍均未在 4-seed 平均 DT test Accuracy 上击败 MI-27。最佳 β=0.1 与 fixed 扫描的最佳 β=0.5 同为 0.9680、相对 MI-27 为 -0.0016，但 trained-GCN 平均使用 38.25 个特征，fixed 只用 33 个；trained-GCN 每组约 1037 秒，约为 fixed 的 7.7 倍。该模型没有把额外计算转化为更高 outer-test Accuracy。

### 产物与检查

- experiments/radar_ship_stage2_beta_sweep_gcn_selection/：16 个 selection.json、16 份 trajectory.csv、全轨迹与聚合表；
- experiments/radar_ship_stage2_beta_sweep_gcn_dt_test/：4 个 seed 的最终 DT 结果与聚合表；
- results/tables/radar_ship_stage2_beta_sweep_gcn_selection_*.csv：选择阶段汇总；
- results/tables/radar_ship_stage2_beta_sweep_gcn_dt_test_*.csv：DT 最终比较；
- 定向测试：14 passed in 1.65s；
- Ruff：All checks passed。


## 2026-07-22 阶段 2：Hybrid Teaching 指导比例扫描

### 动机与 advisor 语义

此前所有阶段 2 和 β 扫描均固定使用 83/83/84，即 83 步 MI relevance、83 步 DT importance、84 步撤回指导。本轮固定 fixed 状态编码与 β=0.5，只改变三阶段步数。

需要特别区分“MI relevance 指导”和“MI Top-27”：

1. relevance advisor 只处理上一轮已选择、本轮准备删除的 hesitant 特征；
2. 若该特征 MI 严格高于 assertive 特征 MI 的中位数，advisor 只把该次删除票改回保留；
3. 它不能主动加入当前未选择的 MI Top-27 特征，也不会主动删除低 MI 特征；
4. 它没有 k=27 约束；
5. advisor 的 MI 随机状态从运行 RNG 另行抽取，而最终 MI-27 使用 seed 作为 random_state，两个随机 MI 排名不保证逐项相同。

因此该机制是“单向防止较高 MI 特征过早删除”，不是“让 RL 复制或改进 Top-27”。

### 扫描配置

| 名称 | MI relevance | DT importance | 撤回指导 |
|---|---:|---:|---:|
| thirds_83_83_84 | 83 | 83 | 84 |
| mi_heavy_200_25_25 | 200 | 25 | 25 |
| mi_only_250_0_0 | 250 | 0 | 0 |
| dt_heavy_25_200_25 | 25 | 200 | 25 |

- seeds：42、43、44、45；
- 每组 250 步，共 16 次 fixed 搜索；
- 16 次搜索全部完成并通过 test-sealed 预检后，才统一执行 outer-test DT；
- 一键命令：conda run --no-capture-output -n dl-lab python src/run_stage2_guidance_sweep.py；
- 搜索总耗时约 2207.3 秒，即 36.8 分钟。

### 结果

| 调度 | 平均特征数 | Inner-CV DT | 平均相关性 | DT Test Accuracy | 相对 MI-27 | 胜/平/负 |
|---|---:|---:|---:|---:|---:|---:|
| MI-27 | 27.0 | N/A | N/A | 0.9696 ± 0.0073 | 0 | N/A |
| 83/83/84 | 33.00 ± 2.16 | 0.9712 | 0.2262 | 0.9680 ± 0.0072 | -0.0016 | 1/1/2 |
| 200/25/25 | 38.75 ± 5.38 | 0.9695 | 0.2505 | 0.9653 ± 0.0121 | -0.0043 | 0/1/3 |
| 250/0/0 | 38.75 ± 5.38 | 0.9695 | 0.2505 | 0.9653 ± 0.0121 | -0.0043 | 0/1/3 |
| 25/200/25 | 35.00 ± 4.55 | 0.9702 | 0.2463 | 0.9631 ± 0.0094 | -0.0065 | 0/0/4 |

结论：原三等分仍是四档中最好，但仍未超过 MI-27。延长 MI 指导不仅没有提高 test Accuracy，还把平均特征数由 33 增到 38.75、平均相关性由 0.2262 增到 0.2505。这与 advisor 的单向保留语义一致：指导越久，高 MI hesitant 特征越不容易被删除，但低 MI 特征并不会被对称清除。

“用 Top-K 指导却干不过 Top-K”的直接答案是：当前代码并没有执行 Top-K 集合指导，只使用了 MI 中位数阈值的单向保留规则。若目标是与 MI-27 公平对打，下一步应实现真正的双向 Top-27 advisor 或固定 k=27 的交换动作：主动补入缺失的 MI Top-27、主动移除低排名特征，并让 RL 只优化同规模子集的 DT inner-CV Accuracy。

### 产物与检查

- experiments/radar_ship_stage2_guidance_sweep_selection/：四调度×四 seed 的16个选择 JSON 与轨迹；
- experiments/radar_ship_stage2_guidance_sweep_dt_test/：四调度的 DT 结果与跨调度汇总；
- results/tables/radar_ship_stage2_guidance_sweep_dt_test_aggregate.csv：最终对比表；
- 定向测试：29 passed in 9.35s；
- Ruff：All checks passed。


## 2026-07-22 阶段 2：超预算惩罚扫描（已完成）

### 动机与假设

前一轮指导比例扫描表明，延长 MI/DT advisor 没有解决子集偏大的问题；最佳 fixed 方案仍平均选择
33 个特征，而公平基线 MI-27 固定为 27 个。当前实验改为直接优化这个错位：

```text
J(S) = Accuracy_CV(S) - beta * Corr(S) - lambda * max(0, (|S| - 27) / 27)
```

其中 `beta=0.02`，避免相关性再次淹没 Accuracy；`lambda` 扫描
`{0.01, 0.025, 0.05, 0.1}`。状态编码、DQN、动作空间、250 步预算以及 83/83/84
Hybrid Teaching 调度均保持不变。

### 最终候选协议

1. 初始随机半数子集在雷达数据上恰为 27 个特征，并在第 1 个动作前计算 inner-CV DT Accuracy；
2. 该初始子集与 250 步轨迹共同构成候选池，不占用额外 RL step；
3. 超过 27 个特征的轨迹仍用于奖励和 DQN 学习，但没有资格成为最终输出；
4. 最终只在 `|S| <= 27` 的候选中选择 inner-CV DT Accuracy 最高者，同分选择特征更少者；
5. 4 个 lambda × 4 个 seed 的 16 个选择产物全部完成并通过密封预检后，才允许 outer-test DT；
6. MI-27 仍只在各 seed 的 development 上拟合，并作为配对胜/平/负基线。

### 实现与运行

- 一键命令：`conda run --no-capture-output -n dl-lab python src/run_stage2_budget_sweep.py`；
- 选择产物：`experiments/radar_ship_stage2_budget_sweep_selection/`；
- DT test 产物：`experiments/radar_ship_stage2_budget_sweep_dt_test/`；
- 汇总表前缀：`results/tables/radar_ship_stage2_budget_sweep_*`；
- 断点续跑会校验 beta、lambda、k、完整有效配置、250 步轨迹和初始候选标记；
- 旧的无预算 beta/guidance 产物签名保持兼容，不会因新增默认关闭的配置项而误重跑；
- 全量测试：106 passed；
- 真实雷达数据 1-step 烟测：初始 27、首步 32、最终 27，预算筛选生效；
- Ruff：All checks passed。

- 正式搜索总耗时：2228.2 秒（37.14 分钟）；
- 16 个 selection.json 与 4 个逐 seed DT results.json 均已生成并通过预检。

### 结果

| 方法 | 平均特征数 | Inner-CV DT | 平均相关性 | 产生新预算内轨迹的 seed | DT Test Accuracy | 相对 MI-27 | 胜/平/负 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MI-27 | 27.0 | N/A | N/A | N/A | 0.9696 ± 0.0073 | 0 | N/A |
| lambda=0.01 | 27.0 ± 0.0 | 0.9257 ± 0.0515 | 0.2203 | 2/4 | 0.9241 ± 0.0464 | -0.0456 | 1/0/3 |
| lambda=0.025 | 25.5 ± 1.29 | 0.9607 ± 0.0092 | 0.2178 | 3/4 | 0.9680 ± 0.0045 | -0.0016 | 2/1/1 |
| lambda=0.05 | 26.0 ± 1.15 | 0.9434 ± 0.0446 | 0.2017 | 2/4 | 0.9431 ± 0.0387 | -0.0266 | 1/0/3 |
| lambda=0.1 | 27.0 ± 0.0 | 0.9217 ± 0.0520 | 0.1938 | 1/4 | 0.9176 ± 0.0497 | -0.0521 | 1/0/3 |

最佳 `lambda=0.025` 的逐 seed 配对结果：

| seed | 特征数 | 候选来源 | Inner-CV DT | DT Test | MI-27 Test | 差值 |
|---:|---:|---|---:|---:|---:|---:|
| 42 | 25 | trajectory | 0.9544 | 0.9718 | 0.9718 | 0.0000 |
| 43 | 24 | trajectory | 0.9653 | 0.9631 | 0.9783 | -0.0152 |
| 44 | 26 | trajectory | 0.9517 | 0.9718 | 0.9675 | +0.0043 |
| 45 | 27 | initial | 0.9712 | 0.9653 | 0.9610 | +0.0043 |

### 结论

本轮没有实现“碾压 MI-27”。`lambda=0.025` 是唯一接近成功的配置：平均少 1.5 个特征，
2 胜 1 平 1 负，但 seed 43 的 -0.0152 抵消了两个 +0.0043，最终均值仍低 0.0016。

超预算惩罚确实能产生合格轨迹，但数量过少且不随 lambda 单调增加。最佳档每个 seed 平均只有
1.75 个不同合格候选（其中 1 个是初始子集），只有 3/4 seed 产生了新的预算内轨迹。更大的
lambda 反而明显变差。结合当前个性化奖励实现，整个 J(S) 会再乘 DT importance，基数惩罚被分摊到
各 selected agent，而 deselected agent 仍为零；因此单纯放大 lambda 不能形成稳定的删减信用。

下一步若仍以“稳定超过 MI-27”为目标，不应继续扩大 lambda 网格。更直接的方向是固定预算交换动作
或每步投票后的 k=27 repair，让全部 250 个候选都在同预算空间内竞争；次选是把基数惩罚作为不经
importance 缩放的对称 per-agent 信号。


## 2026-07-22 阶段 2：纯 Decision-Tree Accuracy 奖励控制（已完成）

### 配置

本轮令 `beta=0、lambda=0`，所以 RL 奖励严格等于 development 内 5 折 DT Accuracy。
为保持与 MI-27 的公平比较，仍保留初始 27 特征候选，并只允许 `|S|<=27` 的候选成为最终输出。
其余 DQN、fixed 状态、83/83/84 Hybrid Teaching、250 步预算和 seeds 42–45 不变。

- 一键命令：`conda run --no-capture-output -n dl-lab python src/run_stage2_accuracy_only.py`；
- 4 次选择总耗时：539.6 秒（8.99 分钟）；
- 4 个 selection.json 全部完成并通过密封预检后，才执行 outer-test DT；
- 选择目录：`experiments/radar_ship_stage2_accuracy_only_selection/`；
- DT test 目录：`experiments/radar_ship_stage2_accuracy_only_dt_test/`。

### 结果

| 方法 | 平均特征数 | Inner-CV DT | 平均相关性 | 新预算内轨迹 seed | DT Test Accuracy | 相对 MI-27 | 胜/平/负 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MI-27 | 27.0 | N/A | N/A | N/A | 0.9696 ± 0.0073 | 0 | N/A |
| Accuracy-only | 27.0 | 0.9418 ± 0.0438 | 0.2164 | 2/4 | 0.9420 ± 0.0510 | -0.0277 | 1/0/3 |
| beta=0.02, lambda=0.025 | 25.5 | 0.9607 ± 0.0092 | 0.2178 | 3/4 | 0.9680 ± 0.0045 | -0.0016 | 2/1/1 |

逐 seed：

| seed | 候选来源 | Inner-CV DT | DT Test | MI-27 Test | 差值 |
|---:|---|---:|---:|---:|---:|
| 42 | trajectory | 0.9550 | 0.9696 | 0.9718 | -0.0022 |
| 43 | trajectory | 0.9642 | 0.9675 | 0.9783 | -0.0108 |
| 44 | initial | 0.8768 | 0.8655 | 0.9675 | -0.1020 |
| 45 | initial | 0.9712 | 0.9653 | 0.9610 | +0.0043 |

### 结论

只看 DT Accuracy 明显失败，并且比最佳软惩罚 `lambda=0.025` 低 0.0260。4 个 seed 平均仅有
1.75 个不同合格候选（包含初始集合），只有 seed 42/43 产生新预算内轨迹；seed 44 全程停留在
超预算高分区，只能退回低质量初始集合，导致 test=0.8655。

因此问题不是相关性或 lambda 压低了 Accuracy，而是当前独立二元投票动作无法稳定生成固定预算候选。
后续应直接约束候选生成过程，例如每步投票后做 k=27 repair，或改成固定 27 个特征之间的一进一出交换；
继续删除奖励项或扫描软权重都不太可能稳定超过 MI-27。

## 2026-07-27 v15：主要实验和 baseline 复跑

### 这次做了什么

这次把数据切换到 v15，但没有另写一套数据处理流程，仍然直接走原来的雷达舰船数据加载和清洗代码。新数据包含两个 SVM-light 文件：

| 文件 | 样本数 | -1 类 | +1 类 | 原始特征数 |
|---|---:|---:|---:|---:|
| `sim_ship_cr_v15.train.svm` | 556 | 288 | 268 | 75 |
| `sim_ship_cr_v15.test.svm` | 140 | 72 | 68 | 75 |
| 合计 | 696 | 360 | 336 | 75 |

清洗规则仍然只在第一份源训练文件上拟合，然后把相同的列 mask 用到第二份文件。v15 中第 36、70 列是常量列，另外发现 7 组完全重复的特征：5→2、6→3、8→3、27→23、46→42、61→60、71→69。去掉这些列后剩下 66 个有效特征，也就是 66/75。

两份源文件随后合并成 696 行。每个 seed 都重新做一次分层 80%/20% 划分，得到 556 行 development 和 140 行 test。RL 只在 development 内做固定分层 5 折 DT 评价，完成特征选择后才用外层 test。因为有效特征从 v10 的 54 个变成了 66 个，半数特征的 MI baseline 也相应从 KBest-27 改为 KBest-33。

本次实际运行的实验如下：

| 实验组 | 跑的内容 | Seeds | 最终评价 |
|---|---|---|---|
| 基础 baseline | All Features、MI-KBest（k=33） | 42–46 | LR |
| 主 RL 实验 | MARLFS、Full-IRFS-fixed、Full-IRFS-trained-GCN | 42–46 | DT 和 LR |
| fixed 优选参数 | Full-IRFS-fixed，β=0.5 | 42–45 | DT |
| trained-GCN 优选参数 | Full-IRFS-trained-GCN，β=0.1 | 42–45 | DT |
| 预算惩罚优选参数 | Full-IRFS-fixed，β=0.02、λ=0.025、预算 33 | 42–45 | DT |

主 RL 和三组优选参数都沿用原来的 250 步搜索、Hybrid Teaching 调度和内部 5 折 DT 设置。这里没有重跑旧数据上表现较差的其余 β/λ 档位，也没有重跑 Accuracy-only 控制。

执行命令：

```bash
conda run --no-capture-output -n dl-lab python src/run_v15_key_experiments.py
```

### 主 RL 的内部 5 折选择结果

| 方法 | 平均选择特征数 | 最佳 inner-CV DT Accuracy | 平均搜索耗时/seed |
|---|---:|---:|---:|
| MARLFS | 32.6 | 0.9741 | 120.6 s |
| Full-IRFS-fixed | 41.4 | 0.9698 | 121.0 s |
| Full-IRFS-trained-GCN | 40.6 | 0.9701 | 1144.5 s |

MARLFS 平均选择的特征最少，内部 5 折分数也是三种方法中最高的。trained-GCN 的内部得分与 fixed 接近，但每个 seed 的运行时间大约是另外两种方法的 9 倍。

### 使用所选特征做 outer-test DT

| 方法 | 平均特征数 | DT Test Accuracy | 相对 MI-33 | 对 MI-33 胜/平/负 |
|---|---:|---:|---:|---:|
| All Features | 66.0 | 0.9486 ± 0.0137 | -0.0086 | 1/2/2 |
| MI-KBest，k=33 | 33.0 | 0.9571 ± 0.0152 | 0 | 0/5/0 |
| MARLFS | 32.6 | 0.9586 ± 0.0093 | +0.0014 | 3/0/2 |
| Full-IRFS-fixed | 41.4 | 0.9471 ± 0.0164 | -0.0100 | 1/1/3 |
| Full-IRFS-trained-GCN | 40.6 | 0.9529 ± 0.0120 | -0.0043 | 0/3/2 |
| MI-KBest，k 匹配 fixed | 41.4 | 0.9543 ± 0.0156 | -0.0029 | N/A |

这组 DT 结果里，MARLFS 的平均 Accuracy 为 0.9586，是表中最高值；它平均使用 32.6 个特征，与 MI-33 的规模基本相同。MI-33 的平均 Accuracy 为 0.9571，两者差距为 0.0014。

### 使用相同特征做独立 LR test

| 方法 | 平均特征数 | LR Test Accuracy |
|---|---:|---:|
| All Features | 66.0 | 0.9714 ± 0.0113 |
| MI-KBest，k=33 | 33.0 | 0.9657 ± 0.0137 |
| MARLFS 前面筛选出的特征 | 32.6 | 0.9614 ± 0.0229 |
| Full-IRFS-fixed 前面筛选出的特征 | 41.4 | 0.9571 ± 0.0168 |
| Full-IRFS-trained-GCN 前面筛选出的特征 | 40.6 | 0.9643 ± 0.0134 |

LR 评价中 All Features 的平均 Accuracy 最高，为 0.9714。三种 RL 特征里 trained-GCN 最高，为 0.9643，和 MI-33 的 0.9657 比较接近。

### 旧数据上较好参数的 v15 结果

下面三组配置是根据 v10 日志挑出来的较好参数，统一在 seeds 42–45 上复跑。同一批划分下，MI-33 的 DT Test Accuracy 是 `0.9518 ± 0.0107`。

| 配置 | 平均特征数 | Inner-CV DT | DT Test Accuracy | 相对 MI-33 | 胜/平/负 | 搜索总耗时 |
|---|---:|---:|---:|---:|---:|---:|
| Full-IRFS-fixed，β=0.5 | 47.5 | 0.9708 | 0.9446 ± 0.0122 | -0.0071 | 1/1/2 | 486.0 s |
| Full-IRFS-trained-GCN，β=0.1 | 42.8 | 0.9694 | 0.9393 ± 0.0170 | -0.0125 | 1/1/2 | 4751.6 s |
| Full-IRFS-fixed，β=0.02、λ=0.025、预算 33 | 32.2 | 0.9609 | 0.9446 ± 0.0179 | -0.0071 | 1/0/3 | 490.9 s |

这三组参数在 v15 上的平均 DT Accuracy 都低于同批 MI-33。β=0.5 和 λ=0.025 都得到 0.9446，trained-GCN β=0.1 得到 0.9393；其中预算惩罚方案把平均特征数控制到了 32.2。
