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

## 2026-07-29 v15.1：baseline 与主 RL 矩阵复跑

### 数据与协议

本轮将阶段 2 固定配置切换到 `DATA_VERSION="v15.1"`。v15.1 延续 v15 的 75 维
Z-score 特征空间，并在 train/test 各自分区内按标签独立增广；Swap 与 Mixup 的 donor 不跨标签、
不跨分区。原始与增广后的数据如下：

| 分区 | 原始行数 | 增广后行数 | -1 类 | +1 类 | 用途 |
|---|---:|---:|---:|---:|---|
| source train | 487 | 1461 | 756 | 705 | 清洗规则拟合、特征选择、最终模型拟合 |
| source test | 209 | 627 | 324 | 303 | 仅最终 DT/LR 评价 |
| 合计 | 696 | 2088 | 1080 | 1008 | N/A |

清洗规则只在 source train 上拟合，再原样应用到 source test。第 36、70 个原始特征为常量，
另有 7 组完全重复特征：5→2、6→3、8→3、27→23、46→42、61→60、71→69；最终保留
66/75 个特征，MI-KBest 固定为 `k=33`。加载时核验的输出文件 SHA-256 为：

- train：`c4767868802e4a46842fb54ddabf83df5ae3fee5a0cba3a98c26e43d1b69f695`；
- test：`5e649c1883167cfecd79f8098e18a4c987c8b01cbb06b821b43fa4f51d73d4da`。

本轮不再把两个源文件合并重切。5 个 seed 都固定使用 1461 行 source train 作为 development、
627 行 source test 作为最终评价集；seed 只改变 inner-CV、RL 和最终 DT 的随机过程。RL 在
source train 内使用固定分层 5 折 DT Accuracy 反馈，完成全部选择并保存特征编号后，独立入口才读取
source test。每种 RL 每个 seed 运行 250 步，Hybrid Teaching 仍为 83/83/84。

### 实验矩阵与命令

| 类别 | 方法 | Seeds | 最终评价 |
|---|---|---|---|
| baseline | All Features-66、MI-KBest-33 | 42–46 | LR；并在 DT 入口复评 |
| 同规模 baseline | MI-KBest，k 匹配各 seed 的 Full-IRFS-fixed 特征数 | 42–46 | DT |
| RL | MARLFS、Full-IRFS-fixed、Full-IRFS-trained-GCN | 42–46 | inner-CV DT、source-test DT/LR |

```bash
conda run -n dl-lab python src/run_basic_baselines.py
conda run --no-capture-output -n dl-lab python src/run_stage2_rl_selection.py
conda run --no-capture-output -n dl-lab python src/run_stage2_dt_test.py
conda run --no-capture-output -n dl-lab python src/run_stage2_rl_final_lr.py
```

### 基础 LR baseline

source train/test 固定，且 5 个 seed 的 MI 子集完全一致，因此这两项的跨 seed 标准差为 0。

| 方法 | 特征数 | Test Accuracy | Balanced Accuracy | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| All Features | 66 | 0.9601 ± 0.0000 | 0.9600 | 0.9587 | 0.9925 |
| MI-KBest | 33 | 0.9633 ± 0.0000 | 0.9630 | 0.9617 | 0.9961 |

### RL 内部 5 折选择结果

共完成 3 方法 × 5 seeds = 15 次选择、3750 个 RL step；选择阶段所有产物均满足
`test_used_during_selection=false` 和 `lr_final_called=false`。

| 方法 | 平均特征数 | 压缩率 | 最佳 inner-CV DT Accuracy | 最佳步数 | 选择 Jaccard | 耗时/seed |
|---|---:|---:|---:|---:|---:|---:|
| MARLFS | 30.0 ± 5.1 | 54.55% | 0.9951 ± 0.0018 | 147.0 | 0.3004 | 137.1 s |
| Full-IRFS-fixed | 44.6 ± 6.3 | 32.42% | 0.9940 ± 0.0012 | 152.0 | 0.5237 | 148.2 s |
| Full-IRFS-trained-GCN | 46.0 ± 3.5 | 30.30% | 0.9940 ± 0.0011 | 102.6 | 0.5620 | 1310.4 s |

15 次搜索记录的累计耗时约 7978.3 秒（2.22 小时）。trained-GCN 每个 seed 约为 fixed 的
8.8 倍，但内部 5 折均值没有提高。

### 使用已冻结特征的 source-test DT

每组最终 DT 都用全部 1461 行 source train 拟合，在固定的 627 行 source test 上评价；RL 不重跑。

| 方法 | 平均特征数 | DT Test Accuracy | 相对 MI-33 | 对 MI-33 胜/平/负 |
|---|---:|---:|---:|---:|
| All Features | 66.0 | 0.9493 ± 0.0052 | -0.0054 | 1/0/4 |
| MI-KBest，k=33 | 33.0 | 0.9547 ± 0.0056 | 0 | 0/5/0 |
| MARLFS | 30.0 ± 5.1 | 0.9455 ± 0.0102 | -0.0093 | 2/0/3 |
| Full-IRFS-fixed | 44.6 ± 6.3 | 0.9496 ± 0.0064 | -0.0051 | 2/0/3 |
| Full-IRFS-trained-GCN | 46.0 ± 3.5 | 0.9445 ± 0.0132 | -0.0102 | 1/0/4 |
| MI-KBest，k 匹配 fixed | 44.6 ± 6.3 | 0.9534 ± 0.0060 | -0.0013 | 2/1/2 |

### 使用相同 RL 特征的独立 source-test LR

| 方法 | 平均特征数 | LR Test Accuracy | Balanced Accuracy | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| MARLFS | 30.0 ± 5.1 | 0.9464 ± 0.0079 | 0.9459 | 0.9437 | 0.9852 |
| Full-IRFS-fixed | 44.6 ± 6.3 | 0.9563 ± 0.0083 | 0.9560 | 0.9544 | 0.9869 |
| Full-IRFS-trained-GCN | 46.0 ± 3.5 | 0.9547 ± 0.0110 | 0.9543 | 0.9525 | 0.9827 |

同一评价器下，基础 LR 的 MI-33 为 0.9633、All Features 为 0.9601，均高于三种 RL 子集。

### 结论与产物

1. v15.1 的三种 RL 在 inner-CV 上都达到约 0.994–0.995，但没有把该优势带到增广后的固定
   source test；DT 上 MI-33 以 0.9547 居首，LR 上同样是 MI-33 的 0.9633 居首。
2. RL 中 Full-IRFS-fixed 最值得保留观察：它在 RL 方法里取得最高 DT（0.9496）和 LR（0.9563），
   但平均使用 44.6 个特征，仍未超过更紧凑的 MI-33。
3. MARLFS 最紧凑（平均 30 个特征），但 DT/LR 分别低于 MI-33 约 0.0093/0.0169。
4. trained-GCN 没有把约 8.8 倍于 fixed 的计算时间转化为更高的 outer-test 指标，本数据版本下不建议
   继续扩大其网格。5 个 seed 也不足以据此宣称统计显著性。
5. inner-CV 与固定增广 test 的差距提示搜索选择偏差或分布差异；下一步若继续优化，应优先检查增广
   train/test 的分布一致性、用嵌套选择估计搜索偏差，或约束为固定 33 特征的交换动作，而不是继续堆
   trained-GCN 计算量。

主要产物：

- `experiments/radar_ship_v15.1_basic_lr/`；
- `experiments/radar_ship_v15.1_stage2_rl_selection/`；
- `experiments/radar_ship_v15.1_stage2_dt_test/`；
- `experiments/radar_ship_v15.1_stage2_rl_final_lr/`；
- `results/tables/radar_ship_v15.1_stage2_*.csv`；
- `logs/v15.1_baselines_2026-07-29.log`；
- `logs/v15.1_rl_selection_2026-07-29.log`；
- `logs/v15.1_dt_test_2026-07-29.log`；
- `logs/v15.1_rl_final_lr_2026-07-29.log`。

运行前定向测试：`18 passed in 1.70s`。产物核验：15 个选择 JSON、每个 250 步、共 3750 条轨迹，
数据版本均为 v15.1，选择期 test/LR 密封标记全部正确。

## 2026-07-29 v15.1 → v15.2：具体加噪处理与难度校准

### 前提与结构约束

本次增强直接作用于已经做过 Z-score 标准化的 v15 SVM 数据。因此没有在标准化数值上继续套用
RCS 非负、角度周期或 `[0,1]` 区间等原始物理域约束，而是在标准化特征空间内进行扰动。

原始 75 维中，第 36、70 列为常量，另有 7 组完全重复列：
5→2、6→3、8→3、27→23、46→42、61→60、71→69。所有版本都只对 66 个有效独立特征加噪，
然后映射回 75 维：常量列保持原值，重复列重新复制其代表列。这样可以避免把原本无信息的常量列
变成只在某个类别中波动的伪特征，也不会用独立噪声人为拆开完全重复的特征。

训练集和测试集始终分别增强。Swap、Mixup 和异类 donor 都只能从同一分区取样，不允许
train/test 交叉取样。随机种子固定为 train=`42`、test=`43`。

### 第一次尝试：v15.1 同类增强

v15.1 按参考增强脚本给不同标签配置不同策略。每项策略都从该标签的原始样本独立生成一份副本，
不是把多个策略串行叠加到同一副本上：

| 标签 | 策略 | 参数 | 具体处理 |
|---:|---|---|---|
| +1 | Feature Dropout | `drop_rate=0.05` | 随机将 5% 特征置 0；标准化空间中的 0 表示训练均值 |
| +1 | Gaussian Jitter | `std=0.05` | 各维独立加入 `N(0, 0.05²)` |
| -1 | Same-class SwapNoise | `prob=0.10` | 10% 单元替换为同分区、同标签样本的同维特征 |
| -1 | Same-class Mixup | `alpha=0.20` | 同分区、同标签样本间插值，权重来自 `Beta(0.2, 0.2)` |

原样本与两份增强副本合并后，每个分区扩大到 3 倍：

| 分区 | 原始 | 增强 | 最终 | 标签分布 |
|---|---:|---:|---:|---|
| train | 487 | 974 | 1461 | -1=756，+1=705 |
| test | 209 | 418 | 627 | -1=324，+1=303 |

该方案没有明显增加难度。固定 source train/test 边界后，各模型结果如下：

| 数据 | LR | DT | Random Forest | Extra Trees | RBF-SVM | 最强单特征 AUC |
|---|---:|---:|---:|---:|---:|---:|
| 原始 v15 | 0.9665 | 0.9522 | 0.9809 | 0.9761 | 0.9569 | 0.9105 |
| v15.1 | 0.9665 | 0.9537 | 0.9793 | 0.9729 | 0.9490 | 0.9082 |

失败原因是同类 Swap/Mixup 主要填充原有类内流形，不会把两类边界拉近；`std=0.05` 和
5% Dropout 相对标准差约为 1 的特征也过弱。另外，两个标签使用不同增强算子，存在让模型学习
“增强方式”而不是目标语义的风险。因此 v15.2 改为两类完全对称的异类近邻混合。

### 四档强度尝试

异类混合的基础公式为：

```text
x_mix = λ * x_source + (1 - λ) * x_opposite
```

`x_opposite` 从同一分区内、与源样本标签相反的 5 个最近邻中随机选择。插值后，再按概率把部分维度
直接替换成 donor 的同维值，随后叠加 Gaussian Jitter 和 Dropout。标签始终保留源样本标签，
没有随机翻转标签。

每档都同时增强 train/test，但 test 使用更强的扰动。下表中参数依次为
`副本数 / λ范围 / 异类特征交换率 / Gaussian std / Dropout`：

| 档位 | train 参数 | test 参数 | LR | DT | RF | RBF-SVM |
|---|---|---|---:|---:|---:|---:|
| M，中等 | `3 / .65–.85 / .10 / .10 / .05` | `3 / .55–.80 / .15 / .15 / .08` | 0.8804 | 0.8660 | 0.9019 | 0.9043 |
| S，强 | `3 / .58–.80 / .15 / .15 / .08` | `3 / .48–.72 / .22 / .20 / .12` | 0.8313 | 0.8086 | 0.8409 | 0.8648 |
| H，更强 | `3 / .52–.75 / .22 / .20 / .12` | `3 / .42–.68 / .30 / .25 / .16` | 0.7787 | 0.7572 | 0.8014 | 0.8050 |
| X，极强 | `4 / .48–.72 / .28 / .25 / .15` | `4 / .38–.62 / .38 / .30 / .20` | 0.7177 | 0.6794 | 0.7464 | 0.7273 |

校准表中的 DT 使用 `min_samples_leaf=2`，RF 使用 200 棵树。M 档仍接近 90%，H/X 档开始接近
过度破坏标签语义，因此最终选择 S 档作为 v15.2：目标是把常见模型压到约 80%–86%，而不是制造
接近随机标签的任务。

### 最终方案：v15.2 对称异类近邻强混合

v15.2 从原始 v15 重新生成，没有在 v15.1 上重复叠加噪声。每个原样本生成 3 个增强副本并保留原样本，
所以两个分区均扩大到 4 倍：

| 分区 | λ 范围 | 异类特征交换率 | Gaussian std | Dropout | 原始/增强/最终 |
|---|---:|---:|---:|---:|---:|
| train | 0.58–0.80 | 0.15 | 0.15 | 0.08 | 487 / 1461 / 1948 |
| test | 0.48–0.72 | 0.22 | 0.20 | 0.12 | 209 / 627 / 836 |

最终标签分布为 train：`-1=1008，+1=940`；test：`-1=432，+1=404`。在正式写出的 SVM 文件上，
使用 66 个有效特征重新评价：

| 模型 | Train Accuracy | Test Accuracy |
|---|---:|---:|
| Logistic Regression | 0.9420 | 0.8313 |
| Decision Tree | 1.0000 | 0.7955 |
| Random Forest，300 trees | 1.0000 | 0.8433 |
| Extra Trees，300 trees | 1.0000 | 0.8636 |
| RBF-SVM | 0.9579 | 0.8648 |

最强单特征 test AUC 为 0.8022（0-based feature 22），相比原始 v15 的 0.9105 明显下降。

文件与 SHA-256：

- `dataset/sim_ship_cr_v15.2.train.svm`：
  `e130d9c7ca0c491cfeea353ed4872e628e8482c72de3d40f3f96c43839fde3a2`；
- `dataset/sim_ship_cr_v15.2.test.svm`：
  `e5ee17adad0e690d532771a96b26e830fcda617bbf23df5ebb02a61fbc009f95`；
- 参数记录：`dataset/sim_ship_cr_v15.2.augmentation.json`；
- 生成器：`dataset/buildv15sim/generate_v15_2.py`。

生成命令：

```bash
conda run -n dl-lab python dataset/buildv15sim/generate_v15_2.py
```

### 校验、限制与使用注意事项

已完成以下校验：

1. train/test 分别为 1948/836 行，每行均为标签加完整的 0-based `0..74` 共 75 个特征；
2. 标签仅包含 `-1/+1`，全部数值有限，无 NaN/Inf；
3. 两个常量列和 7 组重复列在增强后保持原结构；
4. 原始 487/209 行全部保留；
5. sklearn SVM-light 和项目数据加载器都能读取，加载器清洗后为 `2784 × 66`；
6. 相同种子重生成的 train/test 与正式文件逐字节一致。

必须保持 v15.2 的 source train/test 文件边界。若先合并再随机划分，原样本及其增强副本可能被分到
两侧，造成近重复样本泄漏，准确率会重新虚高。当前 SVM 格式也没有记录“增强副本属于哪个原样本”的
group id，因此 train 内部普通分层交叉验证仍可能把相关副本分到不同 fold；如果要把 inner-CV 分数
当作严格泛化估计，应额外保存 provenance/group 信息并改用 GroupKFold。

最后，v15.2 是判别特征空间中的 robustness/stress-test 数据，而不是从原始 IQ 信号重新仿真的
物理分布。异类特征交换和靠近边界的硬混合有意制造语义模糊样本，适合增加分类与特征选择难度，
不应被解释为真实雷达测量噪声模型。

## 2026-07-29 v15.3：关闭 Feature Dropout 的对照版本

根据后续要求，另外生成 v15.3。该版本仍从原始 v15 生成，不在 v15.2 上重复叠加噪声；与 v15.2
相比只关闭 train/test 的 Feature Dropout，其余异类近邻混合、异类特征交换、高斯噪声、分区隔离、
副本数量和随机种子均保持不变。

| 分区 | λ 范围 | 异类特征交换率 | Gaussian std | Dropout | 原始/增强/最终 |
|---|---:|---:|---:|---:|---:|
| train | 0.58–0.80 | 0.15 | 0.15 | **0.00** | 487 / 1461 / 1948 |
| test | 0.48–0.72 | 0.22 | 0.20 | **0.00** | 209 / 627 / 836 |

每个 donor 仍从同一分区、相反标签的 5 个最近邻中随机选择；标签保留源样本标签，不随机翻转，
train/test 之间不交叉取样。75 维结构恢复规则与 v15.2 相同：只增强 66 个有效独立特征，常量列和
重复列保持原结构。最终标签分布仍为 train：`-1=1008，+1=940`；test：`-1=432，+1=404`。

文件与 SHA-256：

- `dataset/sim_ship_cr_v15.3.train.svm`：
  `dd814b46c076c34dbc96994cd4ec48607c480d62759f6a5d179e6fb11befb9d6`；
- `dataset/sim_ship_cr_v15.3.test.svm`：
  `032c60abf54ccd4757787998f996f042fec8f5ef7213fd0cd3e6bed445f0c94b`；
- 参数记录：`dataset/sim_ship_cr_v15.3.augmentation.json`；
- 生成器：`dataset/buildv15sim/generate_v15_3.py`。

生成命令：

```bash
conda run -n dl-lab python dataset/buildv15sim/generate_v15_3.py
```

本次按要求只生成无 Dropout 数据版本，**没有重新运行分类器准确率或难度评估**。使用时仍须保持
source train/test 边界，避免原样本与增强副本在重新随机划分后产生近重复泄漏。

## 2026-07-29 v15.2 覆盖更新：Z 空间物理先验针对性加噪

根据更新后的 `feature_noise_augmentation_report.md`，v15.2 已重新生成并覆盖此前的
nearest-opposite hard-mixing 版本。旧版本生成器保留为
`dataset/buildv15sim/generate_v15_2_hardmix_legacy.py`，不再对应当前 v15.2 文件。

### 方法变更

当前 v15.2 不使用 Feature Dropout、异类 Mixup、特征交换或标签翻转，而是对报告定义的十类特征
分别实施 Z 空间同构噪声：

| 特征类型 | 0-based 索引 | Z 空间处理 |
|---|---|---|
| Type 1A 非负连续量 | 0、2、3、5–7、9–17、23–25、27–29、31–32、36–37、56–57、65、67–68、70、73–74 | 以 source-train 最小值为锚点的中心化 Log-Normal 乘性噪声 |
| Type 1B 下界为 1 | 20 | 同样采用下界锚定的 Log-Normal 放缩 |
| Type 2 离散计数 | 1、4、69 | 从 source-train 唯一值谱推断最小 Z 步长，执行 `±1` 步并在最小值截断 |
| Type 3A `[0,1]` | 8、21、30、33、35、39–50（不含 38）、52、54–55、58、61–62、66、71 | 用 train min/max 映射到虚拟 `[0,1]`，在 Logit 空间加噪后逆映射 |
| Type 3B `[-1,1]` | 51、63–64、72 | train min/max 锚点映射后执行有界 Logit 扰动 |
| Type 4A 反射角 | 53 | 在 train Z 跨度内进行带反射的角度高斯扰动 |
| Type 4B/4C 周期角 | 59、60 | 在 train Z 跨度内执行周期包裹 |
| Type 5A 盒维数 | 38 | 加性高斯后截断到 train min/max |
| Type 5B 无界量/高阶矩 | 18–19、22、26、34 | Z 空间加性高斯；峰度保留下界，多普勒量截断到经验范围 |

所有端点、跨度和离散步长都只在原始 source train 上拟合，再原样用于 test；test 不参与锚点估计。
报告所称的物理边界恢复在实现中按“经验近似”处理，因为有限样本的 min/max 不一定等于真实物理端点。
为避免 Log-Normal 重尾产生数百至数千个标准差的异常值，非负量的增强结果额外截断在
`train_min + 5 × train_span`。第 36、70 个常量特征保持不变，7 组完全重复特征仍复制代表列，
因此加载器有效特征数保持 66。

### 最终强度与数据规模

每个原始样本保留，并生成 3 个针对性噪声副本。train/test 分别独立加噪，test 使用更强的压力测试参数：

| 参数 | Train | Test |
|---|---:|---:|
| Log-Normal `eta` | 0.80 | 1.50 |
| 计数步长变化概率 | 1.00 | 1.00 |
| 有界特征 Logit `sigma` | 5.00 | 12.00 |
| 角度噪声/经验跨度 | 0.50 | 1.20 |
| 盒维数噪声/经验跨度 | 0.50 | 1.20 |
| 无界特征加性 `sigma` | 2.50 | 6.00 |
| Log-Normal 最大经验跨度 | 5.00 | 5.00 |

| 分区 | 原始 | 增强 | 最终 | 标签分布 |
|---|---:|---:|---:|---|
| train | 487 | 1461 | 1948 | -1=1008，+1=940 |
| test | 209 | 627 | 836 | -1=432，+1=404 |

### 全特征测试结果

测试使用正式写出的六位小数 SVM 文件。LR 为
`StandardScaler + LogisticRegression(max_iter=5000)`，DT 为默认参数且 `random_state=42`：

| 特征口径 | LR Train | LR Test | DT Train | DT Test |
|---|---:|---:|---:|---:|
| 导出的全部 75 列 | 0.9677 | 0.8349 | 1.0000 | 0.8864 |
| 加载器清洗后的全部 66 个有效特征 | 0.9677 | 0.8337 | 1.0000 | 0.8696 |

因此当前 v15.2 在全有效特征下不再是 95% 以上的简单任务，同时没有依赖异类混合或 Dropout。

### 文件、复现与校验

- train：`dataset/sim_ship_cr_v15.2.train.svm`，SHA-256
  `8d4bc82df95373ad164502a60e1b53cfb1fbea36d2a9dbad15441236807799d5`；
- test：`dataset/sim_ship_cr_v15.2.test.svm`，SHA-256
  `9c6f76b35272e1578151e6befd6ee709e7ff07e848398e33829c748a2b5118ca`；
- 参数、锚点和 benchmark：`dataset/sim_ship_cr_v15.2.augmentation.json`；
- 当前生成器：`dataset/buildv15sim/generate_v15_2.py`；
- 十类噪声实现：`dataset/buildv15sim/physical_zspace_augmenter.py`。

```bash
conda run -n dl-lab python dataset/buildv15sim/generate_v15_2.py
```

已验证全部值有限，train/test 形状为 `1948×75`/`836×75`，每行包含完整 0-based `0..74`，
原始样本全部保留，十类约束、离散步长、常量/重复结构、SHA-256、sklearn 读取和项目加载器读取均通过；
相同随机种子重生成的两个 SVM 文件与正式文件逐字节一致。


## 2026-07-29 v15.3 覆盖更新：标准化前原始特征的物理感知加噪【已废止：train/test 非同步】

本次用 `dataset/buildv15sim/v15_raw.npz` 覆盖生成 v15.3，取代此前的
“nearest-opposite hard mixing、无 Dropout”版本；旧生成器已保留为
`dataset/buildv15sim/generate_v15_3_no_dropout_hardmix_legacy.py`。首先验证了 raw NPZ 的
`X_train/y_train/X_test/y_test` 形状分别为 `487×75/487/209×75/209`，使用 source train 的
均值和标准差可在六位小数精度下逐项复现原 v15 train/test SVM，因此确认该 NPZ 是对应 v15
划分和顺序的标准化前源数据。

### 管线与数据隔离

train/test 都保留原样本，并各自独立生成 3 个物理噪声副本；最终为 train `1948×75`
（`-1=1008，+1=940`）、test `836×75`（`-1=432，+1=404`）。没有使用 Dropout、异类
Mixup、特征交换、标签翻转，也没有跨 train/test 取样。参考报告最后建议 test 不加噪，但本任务
此前明确要求增强 test 以提高任务难度，因此 v15.3 test 定义为“加噪鲁棒性压力测试”，不是干净
分布上的泛化测试。

所有噪声先施加在原始物理/数学空间。随后只在合并后的 raw train（原始 + 3 份增强）上计算一组
总体均值和总体标准差（`ddof=0`），同一组参数用于转换 train/test；test 从不单独拟合均值、
标准差或噪声尺度。常量列 35、69（0-based）保持不变；公式重复列保持
`4→1、5→2、7→2、26→22、45→41、60→59、70→68`，加载器有效特征数仍为 66。
其中 41/45 是排序前后数学等价的熵公式，raw 浮点值仅有约 `1e-8` 的运算顺序差异，故显式
恢复为重复列。

### 实际加噪处理

原始空间增强器为 `dataset/buildv15sim/physical_raw_augmenter.py`。各组处理如下（索引均为
0-based）：

- 非负连续量
  `0、2–3、5–7、9–17、22–32、34、36–37、39、41–45、48–51、56–57、59–61、65、67–68、70、73`
  使用 `X' = X·exp(N(0,σ²))`；为防止 Log-Normal 重尾数值异常，将抽样的 log 偏差截断到
  `[-3,3]`。
- 下界为 1 的索引 20 使用
  `X' = 1 + (X-1)·exp(N(0,σ²))`。
- 离散索引 1、4、69、74 使用 `{-1,0,+1}` 步进抖动并在 0 截断；从 source train 推断的
  索引 1 步长为 1，索引 74 步长为 `2.875 m`。69 是全零常量，故保持不变。
- `[0,1]` 特征
  `8、21、33、35、46–47、52、55、62、71` 在 Logit 空间加高斯噪声并逆映射。
- `[-1,1]` 特征 `54、58、63–64、66、72` 先映射到 `[0,1]`，在 Logit 空间扰动后还原。
- 平均散射角 53 添加角度高斯噪声后，以余弦/反余弦无限反射到 `[0°,90°]`。
- 盒维数 38 添加高斯噪声后截断到 `[1,2]`。
- 无界/高阶矩 `18、19、40` 使用基于 source train
  `IQR/1.3489795`（退化时回退标准差）的自适应加性高斯；19 额外截断在峰度下界 `-2`。

最终参数：

| 参数 | Train | Test（选定） |
|---|---:|---:|
| Log-Normal `σ` | 0.55 | 1.375 |
| 离散步长改变概率 | 0.75 | 1.00 |
| 有界特征 Logit `σ` | 1.50 | 5.00 |
| 散射角高斯 `σ`（度） | 12.0 | 37.5 |
| 盒维数高斯 `σ` | 0.10 | 0.3125 |
| 自适应加性噪声/robust scale | 0.75 | 2.50 |
| 最大绝对 log 偏差 | 3.0 | 3.0 |

### 对参考报告定义域的修正

实现以 `build_sim_v15.py` 和相关特征提取公式为最终依据，没有把真实数据强制裁剪到报告中写错的
区间：30 的实际公式会超过 1，改用非负乘性噪声；39、41–45 是未归一化熵，改用非负乘性噪声；
40 的 density-histogram 表达式在 v15 中为负且无界，改用自适应加性噪声；48–51 实际是幅度/
幅度积，不是报告所称比例或相关系数；54、58 是有符号各向异性/对比度，按 `[-1,1]` 处理；
59/60 实际是完全重复的非负 `Shv/Shh` 幅度比而不是周期角；66 的当前公式可为负，按
`[-1,1]` 处理；74 是 `2.875 m` 距离单元网格而不是单位整数。上述修正均已写入
`sim_ship_cr_v15.3.augmentation.json`。

### 难度校准尝试与最终测试

保持 train 配方不变，对初始 test 配方测试 `1.0×、1.25×、1.5×、2.0×`。模型均读取正式
六位小数 SVM 值；LR 为
`StandardScaler + LogisticRegression(max_iter=5000, random_state=42)`，DT 为默认
`DecisionTreeClassifier(random_state=42)`：

| Test 相对初始强度 | 66列 LR Test | 66列 DT Test | 75列 LR Test | 75列 DT Test |
|---:|---:|---:|---:|---:|
| 1.00× | 0.9151 | 0.8744 | 0.9163 | 0.8696 |
| **1.25×（选定）** | **0.8983** | **0.8373** | **0.8947** | **0.8254** |
| 1.50× | 0.8792 | 0.8050 | 0.8780 | 0.8038 |
| 2.00× | 0.8469 | 0.7775 | 0.8505 | 0.7691 |

选择 `1.25×` 是因为它是让全部有效特征下 LR 和 DT 都低于 90% 的最弱候选，避免采用更强但
不必要的 test 分布偏移。最终 75 列的 LR/DT train accuracy 分别为 `0.9861/1.0000`；
66 个加载器有效特征分别为 `0.9867/1.0000`。

### 文件、哈希与校验

- train：`dataset/sim_ship_cr_v15.3.train.svm`，SHA-256
  `ca43f838d36d872da24bb0bd161a8ef8fe45cd52c1563beed17dc16852c59414`；
- test：`dataset/sim_ship_cr_v15.3.test.svm`，SHA-256
  `5c1b57f8a54fc240fd8618fb33fa19551f336135a665c63e988b97d7da329741`；
- raw source：`dataset/buildv15sim/v15_raw.npz`，SHA-256
  `94fd97537ade5f33d4d0c4e712b8ce879605a51f1ef84a3915dee386cb44544c`；
- 参数、标准化统计、报告修正和 benchmark：
  `dataset/sim_ship_cr_v15.3.augmentation.json`；
- 当前生成器：`dataset/buildv15sim/generate_v15_3.py`；
- 物理噪声实现：`dataset/buildv15sim/physical_raw_augmenter.py`。

```bash
conda run -n dl-lab python dataset/buildv15sim/generate_v15_3.py
```

已验证 raw train/test 物理约束均零违规、全部值有限、每行完整包含 0-based `0..74`、原始
train/test 样本分别保留 `487/487` 与 `209/209`、常量/重复结构正确、Z-Score 后合并 train
的非恒定列均值/标准差误差小于 `2e-14`、项目正式加载器得到 `2784×66` 且 source train/test
边界为 `1948/836`。相同种子重生成的两个 SVM 与正式文件逐字节一致。
## 2026-07-29 v15.3 再次覆盖：train/test 同步物理噪声的特征选择基准【已覆盖：噪声强度不足】

上一版 raw-space v15.3 将 train 设为常规噪声、test 设为更强压力噪声；该设计适合鲁棒性压力
测试，但 train/test 存在刻意的噪声分布偏移，不适合作为特征选择方法的常规同分布比较。因此该版
已废止，当前 v15.3 改为：

- train/test 使用**完全相同**的逐特征噪声模型、参数、物理边界、log-tail 截断和每原样本副本数；
- 两侧都保留原样本并各生成 3 个增强副本，仅随机种子不同（train=42，test=43）；
- 噪声尺度、离散步长仍只从 source train 拟合；
- Z-Score 仍只在合并后的 raw train（原始 + 增强）上拟合，test 复用 train 的同一组均值和
  标准差，绝不对 test 单独标准化；
- 生成入口会拒绝不相等的 train/test profile，防止以后误生成非同步 v15.3。

同步后的共同参数采用上一版已经确定的 test 物理标准，避免同时改变两侧的目标噪声定义：

| 参数 | Train | Test |
|---|---:|---:|
| Log-Normal `σ` | 1.375 | 1.375 |
| 离散步长改变概率 | 1.00 | 1.00 |
| 有界特征 Logit `σ` | 5.00 | 5.00 |
| 散射角高斯 `σ`（度） | 37.5 | 37.5 |
| 盒维数高斯 `σ` | 0.3125 | 0.3125 |
| 自适应加性噪声/robust scale | 2.50 | 2.50 |
| 最大绝对 log 偏差 | 3.0 | 3.0 |
| 每个原样本的增强副本 | 3 | 3 |

逐特征物理模型、报告定义域修正、常量列和公式重复列规则与上一节相同；本次只纠正
train/test 的噪声标准不一致问题。

### 同步噪声尝试

所有候选均让 train/test 使用同一 profile、独立随机种子，并从正式六位小数 SVM 重新加载后测试。
前五组均为每原样本 3 份增强副本：

| 候选 | `σlog` | 离散概率 | Logit `σ` | 角度 `σ` | 盒维数 `σ` | Additive/scale | 66列 LR Test | 66列 DT Test |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A low | 0.55 | 0.75 | 1.5 | 12° | 0.10 | 0.75 | 0.9533 | 0.9569 |
| B medium | 0.90 | 0.90 | 3.0 | 22° | 0.18 | 1.50 | 0.9569 | 0.9426 |
| C medium-high | 1.20 | 1.00 | 4.5 | 32° | 0.27 | 2.20 | 0.9510 | 0.9354 |
| **最终同步标准** | **1.375** | **1.00** | **5.0** | **37.5°** | **0.3125** | **2.50** | **0.9474** | **0.9402** |
| D high | 1.50 | 1.00 | 6.0 | 45° | 0.36 | 3.00 | 0.9378 | 0.9342 |
| E very-high | 2.00 | 1.00 | 8.0 | 60° | 0.45 | 4.00 | 0.9378 | 0.9103 |

为确认“增加副本或极端噪声”是否能在同分布条件下合理降难度，还尝试了：

| 候选 | 副本数 | 66列 LR Test | 66列 DT Test |
|---|---:|---:|---:|
| Extreme（`σlog=3, Logit=12, angle=90°, box=0.70, additive=8`） | 3 | 0.9306 | 0.9031 |
| D high | 7 | 0.9390 | 0.9151 |
| E very-high | 7 | 0.9300 | 0.9175 |
| Extreme | 7 | 0.9199 | 0.8989 |
| Extreme | 11 | 0.9215 | 0.9111 |

结果表明：当 train/test 真正同分布时，模型会从增强 train 中学习相应噪声不变性，不能像上一版
那样依靠单独加重 test 将准确率压低。Extreme 和 7/11 副本虽然能进一步降分，但噪声已接近
物理边界饱和且数据量显著扩大，容易让特征选择方法主要适应人为噪声，故没有采用。最终标准优先
保持物理合理性、与上一版 test 标准的连续性，以及不同特征选择算法之间的公平同分布比较。

### 最终规模、测试和文件

最终规模不变：train `1948×75`（`-1=1008，+1=940`），test `836×75`
（`-1=432，+1=404`）。基准模型与此前相同：

| 特征口径 | LR Train | LR Test | DT Train | DT Test |
|---|---:|---:|---:|---:|
| 导出的全部 75 列 | 0.9564 | 0.9462 | 1.0000 | 0.9342 |
| 加载器清洗后的 66 个有效特征 | 0.9548 | 0.9474 | 1.0000 | 0.9402 |

- train：`dataset/sim_ship_cr_v15.3.train.svm`，SHA-256
  `4a3ac224d147c1a48b177d078cae9d4fc6c5e201c85a63e17fb04a251378c288`；
- test：`dataset/sim_ship_cr_v15.3.test.svm`，SHA-256
  `8a3001baef945e9f3661a84b0ae98a7605ada2c2f827d6048373105793a4974a`；
- 同步 profile、标准化统计、物理约束和 benchmark：
  `dataset/sim_ship_cr_v15.3.augmentation.json`；
- 生成器：`dataset/buildv15sim/generate_v15_3.py`。

已重新验证两侧 profile 和副本数完全相等、不同 profile 会被生成器拒绝、raw 物理约束零违规、
原始 train/test 样本全部保留、常量/重复结构正确、项目加载器输出 `2784×66` 且 source
train/test 边界为 `1948/836`。相同随机种子重生成的两个 SVM 与正式文件逐字节一致。

## 2026-07-29 v15.3 再次覆盖：同步增强强度提高至全特征准确率低于 90%

上一同步版已经保证 train/test 同分布，但 66 个有效特征的 LR/DT test accuracy 仍为
`0.9474/0.9402`，难度不足。当前版本继续严格保持 train/test 使用相同 profile、相同 3 个
增强副本和独立随机种子，只同步提高原始物理空间的噪声强度；没有恢复 test 专属噪声、Dropout、
异类混合、特征交换或标签翻转。

### 校准过程

首先提高噪声尾部，并比较增强副本数。表中顺序为 66 列 LR/DT 和 75 列 LR/DT：

| 候选 | `σlog` | Logit `σ` | 角度 `σ` | 盒维数 `σ` | Additive/scale | log 截断 | 副本 | 66 LR/DT | 75 LR/DT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| G | 3.0 | 14 | 90° | 0.70 | 8 | ±4 | 3 | 0.9175 / 0.9139 | 0.9246 / 0.9079 |
| H | 4.0 | 18 | 135° | 1.00 | 12 | ±5 | 3 | 0.8959 / 0.8911 | 0.8971 / 0.8816 |
| I | 5.0 | 24 | 180° | 1.50 | 16 | ±6 | 3 | 0.8768 / 0.8888 | 0.8792 / 0.8888 |
| G | 3.0 | 14 | 90° | 0.70 | 8 | ±4 | 5 | 0.9035 / 0.8931 | 0.9051 / 0.8931 |
| H | 4.0 | 18 | 135° | 1.00 | 12 | ±5 | 5 | 0.8876 / 0.9067 | 0.8907 / 0.8971 |
| H | 4.0 | 18 | 135° | 1.00 | 12 | ±5 | 7 | 0.8888 / 0.8911 | 0.8894 / 0.8906 |

H（三副本）首次让全部四项低于 90%，I 明显更强，因此继续在 G/H 之间插值。三副本的细化结果：

| 相对 G→H 强度 | `σlog` | Logit `σ` | 角度 `σ` | 盒维数 `σ` | Additive/scale | log 截断 | 66 LR/DT | 75 LR/DT |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 3.25 | 15.0 | 101.25° | 0.775 | 9.0 | ±4.25 | 0.9091 / 0.9031 | 0.9115 / 0.8983 |
| 0.50 | 3.50 | 16.0 | 112.5° | 0.850 | 10.0 | ±4.50 | 0.9031 / 0.8959 | 0.9043 / 0.8911 |
| 0.75 | 3.75 | 17.0 | 123.75° | 0.925 | 11.0 | ±4.75 | 0.9007 / 0.8900 | 0.9007 / 0.8852 |
| **0.80（最终）** | **3.80** | **17.2** | **126°** | **0.940** | **11.2** | **±4.80** | **0.8983 / 0.8864** | **0.8995 / 0.8888** |
| 0.85 | 3.85 | 17.4 | 128.25° | 0.955 | 11.4 | ±4.85 | 0.8995 / 0.8923 | 0.8959 / 0.8840 |
| 0.90 | 3.90 | 17.6 | 130.5° | 0.970 | 11.6 | ±4.90 | 0.8983 / 0.8780 | 0.8983 / 0.8816 |
| 0.95 | 3.95 | 17.8 | 132.75° | 0.985 | 11.8 | ±4.95 | 0.8971 / 0.8744 | 0.8971 / 0.8876 |

选择 0.80 档，因为 0.75 档的 LR 仍为 `0.9007`，而 0.80 是沿统一强度路径第一个让 66/75
特征下 LR 和 DT 全部低于 90% 的候选。0.85–0.95 与 I 虽也达标，但噪声更强，没有采用。
离散步长改变概率在所有本轮候选中均为 1.0。

### 当前正式 profile 与结果

train/test 当前共同参数：

| 参数 | Train | Test |
|---|---:|---:|
| Log-Normal `σ` | 3.80 | 3.80 |
| 离散步长改变概率 | 1.00 | 1.00 |
| 有界特征 Logit `σ` | 17.20 | 17.20 |
| 散射角高斯 `σ`（度） | 126.0 | 126.0 |
| 盒维数高斯 `σ` | 0.940 | 0.940 |
| 自适应加性噪声/robust scale | 11.20 | 11.20 |
| 最大绝对 log 偏差 | 4.80 | 4.80 |
| 每原样本增强副本 | 3 | 3 |

| 特征口径 | LR Train | LR Test | DT Train | DT Test |
|---|---:|---:|---:|---:|
| 导出的全部 75 列 | 0.9117 | **0.8995** | 1.0000 | **0.8888** |
| 加载器清洗后的 66 个有效特征 | 0.9112 | **0.8983** | 1.0000 | **0.8864** |

最终规模仍为 train `1948×75`、test `836×75`。正式文件：

- train SHA-256：`e0c325ebe4eae599d23974fa76ea905078f59e0f23d668306716de33036b5fc3`；
- test SHA-256：`5e8d5bcf906915805ed3a53a79e256cabf6cf9569538785030a90c6cd10daebf`；
- 参数与 benchmark：`dataset/sim_ship_cr_v15.3.augmentation.json`；
- 生成器：`dataset/buildv15sim/generate_v15_3.py`。

已验证 train/test profile 与副本数完全相同、raw 物理约束零违规、全部值有限、常量/重复结构
正确、项目加载器输出 `2784×66` 且 source train/test 边界为 `1948/836`。相同种子重生成的
两个 SVM 与正式文件逐字节一致。

## 2026-07-29 v15.3：baseline 与快速 RL 特征选择复跑（trained-GCN 暂缓）

本轮使用当前正式的同步物理噪声增强数据复跑 baseline 和计算较快的 RL 方法。按本轮要求，
Full-IRFS-trained-GCN 不运行，但在下面各结果表中保留空行，便于后续补跑后直接填写。

### 数据与协议

数据加载器只在 source train 上识别常量列和完全重复列，再把同一列掩码应用到 source test；
最终候选池为 66/75 个特征：

- train：`dataset/sim_ship_cr_v15.3.train.svm`，`1948×75`，SHA-256
  `e0c325ebe4eae599d23974fa76ea905078f59e0f23d668306716de33036b5fc3`；
- test：`dataset/sim_ship_cr_v15.3.test.svm`，`836×75`，SHA-256
  `5e8d5bcf906915805ed3a53a79e256cabf6cf9569538785030a90c6cd10daebf`；
- seeds：42–46；MI-KBest 固定 `k=33`；
- 每种 RL 每个 seed 运行 250 步，Hybrid Teaching 为 83/83/84；
- RL 只使用全部 1948 行 source train 内的固定分层 5 折 DT Accuracy 反馈；836 行 source test
  只在特征冻结后用于最终 DT/LR 评价。

本轮实际 RL 矩阵为 MARLFS 与 Full-IRFS-fixed；Full-IRFS-trained-GCN 仍保留在完整方法注册表，
但不进入当前 active run matrix。

```bash
conda run --no-capture-output -n dl-lab python src/run_basic_baselines.py
conda run --no-capture-output -n dl-lab python src/run_stage2_rl_selection.py
conda run --no-capture-output -n dl-lab python src/run_stage2_dt_test.py
conda run --no-capture-output -n dl-lab python src/run_stage2_rl_final_lr.py
```

### 基础 LR baseline

| 方法 | 平均特征数 | Test Accuracy | Balanced Accuracy | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| All Features | 66.0 | 0.8959 ± 0.0000 | 0.8943 | 0.8872 | 0.9476 |
| MI-KBest，k=33 | 33.0 | 0.8775 ± 0.0073 | 0.8752 | 0.8643 | 0.9315 |

### RL 内部 5 折选择结果

共完成 2 方法 × 5 seeds = 10 次选择、2500 个 RL step。全部选择产物均满足
`test_used_during_selection=false` 和 `lr_final_called=false`。

| 方法 | 平均特征数 | 压缩率 | 最佳 inner-CV DT Accuracy | 最佳步数 | 选择 Jaccard | 耗时/seed |
|---|---:|---:|---:|---:|---:|---:|
| MARLFS | 33.8 ± 3.4 | 48.79% | 0.9161 ± 0.0023 | 126.6 | 0.3725 | 189.6 s |
| Full-IRFS-fixed | 47.0 ± 3.4 | 28.79% | 0.9172 ± 0.0026 | 154.4 | 0.5745 | 221.1 s |
| Full-IRFS-trained-GCN |  |  |  |  |  |  |

10 次搜索记录的累计耗时约 2053.4 秒（34.2 分钟）。

### 使用已冻结特征的 source-test DT

每组最终 DT 都用全部 1948 行 source train 拟合，在固定的 836 行 source test 上评价；RL 不重跑。

| 方法 | 平均特征数 | DT Test Accuracy | 相对 MI-33 | 对 MI-33 胜/平/负 |
|---|---:|---:|---:|---:|
| All Features | 66.0 | 0.8852 ± 0.0063 | -0.0160 | 1/0/4 |
| MI-KBest，k=33 | 33.0 | 0.9012 ± 0.0042 | 0 | 0/5/0 |
| MARLFS | 33.8 ± 3.4 | 0.8900 ± 0.0059 | -0.0112 | 0/0/5 |
| Full-IRFS-fixed | 47.0 ± 3.4 | 0.8873 ± 0.0105 | -0.0139 | 0/0/5 |
| Full-IRFS-trained-GCN |  |  |  |  |
| MI-KBest，k 匹配 fixed | 47.0 ± 3.4 | 0.8923 ± 0.0052 | -0.0089 | 1/0/4 |

### 使用相同 RL 特征的独立 source-test LR

| 方法 | 平均特征数 | LR Test Accuracy | Balanced Accuracy | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| All Features | 66.0 | 0.8959 ± 0.0000 | 0.8943 | 0.8872 | 0.9476 |
| MI-KBest，k=33 | 33.0 | 0.8775 ± 0.0073 | 0.8752 | 0.8643 | 0.9315 |
| MARLFS | 33.8 ± 3.4 | 0.8464 ± 0.0139 | 0.8431 | 0.8234 | 0.9180 |
| Full-IRFS-fixed | 47.0 ± 3.4 | 0.8840 ± 0.0069 | 0.8819 | 0.8724 | 0.9412 |
| Full-IRFS-trained-GCN |  |  |  |  |  |

### 结论

1. v15.3 明显比 v15.1 更难：本轮 All-66 的 LR/DT 分别为 0.8959/0.8852。
2. DT 评价中 MI-33 以 0.9012 居首，同时只保留一半特征；两个已跑 RL 方法均未超过 MI-33。
3. 两个 RL 中，MARLFS 更紧凑（33.8 个特征），DT 略高于 fixed；Full-IRFS-fixed 的 LR 明显更高，
   且跨 seed 选择更稳定，但平均保留 47.0 个特征。
4. trained-GCN 本轮没有运行，不能与上述方法作数值比较；对应表格行已按要求留空。

### 产物与核验

- `experiments/radar_ship_v15.3_basic_lr/`；
- `experiments/radar_ship_v15.3_stage2_rl_selection/`；
- `experiments/radar_ship_v15.3_stage2_dt_test/`；
- `experiments/radar_ship_v15.3_stage2_rl_final_lr/`；
- `results/tables/radar_ship_v15.3_stage2_*.csv`；
- `logs/v15.3_baselines_2026-07-29.log`；
- `logs/v15.3_fast_rl_selection_2026-07-29.log`；
- `logs/v15.3_fast_rl_dt_test_2026-07-29.log`；
- `logs/v15.3_fast_rl_final_lr_2026-07-29.log`。

已核验 10 个 `selection.json`、每个 250 步、合计 2500 条轨迹；数据版本均为 v15.3，
source train/test 行数均为 1948/836，选择期 test/LR 隔离标记正确；v15.3 产物中没有任何
trained-GCN 文件或目录。运行前定向测试为 `16 passed`；运行后全量 pytest 为 `106 passed`，本轮
改动文件的 `ruff check` 全部通过。


## 2026-08-02 stable_v1：强化学习训练稳定性内核重构

本次只处理工程化、训练设施与数值稳定性，不调整奖励公式、advisor 数学规则、同步
SELECT/DESELECT 动作空间和最终候选选择规则。正式实现提交为
`ba1c79d refactor: add stable RL training core`。

### 训练语义

stable 训练路径统一为以下配置：

| 项目 | stable_v1 设置 |
|---|---:|
| 算法 | Double DQN |
| discount | 0.9 |
| optimizer | Adam |
| learning rate | 3e-4 |
| replay capacity | 2048 个 joint 环境步 |
| batch / warm-up | 32 / 32 个有效环境步 |
| target hard sync | 每 25 次 learner update |
| loss | SmoothL1 / Huber，全部有效 `(batch, agent)` 项取 mean |
| gradient clipping | global norm 10 |
| epsilon | 1.0 线性降至 0.05，前 70% 总步数完成衰减 |
| tensor dtype / device | float32 / CPU reference path |

一个环境步现在只生成一条 `JointTransition`，其中同时保存全部 feature-agent 的 action 和 reward
vector，不再把同一步复制成 N 份 replay transition。`IndependentQSystem` 仍保留每个特征一个独立
Q head，但每个 head 一次处理完整 batch，统一输出 `[B,N,2]`。

online QSystem 包含全部 online heads 和可选 trainable GCN；target QSystem 是无 optimizer 的显式
副本。minimal/fixed encoder 在 online/target 间共享只读实例，trained-GCN 的 target encoder 独立
复制。trainer 只创建一个 optimizer，覆盖全部 online heads 和可选 online encoder，避免同一参数被
多个 optimizer 重复管理。

### 稳定性保护

- 使用 online QSystem 选择 next action、target QSystem 评价该 action；terminal transition 不做
  bootstrap。
- 全选或全不选继续沿用原 no-op guard：正常记录 accuracy、reward 和动作统计，但
  `applied=false`，不进入 replay，也不触发 optimizer update。
- replay 超容量后淘汰最旧 joint transition；有效环境步达到 warm-up 后每步最多更新一次。
- Huber loss 前检查有限性；反向后执行 global gradient norm clip，并检查 loss、梯度和状态/Q tensor
  中的 NaN/Inf。
- minimal/fixed/trained-GCN 全部提供 `[B,N,D]` 的批量接口。minimal 直接批量生成相关性状态，不再
  调用旧逐 agent encoder；trained-GCN 的参数相关输出不跨 optimizer step 缓存。
- trained-GCN online state 会随 optimizer 更新变化；target GCN 在同步点前保持冻结。

### Checkpoint 与恢复

每 25 步及非整周期结束时原子写入 `checkpoint.pt`，内容包括：

- 当前 step、committed subset、候选 archive 和完整 trajectory；
- online/target QSystem、optimizer、joint replay 和 learner update 计数；
- epsilon scheduler、reward/advisor 内部状态；
- NumPy、Python 和 Torch RNG 状态；
- config SHA-256、development fingerprint、method SHA-256 和 seed。

恢复时会拒绝 config/data/method/seed 或 replay/scheduler/advisor 配置不匹配的 checkpoint。自动测试
确认中断恢复后的 online/target 参数、replay 抽样序列、learner update、trajectory 和 selection 与
连续运行 bit-identical；`elapsed_seconds` 是真实墙钟观测，不纳入 bit-identical 比较。加载路径同时
兼容 Torch 1.x 和 Torch 2.6+ 的 `weights_only` 默认行为。

### 诊断与产物

每步统一记录：subset size、accuracy、epsilon、proposed select count、transition applied、reward
min/mean/max、replay size、是否更新、loss、TD error、Q/target Q 统计、gradient norm、target sync、
advisor override count 和 elapsed seconds。

`StepCompleted`、`UpdateCompleted`、`CheckpointSaved` 是不可变 observer 事件；控制台、
`training.csv` 和 `training.jsonl` 复用同一事件源。恢复时 JSONL 会按 step 去重，避免 checkpoint
之前日志已写出但训练状态尚未提交所造成的重复记录。

每个 stable run 统一生成 `manifest.json`、`training.csv`、`training.jsonl`、`selection.json` 和
`checkpoint.pt`。manifest 记录规范化 TOML、config/method/data hash、Git 状态、数据摘要以及
Python/NumPy/pandas/sklearn/Torch 版本。结果根目录由 `algorithm_version + config_hash` 独占，
禁止 legacy/stable 或不同配置混写。

### 工程隔离与兼容

- 新实现位于 `src/radar_ship_fs/`；stable 模块由架构测试禁止 import legacy 和顶层 stage2 脚本。
- 旧的 3048 行 stage2 orchestration 已整体迁入 `radar_ship_fs.legacy.stage2`，原
  `src/run_stage2_*.py` 均为 12 行兼容包装；旧命令、旧 import、测试 monkeypatch 和历史产物格式
  继续有效。
- trained-GCN stable 路径与测试同步完成，但 `configs/v16n/stable.toml` 默认不启用。
- 本轮没有改变 reward、advisor、动作空间或 final candidate 规则，也没有启动正式的 v16n
  5-seed × 250-step stable 实验。

### 验证记录

```bash
conda run -n dl-lab python -m ruff format .
conda run -n dl-lab python -m ruff check .
conda run -n dl-lab python -m pytest -q
```

- Ruff：全部通过；
- pytest：`129 passed in 47.68s`，包含原有 106 项回归；
- 三种 encoder 均完成 stable 短运行；
- synthetic terminal MDP 上 TD loss 下降并学习到已知较优动作；
- WDBC 两步 CLI smoke 通过，生成完整 manifest、训练日志、selection 和 checkpoint；manifest 中
  method hash 为 `66d931ff7ba5...`，数据摘要为 455 行 search、114 行 held-out、30 个特征。

结论：本轮建立了可诊断、可恢复、目标网络隔离且 optimizer 所有权唯一的 stable 训练基线。后续若
要比较 reward、counterfactual credit、动作空间或 advisor 规则，应在该内核上作为独立实验改动，
避免再与训练设施问题混杂。
