# Experiment Log

本日志记录 Radar-Ship-FS 的代码来源、实验命令、数据版本、随机种子、输出位置和阶段性结论。每次实验完成后请追加一条记录，保证结果可以复查。

## 固定实验约定

- 代码基底：`Yigal-Meshulam/interactive-rl-feature-selection`
- 导入 commit：`f777b4d3e8dd4b89869efd94f28afb7128fa7617`
- 训练期 RL feedback：保留上游 `DecisionTreeProbe`
- 最终主评价：计划新增独立 `StandardScaler + LogisticRegression` scorer
- 首轮方法矩阵：All Features、KBest、连续可微子集、MARLFS、Full-IRFS-fixed、Full-IRFS-trained-GCN
- 推荐切分：训练集 60%，奖励验证集 20%，最终测试集 20%

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
- 接入雷达舰船标注数据的数据适配层。
- 新增与训练期 Decision Tree probe 分离的最终 Logistic Regression scorer。
- 将首轮主实验矩阵固定为 All Features、KBest、连续可微子集、MARLFS、Full-IRFS-fixed、Full-IRFS-trained-GCN。

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
