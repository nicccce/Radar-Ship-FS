# Radar-Ship-FS

Radar-Ship-FS 是面向雷达舰船数据的特征选择实验工程。当前代码基底来自
[Yigal-Meshulam/interactive-rl-feature-selection](https://github.com/Yigal-Meshulam/interactive-rl-feature-selection)，
后续项目代码将在该 MIT 开源复现的基础上继续修改、接入自有数据，并按路线一实验协议补齐统一评价流程。

上游基底主要覆盖：

- 传统特征选择 baseline；
- `marlfs` 多智能体强化学习 baseline；
- `full_irfs` 及 Trainer、个性化奖励、fixed / trained-GCN 状态编码；
- 多随机种子实验产物和聚合统计。

本项目第一阶段采用的实验叙事是：

```text
All Features / KBest / 连续可微子集
        -> MARLFS
        -> Full-IRFS-fixed
        -> Full-IRFS-trained-GCN
```

阶段 2 的 RL 搜索在 80% development 内使用固定分层 5 折 Decision Tree 平均准确率作为反馈，
外层 20% test 在搜索期保持密封。RL 筛选完成后，独立评价入口用全部 development 训练新的
Decision Tree，再在 test 上统一比较 All Features、KBest 和三种 RL 前面筛选出的特征。
Logistic Regression scorer 仍保留为可选的独立评价模块，但不进入 RL engine 或 reward。

## 目录结构

```text
.
|-- src/                    # 上游 IRFS/MARLFS 代码主体
|-- tests/                  # 上游测试
|-- documents/              # 论文、架构说明、技术记录
|   `-- experiment-log.md   # 项目实验日志
|-- configs/                # 项目级配置草稿和说明
|-- data/                   # 本地数据目录，内容默认不入库
|   |-- raw/                # 原始数据
|   |-- processed/          # 清洗或切分后的中间数据
|   `-- external/           # 第三方公开数据或示例数据
|-- experiments/            # 程序自动生成的逐次运行产物
|-- results/                # 汇总后的表格、图和模型文件
|-- logs/                   # 本地运行日志
|-- pyproject.toml
|-- requirements.lock
`-- README.md
```

## 环境安装

建议使用 Python 3.10 或更新版本。当前已在 `dl-lab` 环境的 Python 3.10.20、torch 1.11.0+cu113 上验证通过。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
```

若直接复用已有的 `dl-lab` conda 环境，并希望保留其中已有的 CUDA torch，可只补齐缺失依赖：

```bash
conda run -n dl-lab python -m pip install pandas scikit-learn pytest ruff mrmr-selection
```

依赖声明见 [pyproject.toml](pyproject.toml)，pip 锁定依赖见 [requirements.lock](requirements.lock)；
已通过全量测试的实际 `dl-lab` 快照见 [environment.dl-lab.yml](environment.dl-lab.yml)。

## 快速验证

先用上游自带的 WDBC 数据跑通最小流程：

```bash
python src/run_irfs.py --dataset wdbc --seeds 42 --state-encoder fixed
```

再运行 trained-GCN 状态编码：

```bash
python src/run_irfs.py --dataset wdbc --seeds 42 --state-encoder trained_gcn
```

如需打开额外 Trainer 消融：

```bash
python src/run_irfs.py --dataset wdbc --seeds 42 --diagnostic-ablations
```

默认实验产物会写入 `experiments/<dataset>/seed-<seed>/`，跨随机种子的聚合结果写入
`experiments/<dataset>/aggregate.json`。

## stable_v1 训练内核

新的工程化入口只读取版本化 TOML，默认配置是
[`configs/v16n/stable.toml`](configs/v16n/stable.toml)。先检查展开后的 seed/method 矩阵：

```bash
conda run -n dl-lab env PYTHONPATH=src \
  python -m radar_ship_fs.experiment dry-run --config configs/v16n/stable.toml
```

确认后运行；`--seed`、`--method` 只能过滤 TOML 已声明的矩阵，不能覆盖超参数：

```bash
conda run -n dl-lab env PYTHONPATH=src \
  python -m radar_ship_fs.experiment run --config configs/v16n/stable.toml --resume
```

stable 内核使用 joint replay、批量状态编码、Double DQN、独立 target QSystem、Huber mean loss、
全局梯度裁剪和线性 epsilon 调度。每个 `seed/method` 目录统一生成：

- `manifest.json`：规范化配置、SHA-256、数据身份、Git 与运行时版本；
- `training.csv` / `training.jsonl`：同一不可变 observer 事件产生的逐步诊断；
- `selection.json`：保留原选择核心字段并追加 stable 诊断；
- `checkpoint.pt`：online/target、optimizer、joint replay、scheduler、advisor 与 RNG 状态。

结果根目录由 `algorithm_version + config_hash` 独占，不允许 legacy/stable 或不同配置混写。
`full_irfs_trained_gcn` 已实现并有集成测试，但在默认 TOML 中保持关闭。

下面的 `src/run_stage2_*.py` 是冻结的 `legacy_v1` 历史复现入口。它们继续使用原目录、签名和
完成产物，不再承接 stable 训练功能；新代码也通过架构测试禁止导入这些脚本。

## v16n_2x_noise 核心实验与最新结论

当前项目已升级到 `v16n_2x_noise` 数据集，采用完全统一的 TOML 配置和 `ExperimentRunner` 进行管理，彻底废弃了硬编码参数的老旧执行脚本。

### 1. 核心运行协议
- **数据切分**：合并读取数据后，每个随机种子 (seed=42-46) 划分 1843 行做 development (训练/验证)，461 行做独立 test。
- **特征选择 (Stage 1)**：所有强化学习方法 (PPO, DQN) 和部分需要内部反馈的传统方法 (mi_greedy, dt_rfe) 仅在 development 的 1843 行内部进行 5 折交叉验证（内部折叠计算决策树准确率作为 Reward 或反馈）。绝对禁止偷看 461 行的独立 test 集。
- **独立验证 (Stage 2)**：特征集筛选并冻结后，由统一评价脚本 `src/run_domain_gcn_screen_eval.py` 统筹读取。在 development 集合上重新拟合 Decision Tree 和 Logistic Regression 模型，最后在独立的 461 行 test 集上计算泛化指标。

### 2. 模型矩阵与配置
主配置文件位于 `configs/v16n/run_experiments.toml`，默认包含了完整的模型对比矩阵：
- **强化学习方法**：
  - `gnn_ppo` (单智能体 PPO，极速热启动收敛，泛化效果最佳的 RL)
  - `full_irfs_fixed` (单智能体 DQN)
  - `marlfs` (经典多智能体)
- **传统基线 (Baselines)**：
  - `mi_greedy` (互信息排序后前向贪心，表现极强的过滤式基线)
  - `dt_rfe` (基于决策树的递归特征消除)
  - `mrmr` (最大相关最小冗余)
  - `l1` (L1正则化嵌入)
  - `relevance_topk` (单纯互信息排序 Top-K)

### 3. 一键执行命令
运行所有在 TOML 配置文件里激活的方法：
```bash
conda run --no-capture-output -n dl-lab env PYTHONPATH=src python -u -m radar_ship_fs.experiment run --config configs/v16n/run_experiments.toml
```
运行完成后，执行统一特征评估，计算最终的 DT 和 LR 测试集精度：
```bash
conda run -n dl-lab env PYTHONPATH=src python src/run_domain_gcn_screen_eval.py --config configs/v16n/run_experiments.toml
```


## 数据约定

本地数据内容不提交到 Git。建议按下面约定放置：

- `data/raw/`：原始数据，只做备份和读取；
- `data/processed/`：清洗、对齐、切分后的中间数据；
- `data/external/`：公开数据、上游示例数据或临时对照数据。

接入自有标注数据时，优先只修改数据适配层，使加载函数输出：

```python
X  # shape: [num_samples, num_features]
y  # shape: [num_samples]
```

当前阶段 2 使用外层 `80% development / 20% test`，并在 development 内做 5 折选择；标准化、
填补、筛选器拟合、相关矩阵和搜索期模型训练都不得使用最终 test。

## 结果约定

- `experiments/` 保存程序级原始产物，便于复现；
- `results/tables/` 保存整理后的主表和补充表；
- `results/figures/` 保存论文或汇报用图；
- `results/models/` 保存必要的模型、特征掩码或缓存快照；
- [documents/experiment-log.md](documents/experiment-log.md) 记录每次实验目标、命令、数据版本、随机种子和结论。

## 测试与检查

提交改动前建议运行：

```bash
ruff format --check .
ruff check .
pytest
```

如果只想确认主流程是否能跑通，可先运行：

```bash
python src/run_irfs.py --dataset wdbc --seeds 42 --state-encoder fixed
```

## 上游来源

- Upstream: [Yigal-Meshulam/interactive-rl-feature-selection](https://github.com/Yigal-Meshulam/interactive-rl-feature-selection)
- Imported commit: `f777b4d3e8dd4b89869efd94f28afb7128fa7617`
- License: MIT, see [LICENSE](LICENSE)
