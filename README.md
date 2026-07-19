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

其中 RL 搜索阶段保留上游代码的 Decision Tree 验证反馈；冻结特征子集后的最终主表计划统一使用
`StandardScaler + LogisticRegression` 在独立测试集上评价。这个最终 LR scorer 是项目后续改造项，
不要直接替换训练期的 `DecisionTreeProbe`。

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

依赖声明见 [pyproject.toml](pyproject.toml)，锁定依赖见 [requirements.lock](requirements.lock)。

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

训练、奖励验证和最终测试建议采用 `60% / 20% / 20%` 固定划分，并确保标准化、填补、筛选器拟合、
相关矩阵和模型训练都不使用最终测试集。

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
