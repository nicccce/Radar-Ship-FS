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

## 阶段 2 雷达 RL 正式实验

当前 `stage2_rl_config.py` 选择 v15：先合并两个 SVM-light 文件的 696 行，再按每个 seed 分层随机划分为 556 行
development 和 140 行 test。RL 只在 development 内做固定分层 5 折交叉验证：每个候选特征集
训练 5 棵 Decision Tree，并用 5 个留出折准确率均值作为选择分数。test 不参与特征选择。

RL 搜索运行 MARLFS、Full-IRFS-fixed 和 Full-IRFS-trained-GCN。每种方法运行 250 步，
Hybrid Teaching 的零基边界为 83/166：`[0,83)` 使用 relevance trainer，`[83,166)` 使用
DT-importance trainer，`[166,250)` 不再使用 trainer；候选分数相同时选择特征更少的一组：

```bash
conda run --no-capture-output -n dl-lab python src/run_stage2_rl_selection.py
```

RL 完成并保存特征编号后，独立入口用全部 development 训练最终 Decision Tree，并在 test 上评价
All Features、KBest-33、三种 RL 前面筛选出的特征，以及与 Full-IRFS-fixed 同规模的 KBest：

```bash
conda run --no-capture-output -n dl-lab python src/run_stage2_dt_test.py
```

如需补充 Logistic Regression 评价，可在选择完成后单独运行；它只读取已保存的特征，不会重新运行 RL：

```bash
conda run --no-capture-output -n dl-lab python src/run_stage2_rl_final_lr.py
```

针对 RL 子集经常超过 MI-33 预算的问题，另有独立的超预算惩罚扫描。它固定
`beta=0.02`、`k=33`，扫描 `lambda={0.01, 0.025, 0.05, 0.1}`：

```text
J(S) = Accuracy_CV(S) - beta * Corr(S) - lambda * max(0, (|S| - 33) / 33)
```

RL 学习期间仍可访问超过 33 个特征的子集，但最终只在初始 33 特征与 250 步轨迹中
`|S| <= 33` 的候选里按 inner-CV DT Accuracy 选最优（同分取更少特征）。一键入口会先完成
全部 4×4 个密封搜索并预检产物，随后才解封 outer test：

```bash
conda run --no-capture-output -n dl-lab python src/run_stage2_budget_sweep.py
```

纯 Accuracy 控制（`beta=0、lambda=0`，但保留最终 `|S|<=33` 的公平筛选）使用独立入口：

```bash
conda run --no-capture-output -n dl-lab python src/run_stage2_accuracy_only.py
```

当前 v15 的 baseline、主三方法和旧数据优选点可用统一入口断点续跑：

```bash
conda run --no-capture-output -n dl-lab python src/run_v15_key_experiments.py
```


这些正式入口都不接收命令行实验参数：

- 配置统一写在 `src/stage2_rl_config.py`；
- 每个 seed 的 140 行随机 test 在搜索期保持密封；
- 搜索入口不导入 `lr_final`，只用 development 内部 5 折 DT 分数；
- DT 评价入口不会重新运行 RL，只读取 `selection.json` 中前面筛选出的特征编号；
- KBest 的 Mutual Information 只在该 seed 的 development 上拟合；
- 每个方法保存 250 步原始轨迹、逐折准确率、均值/标准差和子集变化，并生成跨 seed 聚合 CSV。

重复执行搜索入口时，与当前代码内配置完全匹配的已完成方法会被跳过；配置签名不一致时不会误用旧产物。

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
