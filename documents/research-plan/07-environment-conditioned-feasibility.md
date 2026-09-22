# 任务 7：环境条件化特征选择前提验证

日期：2026-09-21

最终状态：**BLOCKED / NO-GO（数据前提失败）**

## 1. 结论

任务 7 未进入性能评价。当前数据无法证明真实 base scene、增强父子关系、环境条件或选择时环境可用性，
也无法构造合法的 scene-group-disjoint 五折。因此不能回答“不同环境是否确实需要不同子集”。

这是数据准入结论，不是任何条件化方法或 RL 算法的性能 NO-GO。未训练 PPO，未运行全局/分环境静态、
监督条件化或 contextual bandit 基线，未打开任务 7 或历史基线的结果文件，也未打开历史 source-test
特征或标签。按任务要求阅读的路线和 3A/4B 协议文档中已有摘要仅用于准入判定。

## 2. 数据与谱系审计

| 检查 | 观察 | 判定 |
|:---|:---|:---|
| source-train 身份 | 3897 行；`-1: 2016, +1: 1881`；SHA-256 `2191fdc297aa49ce6a700df956ec44f68dc13e4c737937b31167543a203d22bb` | 文件级身份可复核 |
| 行级标识 | SVM 中 `qid:` 行数 0、注释行数 0；manifest 无 row ledger | `base_scene_id`/parent 不可恢复 |
| 当前 loader | radar 分支显式设置 `groups=None`；metadata 只含文件、标签和列清洗摘要 | 无合法 group/env 输入 |
| split 能力 | splitter 只有在上游提供真实 `groups` 时才保持整组；radar 当前走行级分层路径 | 不能保证同源不跨 fold |
| 目标版本生成链 | 未发现 `v16n_2x_noise` generator、生成配置、shuffle permutation、split indices 或 lineage manifest | 增强/split 顺序及父子关系未知 |
| 可见增强代码 | `dataset/augment_features.py` 只是未被目标版本调用证明的 helper；`buildv15sim` 与 `v15.3` 是旧版本链 | 不得归因到目标版本 |
| 环境元数据 | 旧 v15 代码曾从文件名解析 `ang/wave/scr/scene_sid`，但未写入下游 SVM；目标 manifest 无可连接字段 | 海况、擦地角、SCR 不可用 |
| 选择时可用性 | selector 输入无环境字段；无传感器接口、采集时点或部署合同 | 未证明，准入失败 |
| 任务 6 final-data protocol | 仓库中不存在 | 无新增 final 数据可继承；历史 test 不得替代 |

本轮使用的当前证据 hash：

- `documents/research-plan/03a-data-lineage-audit.md`：
  `82a2cb969af87be26c31a7734f7d149de66b893a974b0265eb3b5b99cf7577fb`
- `src/data/loader.py`：
  `9a17c0e3fdc91f6d7a2a219d85a383fb86f447d4b2fb77951104666ce67e9d36`
- `src/data/splitter.py`：
  `9148453537dc4ecf90af24cbb8ddac06ff866cc90cd2177e94804765617d9729`
- `src/run_reward_alignment.py`：
  `779f6b9fbec982d1117ea56b73a843711af454d5dc04208efef0a1c7fa20ed92`
- `experiments/v16n_2x_noise/seed-42/marlfs/manifest.json`：
  `49de7bbf895f796a7df3e27a6a237fe498b21bfa956cd0d507fb82c6dbe07813`

历史 source-test 特征/标签读取次数为 **0**。其既有摘要没有用于本次计数、split 或性能评价。

## 3. 准入与 split 判定

| 必须条件 | 结果 |
|:---|:---|
| 可靠 base scene / parent scene | 失败 |
| 可靠且选择前可获得的环境变量 | 失败 |
| 同源场景及增强副本保持同一 outer fold | 不可验证，按失败处理 |
| 每个 `environment × class` 至少 25 个独立 base scenes | 不可计算，按失败处理 |
| 五折中每个 validation fold 每格至少 5 个 scenes | 不可构造，按失败处理 |
| 每个 group fold 同时覆盖全部类别和环境 | 不可构造，直接 NO-GO |

没有创建 row→group/env 映射，也没有创建 folds。现有 3897 个行位置不能升级为独立 scene 计数；增强行
也不能代替独立 base scenes。不得通过改变环境分箱、删除困难环境、重抽 folds 或结果反推补救。

## 4. 基线、指标与成本

| 项目 | 状态/计数 |
|:---|:---|
| 全局静态 K=32 | 未运行 |
| 分环境静态 K=32 | 未运行 |
| 简单监督条件化选择器 | 未运行 |
| contextual bandit | 未运行 |
| PPO / 多步 RL | 未提出、未运行 |
| J evaluator 总请求 | 0 |
| 唯一评分子集 | 0 |
| cache hit | 0 |
| 分类器拟合 | 0 |
| outer folds / validation 评价 | 0 / 0 |
| 任务 7 / 基线结果文件读取 | 0 |

因此平均 LR BAcc、最差环境 BAcc、环境间方差、未见 scene/env 表现、选择稳定性和 GO 门槛均为
`N/A — blocked before evaluation`，不能补零或引用历史结果。

## 5. 最小数据补充需求

解除阻断只需、也至少需要以下可验证包：

1. 目标版本 generator manifest，冻结代码/hash、输入/hash、物理参数与 scene seeds、增强参数、标准化、
   shuffle 和 split 规则；
2. 覆盖最终 train/test 每一行且可直接对齐文件的 row-lineage ledger，含 original row、base scene、
   augmentation parent/copy/type；
3. 以 `base_scene_id` 为唯一键的 environment catalog，含 class、海况、擦地角、SCR、单位和来源；
4. 环境变量在特征选择前可获得的部署/采集合同，证明不依赖待选择特征或标签；
5. 带 hash 的验证脚本和 fold manifest，证明 parent/base scene 不跨 partition/fold，并在冻结环境表示后满足
   每个 `environment × class ≥ 25` 独立 scenes、每个五折 validation 单元 `≥ 5` scenes 及全类/全环境覆盖。

若历史数据无法提供这些材料，应从保留 scene/env 键的新 generator 重新发布一次预冻结、版本化的数据集，
而不是从现有特征矩阵推造 group/env。

## 6. 最终决定

**NO-GO：停止环境条件化路线，直到第 5 节的最小数据包完整通过审计。**

在此之前，不执行条件化基线、不读取性能结果、不设置环境分箱、不训练 PPO，也不把本报告解释为方法性能
失败。解除阻断后仍必须先按
`documents/research-plan/07-environment-conditioned-feasibility-protocol.md` 冻结环境表示、group folds、
K=32、4B scorer、总 J 请求预算、指标、回退和门槛，才能开始评价。
