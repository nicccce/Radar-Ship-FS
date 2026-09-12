# 实验配置

新科研比较入口为 `configs/v16n/research_baseline.toml`，冻结细节见 `documents/research-plan/protocol.md`。`configs/v16n/research_smoke.toml` 只用于小规模链路验证。旧 `stable.toml`、`run_experiments.toml` 及其输出保留为历史复现协议，不能与新结果混写。

新主配置将 `feature_id_node_feature=false` 且 `feature_id_reward_weight=0.0`；实现还保证随机 feature ID 不进入候选池、shaping 或 terminal objective。仅在旧配置中把 reward weight 改成 0 并不会自动改变旧产物的协议身份。

`configs/v16n/stable.toml` 是新的 `stable_v1` 正式入口配置。配置由严格 schema 一次性校验：
未知字段、非法枚举、无效 DQN 参数、空运行矩阵和 Hybrid 边界错误都会在数据加载前失败。

```bash
python -m radar_ship_fs.experiment dry-run --config configs/v16n/stable.toml
python -m radar_ship_fs.experiment run --config configs/v16n/stable.toml --resume
```

CLI 只允许选择配置、恢复策略，以及过滤 TOML 中已经声明的 seed/method。学习率、batch、
replay、target 同步等参数必须修改 TOML，使 manifest 中的规范化配置与实际运行始终一致。

历史 PPO 配置默认用独立于训练 seed 的 `feature_id_seed = 0` 给特征生成固定随机编号，并将归一化编号
作为节点静态特征。`feature_id_reward_weight = 0.1` 控制编号在 shaping/terminal reward 中的
弱偏好；`archive_accuracy_tolerance = 0.001` 允许最终归档在准确率近似时用含编号项的 objective
打破平局。跨 seed 对比时必须保持 `feature_id_seed` 一致；将 reward weight 设为 `0.0` 可关闭偏好。
swap 模式下，编号偏好同时参与候选排序；`swap_candidate_pool` 控制每个半步可见的候选数，
`max_swaps` 控制每个 episode 的最大替换次数。扩大二者会增加探索覆盖率和运行成本。


默认矩阵启用 `marlfs/minimal` 和 `full_irfs_fixed/fixed`；`trained_gcn` 与 stable trainer 同步实现、
同步测试，但默认关闭。需要单独运行 trained-GCN 时使用
`configs/v16n/stable_trained_gcn.toml`；它保持相同数据和训练超参数，只限定 GCN 方法并使用
独立结果根目录。stable 结果不能与 `legacy_v1` 历史产物混放。

物理域虚拟节点的四臂筛选使用 `configs/v16n/domain_gcn_screen.toml`。它是无硬预算的结构识别
实验；`configs/v16n/domain_gcn_k32_diagnostic.toml` 仅保留 K=32 归档失效的复现配置，不能用于
模型优劣结论。冻结选择的 source-test 汇总命令为：

`PYTHONPATH=src python src/run_domain_gcn_screen_eval.py --config configs/v16n/domain_gcn_screen.toml`

旧的无 TOML stage2 配置与命令保留用于历史复现，不再作为新实验的扩展点。
