# 实验配置

`configs/v16n/stable.toml` 是新的 `stable_v1` 正式入口配置。配置由严格 schema 一次性校验：
未知字段、非法枚举、无效 DQN 参数、空运行矩阵和 Hybrid 边界错误都会在数据加载前失败。

```bash
python -m radar_ship_fs.experiment dry-run --config configs/v16n/stable.toml
python -m radar_ship_fs.experiment run --config configs/v16n/stable.toml --resume
```

CLI 只允许选择配置、恢复策略，以及过滤 TOML 中已经声明的 seed/method。学习率、batch、
replay、target 同步等参数必须修改 TOML，使 manifest 中的规范化配置与实际运行始终一致。

默认矩阵启用 `marlfs/minimal` 和 `full_irfs_fixed/fixed`；`trained_gcn` 与 stable trainer 同步实现、
同步测试，但默认关闭。stable 结果必须使用独立根目录，不能与 `legacy_v1` 历史产物混放。

旧的无 TOML stage2 配置与命令保留用于历史复现，不再作为新实验的扩展点。
