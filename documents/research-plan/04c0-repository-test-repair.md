# 任务 4C-0：仓库兼容边界与全量测试修复

日期：2026-09-16  
状态：**PASS — 允许进入任务 4C 的工程门槛**

## 1. 范围与结论

本任务只修复仓库兼容边界和全量测试可信度，没有运行正式 PPO/DQN，没有科研调参，也没有修改
4A/4B 的协议、配置、manifest、实验结果或已冻结 hash。

最终结果：

- pytest 收集：174 tests，0 collection errors；
- 全量 pytest：174 executed，174 passed，0 failed，0 skipped，0 errors；
- 4A、4B、PPO、harness、invariants、methods、stable architecture 和 stage2 测试均实际执行；
- legacy-data profile：未启用，计数为 0；
- Ruff：全部本任务修改文件通过；
- 工程准入：**PASS**。这只解除 4C 的仓库测试门槛，不改写 4B 的整体 NO-GO，也不改变
  4C 仅限 K=32 post-selection development robustness check 的研究边界。

## 2. 修改前基线

修复前先在原工作树执行了两条要求的命令。原始输出保存在：

- `documents/research-plan/04c0-logs/pytest-collect-only-before.txt`
- `documents/research-plan/04c0-logs/pytest-before.txt`

为确保输出在修复后仍可审计，日志又在隔离的 pre-fix 临时快照中原样复现；错误集合和计数与首次
工作树运行一致。

| 命令 | collected | executed | passed | failed | skipped | collection errors | 状态 |
|---|---:|---:|---:|---:|---:|---:|---|
| `pytest --collect-only -q` | 110 | 0 | 0 | 0 | 0 | 11 | exit 2 |
| `pytest -q` | 110 可收集项 | 0 | 0 | 0 | 0 | 11 | 收集阶段中止，exit 2 |

11 个收集错误与任务说明一致：

1. `tests/test_basic_baselines.py`
2. `tests/test_engine.py`
3. `tests/test_harness.py`
4. `tests/test_invariants.py`
5. `tests/test_methods.py`
6. `tests/test_stable_architecture.py`
7. `tests/test_stage2_beta_sweep.py`
8. `tests/test_stage2_budget_sweep.py`
9. `tests/test_stage2_dt_test.py`
10. `tests/test_stage2_guidance_sweep.py`
11. `tests/test_stage2_rl.py`

## 3. Git history 证据与兼容决策

关键历史：

- `ba1c79d`（refactor: add stable RL training core）把 3,048 行历史 stage2 orchestration
  隔离到 `radar_ship_fs.legacy.stage2`，顶层 `run_stage2_*.py` 变成最多 12 行的兼容 wrapper；
  同时用架构测试禁止 stable 包反向 import legacy/stage2。
- `5f1cfe5`（Archive unsuccessful experiment parameter sweeps）把部分顶层 stage2 wrapper
  移入 `src/archive/`，但测试和历史 import 契约仍使用顶层名称。
- `bc9ad50` 删除了 `methods.advice`、`methods.configure` 及其组合子模块和整个
  `radar_ship_fs.legacy`，同时没有迁移 `src/run_irfs.py`、`src/harness/aggregate.py`、
  stage2 wrapper 与测试。
- 同一提交还把 `data.loader.load()` 从配置驱动的 WDBC/Parkinson's/radar dispatcher
  覆盖为无条件 radar v10 loader，并移除了 grouped split；`IrfsConfig.dataset` 默认仍是
  `wdbc`，测试与 harness 文档也仍声明该契约。

据此判定：

- `methods.configure/advice` 不是已退役测试残留：active `run_irfs.py` 和
  `harness.aggregate` 仍直接依赖它们，应恢复最后正常的组合 facade 和适配层。
- `radar_ship_fs.legacy.stage2` 是已经设计好的隔离边界，应恢复冻结实现；不能把 stable
  模块改回依赖 legacy。
- 被归档的顶层 stage2 名称仍是 README、历史命令、测试 monkeypatch 和 wrapper 架构测试承诺的
  兼容入口；只恢复薄 wrapper，不把实现复制回顶层。
- 缺失 `data/sim_ship_cr_v10.*` 不能划入 legacy-data profile，因为 stable checkpoint、
  stable RL、harness 和 invariants 同样失败。正确修复是恢复配置驱动的 loader/splitter，而不是
  skip 或伪造数据。

## 4. 根因到修复映射

| 受阻模块/路径 | 根因 | 决策与修复 | 保护证据 |
|---|---|---|---|
| basic baselines、stage2 DT/RL | 现有薄 wrapper 指向被删 `radar_ship_fs.legacy.stage2` | 恢复冻结 legacy stage2 实现与 selector 边界 | stage2、basic baseline、stable architecture tests |
| engine、invariants | `methods.advice` 被删 | 恢复 trainer-to-engine 的窄 advice adapter | engine 与 invariants tests |
| methods、`run_irfs.py` | `methods.configure` 及其拆分子模块被删 | 恢复 facade、engine builders、seam adapters、reinforced runner、suite | methods 与 invariants tests |
| `harness.aggregate` | active import `REINFORCED_METHOD_NAMES` 悬空 | 由恢复的 configure facade 继续提供稳定 registry | harness tests |
| stable architecture | legacy selector 包被删 | 恢复只暴露 Selector contract 的 `radar_ship_fs.legacy` | stable package import 扫描为 0 violations |
| beta/budget/guidance wrappers | 顶层名称被归档但测试/历史命令未迁移 | 从已知正常历史恢复 12 行薄 wrapper，真实实现留在 legacy namespace | 全部 stage2 wrapper AST 检查 |
| 29 failures + 25 setup errors | `load()` 无视 `config.dataset`，默认 WDBC 被错误送入缺失 v10 路径 | 合并恢复 WDBC、Parkinson's、radar dispatcher，保留现有 source-train/source-test helpers | data、4B、unified baseline tests |
| grouped dataset split | `GroupShuffleSplit` 路径被删 | 恢复 group-disjoint train/validation/test 切分 | Parkinson's group integrity test |

## 5. 实现边界

恢复/新增的兼容文件包括：

- `src/methods/advice.py`
- `src/methods/configure.py`
- `src/methods/engine_builders.py`
- `src/methods/reinforced_run.py`
- `src/methods/seam_adapters.py`
- `src/methods/suite.py`
- `src/radar_ship_fs/legacy/__init__.py`
- `src/radar_ship_fs/legacy/selector.py`
- `src/radar_ship_fs/legacy/stage2/*.py`
- 7 个恢复的顶层 `src/run_stage2_*.py` 薄 wrapper。

迁移合并的 active 文件：

- `src/data/loader.py`：恢复通用 dispatcher，同时保留 4A/4B/v16n 使用的
  `load_radar_ship_source_train`、`load_radar_ship_source_test` 和 `load_radar_ship`。
- `src/data/splitter.py`：恢复 grouped split；radar 的 predefined source-test 边界仍优先。

没有修改 stable `src/radar_ship_fs/{rl,ppo,experiment,selection}` 对 legacy 的依赖方向。
`test_stable_package_never_imports_legacy_or_stage2_scripts` 实际通过。

## 6. 执行中发现的数据问题

首次解除 collection errors 后，全量 suite 已真实跑到结束，暴露了第二层问题：

| 阶段 | collected | executed | passed | failed | skipped | runtime/setup errors | collection errors |
|---|---:|---:|---:|---:|---:|---:|---:|
| 兼容层恢复后、数据修复前 | 174 | 174 | 120 | 29 | 0 | 25 | 0 |

29 个失败和 25 个 setup errors 均收敛到无条件读取
`data/sim_ship_cr_v10.train.svm/test.svm`。由于它覆盖 active stable/harness/invariants，
本任务没有创建 legacy-data profile，而是修复 loader/splitter 契约。

## 7. 最终验证

### 7.1 收集与全量执行

| 命令 | collected | executed | passed | failed | skipped | errors |
|---|---:|---:|---:|---:|---:|---:|
| `conda run -n dl-lab pytest --collect-only -q` | 174 | 0 | 0 | 0 | 0 | 0 |
| `conda run -n dl-lab pytest -q` | 174 | 174 | 174 | 0 | 0 | 0 |

全量运行用时 55.63 秒。唯一警告是 4B 测试中的 15 条 sklearn `liblinear` multiclass
future warning，不是失败、skip 或 collection error。

### 7.2 必须实际执行的领域

显式领域切片命令覆盖：

- 4A：`tests/test_action_support.py`
- 4B：`tests/test_reward_alignment.py`
- PPO：`tests/test_ppo_feature_ids.py`、`tests/test_ppo_initialization.py`
- harness：`tests/test_harness.py`
- invariants：`tests/test_invariants.py`
- methods/engine/stable architecture
- 全部五个 stage2 测试模块

结果：**85 passed，0 failed，0 skipped，0 errors**，用时 49.75 秒。

loader/scorer/v16n 定向切片：

- `tests/test_data.py`
- `tests/test_reward_alignment.py`
- `tests/test_unified_baselines.py`

结果：**18 passed，0 failed，0 skipped，0 errors**。

### 7.3 Ruff

对所有恢复或合并文件运行 `ruff check`，结果：**All checks passed**。

### 7.4 冻结产物完整性

下列 hash 与 4A/4B delivery manifest 记录一致：

| 文件 | SHA-256 |
|---|---|
| `documents/research-plan/04a-action-support.md` | `ae569852f63c17c45d073584ab7cbaa7e8c7e1fc926ef4d1cf18b9bf03692dc4` |
| `experiments/action_support_v1/manifest.json` | `3c4527ddefad15579944372359e69214ebf772fece6a8ff72c3570ba858418f6` |
| `experiments/action_support_v1/audit.json` | `1f64a8d21d158e23c8861ee00f5cd48c472a1544066978b5d203e23993873256` |
| `documents/research-plan/04b-reward-alignment.md` | `e7ba33675f4535beb4dd1fa063623def73784ee60dc97c9954242e241868a1ed` |
| `experiments/reward_alignment_v1/manifest.json` | `3e92bb438d120c3d1b6fb4cf8b6dfc327930aab0318405e4d27b804cf7672dfc` |
| `experiments/reward_alignment_v1/audit.json` | `4a2b7859fd1e7abf898e660f9c2e5209008cdf5f7b8bfab3bb7b78abc5421e33` |
| `experiments/reward_alignment_v1/verification.json` | `6bd6b64b4d51f96dd87f2cea0810a9202dc8f05cd17fc756f28b5c9935f6a8ec` |

## 8. 最终决策

**任务 4C-0：PASS。允许进入任务 4C。**

理由是工程硬门槛全部满足：0 collection errors、全量 active suite 174/174 通过、必需研究与
训练路径实际执行、无 skip/xfail 掩盖、无 legacy-data 例外、stable/legacy 依赖方向保持隔离、
修改文件 Ruff 通过、4A/4B 冻结产物未改。

该 PASS 不表示 4B 变为 GO，不证明 PPO/DQN 有效，也不授权扩大 K/reward 搜索；任务 4C 仍必须按
路线图只执行锁定的 K=32 development robustness check。

