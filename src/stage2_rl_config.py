"""阶段 2 雷达舰船 RL 实验的代码内固定配置。

本项目当前约定不通过命令行切换实验。要改变数据、种子、搜索预算或最终评价参数，必须先修改
本文件并接受代码审查；正式实验入口只读取这里的常量：

1. ``run_basic_baselines.py`` 可选运行 All Features/KBest 的 LR 对照；
2. ``run_stage2_rl_selection.py`` 在源 train 数据内执行 5 折 Decision-Tree 引导的 RL 特征选择；
3. ``run_stage2_dt_test.py`` 用全部 development 训练 DT，并在 test 上比较六组特征；
4. ``run_stage2_rl_final_lr.py`` 可选读取前一步保存的特征编号并调用统一 ``lr_final``。
"""

from __future__ import annotations

from pathlib import Path

# 数据和重复实验设置。
DATASET = "radar_ship"
DATA_DIR = "../dataset"
DATA_VERSION = "v16n"
EXPECTED_CLEAN_FEATURES = 65
SEEDS = (42, 43, 44, 45, 46)

# RL 搜索设置。三个方法共享预算、数据划分、DT probe 和其余超参数。
VALIDATION_FRACTION = 0.25
EXPLORATION_STEP_BUDGET = 250
HYBRID_SWITCH_STEP = 83
HYBRID_WITHDRAW_STEP = 166
INNER_CV_FOLDS = 5

# 每个方法都会在完成时立即写 selection.json；重跑时只复用与当前配置完全匹配的产物。
RESUME_COMPLETED_SELECTIONS = True
TRAJECTORY_ROLLING_WINDOW = 25

# 使用筛选出的特征进行统一 Logistic Regression 验证。
LR_C = 1.0
LR_SOLVER = "liblinear"
LR_MAX_ITER = 5000
LR_CLASS_WEIGHT = "balanced"

# 搜索和最终评价刻意使用不同根目录，防止把测试指标混进 RL 训练产物。
EXPERIMENT_PREFIX = f"radar_ship_{DATA_VERSION}_stage2"
SELECTION_ROOT = Path("experiments") / f"{EXPERIMENT_PREFIX}_rl_selection"
FINAL_LR_ROOT = Path("experiments") / f"{EXPERIMENT_PREFIX}_rl_final_lr"
TABLE_ROOT = Path("results") / "tables"
DT_TEST_ROOT = Path("experiments") / f"{EXPERIMENT_PREFIX}_dt_test"
BASIC_ROOT = Path("experiments") / f"radar_ship_{DATA_VERSION}_basic_lr"
TABLE_PREFIX = EXPERIMENT_PREFIX
K_BEST = EXPECTED_CLEAN_FEATURES // 2

# Full-IRFS-fixed 的相关性惩罚强度扫描。少跑一个种子以控制总耗时；所有 RL 搜索完成后，
# 扫描结束后再由独立入口在源 test 上做 Decision-Tree 最终验证。
BETA_SWEEP_VALUES = (0.0, 0.02, 0.1, 0.5)
BETA_SWEEP_SEEDS = SEEDS[:-1]
BETA_SWEEP_SELECTION_ROOT = Path("experiments") / f"{EXPERIMENT_PREFIX}_beta_sweep_selection"
BETA_SWEEP_DT_TEST_ROOT = Path("experiments") / f"{EXPERIMENT_PREFIX}_beta_sweep_dt_test"
BETA_SWEEP_TABLE_PREFIX = f"{EXPERIMENT_PREFIX}_beta_sweep"
BETA_SWEEP_GCN_SELECTION_ROOT = Path("experiments") / f"{EXPERIMENT_PREFIX}_beta_sweep_gcn_selection"
BETA_SWEEP_GCN_DT_TEST_ROOT = Path("experiments") / f"{EXPERIMENT_PREFIX}_beta_sweep_gcn_dt_test"
BETA_SWEEP_GCN_TABLE_PREFIX = f"{EXPERIMENT_PREFIX}_beta_sweep_gcn"

# 固定 beta 后扫描 Hybrid Teaching 三阶段的步数比例。每组总预算仍为 250；名称后的三个
# 数字依次表示 MI relevance / DT importance / withdrawn 的步数。
GUIDANCE_SWEEP_BETA = 0.5
GUIDANCE_SCHEDULE_SPECS = (
    ("thirds_83_83_84", 83, 166),
    ("mi_heavy_200_25_25", 200, 225),
    ("mi_only_250_0_0", 250, 250),
    ("dt_heavy_25_200_25", 25, 225),
)
GUIDANCE_SWEEP_SELECTION_ROOT = Path("experiments") / f"{EXPERIMENT_PREFIX}_guidance_sweep_selection"
GUIDANCE_SWEEP_DT_TEST_ROOT = Path("experiments") / f"{EXPERIMENT_PREFIX}_guidance_sweep_dt_test"
GUIDANCE_SWEEP_TABLE_PREFIX = f"{EXPERIMENT_PREFIX}_guidance_sweep"

# 固定低相关性权重后，扫描“超预算惩罚”强度。RL 仍可访问任意非退化子集，但最终候选
# 只能来自 |S| <= K_BEST 的初始子集或轨迹；全扫描完成后再用源 test 做最终评价。
BUDGET_SWEEP_BETA = 0.02
BUDGET_SWEEP_FEATURE_BUDGET = K_BEST
BUDGET_SWEEP_VALUES = (0.01, 0.025, 0.05, 0.1)
BUDGET_SWEEP_SEEDS = BETA_SWEEP_SEEDS
BUDGET_SWEEP_SELECTION_ROOT = Path("experiments") / f"{EXPERIMENT_PREFIX}_budget_sweep_selection"
BUDGET_SWEEP_DT_TEST_ROOT = Path("experiments") / f"{EXPERIMENT_PREFIX}_budget_sweep_dt_test"
BUDGET_SWEEP_TABLE_PREFIX = f"{EXPERIMENT_PREFIX}_budget_sweep"

# 纯 Accuracy 控制：奖励中 beta=lambda=0，但保留 |S|<=K_BEST 的最终候选硬筛选。
ACCURACY_ONLY_SELECTION_ROOT = Path("experiments") / f"{EXPERIMENT_PREFIX}_accuracy_only_selection"
ACCURACY_ONLY_DT_TEST_ROOT = Path("experiments") / f"{EXPERIMENT_PREFIX}_accuracy_only_dt_test"
ACCURACY_ONLY_TABLE_PREFIX = f"{EXPERIMENT_PREFIX}_accuracy_only"


# (报告名称, 引擎注册名, 状态编码)。MARLFS 按源码设计忽略 state_encoder 并使用最小状态。
RL_METHOD_SPECS = (
    ("marlfs", "marlfs", "minimal_relevance_redundancy"),
    ("full_irfs_fixed", "full_irfs", "fixed"),
    ("full_irfs_trained_gcn", "full_irfs", "trained_gcn"),
)

# 当前复跑矩阵。trained-GCN 保留在完整方法注册表中，但本轮按实验要求暂不执行。
RL_METHODS_TO_RUN = ("marlfs", "full_irfs_fixed")
ACTIVE_RL_METHOD_SPECS = tuple(spec for spec in RL_METHOD_SPECS if spec[0] in RL_METHODS_TO_RUN)
