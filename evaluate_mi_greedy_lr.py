import numpy as np
from radar_ship_fs.experiment.config import load_experiment_spec
from stage2_cv import build_stage2_cv_context
from methods.mi_greedy import MIGreedySelector
from harness.lr_final import score_selected_features_with_lr

spec = load_experiment_spec("configs/v16n/run_experiments.toml")
lr_accs = []
for seed in spec.dataset.seeds:
    context = build_stage2_cv_context(
        spec.irfs_config(spec.enabled_methods[0]),
        seed=seed,
        n_splits=spec.dataset.inner_cv_folds,
    )
    selector = MIGreedySelector()
    subset = selector.select(context).selected
    lr_metrics = score_selected_features_with_lr(
        context.split.train.X, context.split.train.y,
        context.split.test.X, context.split.test.y,
        subset, random_state=seed
    )
    lr_accs.append(lr_metrics.test_accuracy)

print(f"mi_greedy lr={np.mean(lr_accs):.4f}")
