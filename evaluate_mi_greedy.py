import numpy as np
from radar_ship_fs.experiment.config import load_experiment_spec
from stage2_cv import build_stage2_cv_context
from methods.mi_greedy import MIGreedySelector
from run_domain_gcn_screen_eval import _dt_metrics

spec = load_experiment_spec("configs/v16n/run_experiments.toml")

dt_accs = []
for seed in spec.dataset.seeds:
    context = build_stage2_cv_context(
        spec.irfs_config(spec.enabled_methods[0]),
        seed=seed,
        n_splits=spec.dataset.inner_cv_folds,
    )
    selector = MIGreedySelector()
    selection = selector.select(context)
    subset = selection.selected
    metrics = _dt_metrics(context, spec.irfs_config(spec.enabled_methods[0]), subset, seed)
    dt_accs.append(metrics["dt_test_accuracy"])
    print(f"seed={seed} k={len(subset)} test_acc={metrics['dt_test_accuracy']:.4f}")
print(f"\nmethod=mi_greedy  dt_test_accuracy_mean={np.mean(dt_accs):.4f}")
