import numpy as np
import time
from radar_ship_fs.experiment.config import load_experiment_spec
from stage2_cv import build_stage2_cv_context
from methods.dt_rfe import DTImportanceEliminator
from methods.mrmr import MRMRSelector
from methods.l1 import L1Selector
from methods.mi_greedy import MIGreedySelector
from harness.lr_final import score_selected_features_with_lr
from run_domain_gcn_screen_eval import _dt_metrics

spec = load_experiment_spec("configs/v16n/run_experiments.toml")
methods = {
    "dt_rfe": DTImportanceEliminator,
    "mrmr": MRMRSelector,
    "l1": L1Selector,
    "mi_greedy": MIGreedySelector
}

for method_name, method_cls in methods.items():
    dt_accs, lr_accs, counts, times = [], [], [], []
    for seed in spec.dataset.seeds:
        context = build_stage2_cv_context(
            spec.irfs_config(spec.enabled_methods[0]),
            seed=seed,
            n_splits=spec.dataset.inner_cv_folds,
        )
        
        start = time.perf_counter()
        selector = method_cls()
        subset = selector.select(context).selected
        elapsed = time.perf_counter() - start
        
        dt_metrics = _dt_metrics(context, spec.irfs_config(spec.enabled_methods[0]), subset, seed)
        dt_accs.append(dt_metrics["dt_test_accuracy"])
        
        lr_metrics = score_selected_features_with_lr(
            context.split.train.X, context.split.train.y,
            context.split.test.X, context.split.test.y,
            subset, random_state=seed
        )
        lr_accs.append(lr_metrics.test_accuracy)
        counts.append(len(subset))
        times.append(elapsed)
        
    print(f"{method_name}: K={np.mean(counts):.1f}, time={np.mean(times):.1f}s, dt={np.mean(dt_accs):.4f}, lr={np.mean(lr_accs):.4f}")
