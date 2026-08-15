#!/usr/bin/env python3
"""Evaluate a completed four-arm domain-GCN screen without rerunning selection."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

from harness.lr_final import lr_metrics_to_dict, score_selected_features_with_lr
from probe import DecisionTreeProbe
from radar_ship_fs.experiment.artifact import ArtifactStore, development_fingerprint
from radar_ship_fs.experiment.config import load_experiment_spec
from radar_ship_fs.feedback.encoders import radar_feature_domains
from radar_ship_fs.rl.checkpoint import CheckpointStore
from rng import SeededRng
from stage2_cv import build_stage2_cv_context


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _mean_abs_correlation(X: np.ndarray, subset: tuple[int, ...]) -> float:
    if len(subset) < 2:
        return 0.0
    correlation = np.corrcoef(X[:, np.asarray(subset, dtype=int)], rowvar=False)
    upper = np.triu_indices(len(subset), k=1)
    return float(np.nanmean(np.abs(correlation[upper])))


def _jaccard(values: list[tuple[int, ...]]) -> float:
    scores = []
    for left, right in combinations(values, 2):
        left_set, right_set = set(left), set(right)
        scores.append(len(left_set & right_set) / len(left_set | right_set))
    return float(mean(scores)) if scores else 1.0


def _learned_gates(checkpoint_path: Path) -> tuple[float | None, float | None]:
    state = CheckpointStore(checkpoint_path).load()["trainer"]["online"]
    membership = state.get("encoder.membership_gate")
    inter_domain = state.get("encoder.inter_domain_gate")
    membership_value = None if membership is None else float(torch.tanh(membership).item())
    inter_value = None if inter_domain is None else float(torch.tanh(inter_domain).item())
    return membership_value, inter_value


def _dt_metrics(context, config, subset: tuple[int, ...], seed: int) -> dict[str, float]:
    probe = DecisionTreeProbe(context.split.train, config, SeededRng.from_seed(seed))
    result = probe.probe(subset, context.split.test)
    subset_array = np.asarray(subset, dtype=int)
    train_prediction = result.tree.predict(context.split.train.X[:, subset_array])
    test_prediction = result.tree.predict(context.split.test.X[:, subset_array])
    probabilities = result.tree.predict_proba(context.split.test.X[:, subset_array])
    positive_label = int(np.unique(context.split.train.y)[-1])
    positive_column = int(np.flatnonzero(result.tree.classes_ == positive_label)[0])
    binary_target = (context.split.test.y == positive_label).astype(int)
    return {
        "dt_train_accuracy": float(np.mean(train_prediction == context.split.train.y)),
        "dt_test_accuracy": float(result.accuracy),
        "dt_test_balanced_accuracy": float(balanced_accuracy_score(context.split.test.y, test_prediction)),
        "dt_test_f1": float(
            f1_score(
                context.split.test.y,
                test_prediction,
                pos_label=positive_label,
                zero_division=0,
            )
        ),
        "dt_test_roc_auc": float(roc_auc_score(binary_target, probabilities[:, positive_column])),
    }


def _validate_complete_matrix(spec) -> None:
    missing = []
    for seed in spec.dataset.seeds:
        for method in spec.enabled_methods:
            run_dir = Path(spec.output.root) / f"seed-{seed}" / method.name
            for filename in ("manifest.json", "selection.json"):
                path = run_dir / filename
                if not path.is_file():
                    missing.append(str(path))
    if missing:
        raise FileNotFoundError(
            "all selections must finish before source-test evaluation; missing: " + ", ".join(missing)
        )


def _run_row(spec, seed: int, method, context) -> dict[str, Any]:
    run_dir = Path(spec.output.root) / f"seed-{seed}" / method.name
    selection = _read_json(run_dir / "selection.json")
    signature = selection.get("experiment_signature", {})
    if signature.get("config_hash") != spec.config_hash:
        raise ValueError(f"config hash mismatch in {run_dir}")
    if signature.get("seed") != seed or signature.get("method", {}).get("name") != method.name:
        raise ValueError(f"method/seed identity mismatch in {run_dir}")
    if signature.get("development_fingerprint") != development_fingerprint(context):
        raise ValueError(f"development data fingerprint mismatch in {run_dir}")
    trajectory = selection.get("trajectory", [])
    if method.type == "dqn" and len(trajectory) != spec.training.steps:
        raise ValueError(f"incomplete selection trajectory in {run_dir} (expected {spec.training.steps}, got {len(trajectory)})")

    subset = tuple(int(value) for value in selection["selected_clean_indices"])
    budget = spec.training.feature_budget
    if budget is not None and len(subset) > budget:
        raise ValueError(f"selected subset exceeds feature budget in {run_dir}")

    config = spec.irfs_config(method)
    dt = _dt_metrics(context, config, subset, seed)
    lr = lr_metrics_to_dict(
        score_selected_features_with_lr(
            context.split.train.X,
            context.split.train.y,
            context.split.test.X,
            context.split.test.y,
            subset,
            random_state=seed,
        )
    )
    domains = radar_feature_domains(context)
    if (run_dir / "checkpoint.pt").exists():
        try:
            membership_gate, inter_domain_gate = _learned_gates(run_dir / "checkpoint.pt")
        except Exception:
            membership_gate, inter_domain_gate = None, None
    else:
        membership_gate, inter_domain_gate = None, None
    return {
        "seed": seed,
        "method": method.name,
        "encoder": method.encoder,
        "selected_count": len(subset),
        "selected_clean_indices": list(subset),
        "best_inner_cv_dt_accuracy": float(selection["best_dt_inner_cv_accuracy"]),
        "mean_abs_train_correlation": _mean_abs_correlation(context.split.train.X, subset),
        "selected_domain_count": len({domains[index] for index in subset}),
        "elapsed_seconds": float(trajectory[-1]["elapsed_seconds"]) if trajectory else 0.0,
        "membership_gate": membership_gate,
        "inter_domain_gate": inter_domain_gate,
        **dt,
        **{f"lr_{key}": value for key, value in lr.items() if key != "confusion_matrix"},
        "lr_confusion_matrix": lr["confusion_matrix"],
    }


def _aggregate(rows: list[dict[str, Any]], methods) -> dict[str, Any]:
    numeric = (
        "selected_count",
        "best_inner_cv_dt_accuracy",
        "mean_abs_train_correlation",
        "selected_domain_count",
        "elapsed_seconds",
        "dt_test_accuracy",
        "dt_test_balanced_accuracy",
        "dt_test_f1",
        "dt_test_roc_auc",
        "lr_test_accuracy",
        "lr_balanced_accuracy",
        "lr_f1",
        "lr_roc_auc",
    )
    summary: dict[str, Any] = {}
    by_method: dict[str, list[dict[str, Any]]] = {
        method.name: [row for row in rows if row["method"] == method.name] for method in methods
    }
    for method in methods:
        method_rows = by_method[method.name]
        item: dict[str, Any] = {"runs": len(method_rows)}
        for key in numeric:
            values = [float(row[key]) for row in method_rows]
            item[f"{key}_mean"] = float(mean(values))
            item[f"{key}_std"] = float(stdev(values)) if len(values) > 1 else 0.0
        item["selection_jaccard"] = _jaccard([tuple(row["selected_clean_indices"]) for row in method_rows])
        gates = [row["membership_gate"] for row in method_rows if row["membership_gate"] is not None]
        inter = [row["inter_domain_gate"] for row in method_rows if row["inter_domain_gate"] is not None]
        item["membership_gate_mean"] = None if not gates else float(mean(gates))
        item["inter_domain_gate_mean"] = None if not inter else float(mean(inter))
        summary[method.name] = item

    base_name = methods[0].name
    base_by_seed = {row["seed"]: row for row in by_method[base_name]}
    for method in methods[1:]:
        method_rows = by_method[method.name]
        for metric in ("dt_test_balanced_accuracy", "lr_balanced_accuracy"):
            deltas = [row[metric] - base_by_seed[row["seed"]][metric] for row in method_rows]
            summary[method.name][f"paired_{metric}_delta_vs_{base_name}"] = float(mean(deltas))
            summary[method.name][f"paired_{metric}_wins_vs_{base_name}"] = int(
                sum(delta > 0.0 for delta in deltas)
            )
    return summary


def evaluate(config_path: str) -> dict[str, Any]:
    spec = load_experiment_spec(config_path)
    _validate_complete_matrix(spec)
    rows: list[dict[str, Any]] = []
    first_method = spec.enabled_methods[0]
    for seed in spec.dataset.seeds:
        context = build_stage2_cv_context(
            spec.irfs_config(first_method),
            seed=seed,
            n_splits=spec.dataset.inner_cv_folds,
        )
        for method in spec.enabled_methods:
            row = _run_row(spec, seed, method, context)
            rows.append(row)
            print(
                f"seed={seed} method={method.name:<27} k={row['selected_count']:>2} "
                f"dt_bacc={row['dt_test_balanced_accuracy']:.4f} "
                f"lr_bacc={row['lr_balanced_accuracy']:.4f}",
                flush=True,
            )

    payload = {
        "protocol": {
            "selection_rerun": False,
            "all_selections_completed_before_test_evaluation": True,
            "source_test_role": "screening benchmark",
            "feature_budget": spec.training.feature_budget,
            "config_hash": spec.config_hash,
        },
        "runs": rows,
        "summary": _aggregate(rows, spec.enabled_methods),
    }
    output = Path(spec.output.root) / "evaluation"
    ArtifactStore.write_json(payload, output / "summary.json")
    csv_rows = [
        {
            key: value
            for key, value in row.items()
            if key not in {"selected_clean_indices", "lr_confusion_matrix"}
        }
        for row in rows
    ]
    ArtifactStore.write_csv(csv_rows, output / "runs.csv")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v16n/domain_gcn_screen.toml")
    args = parser.parse_args()
    payload = evaluate(args.config)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
