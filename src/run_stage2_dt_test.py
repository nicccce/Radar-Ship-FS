#!/usr/bin/env python3
"""Evaluate baselines and all three RL-selected feature sets on each outer test split.

This entry never reruns RL. It reads the inner-CV selections, fits fresh Decision Trees on all
development rows, and evaluates them once on the sealed test rows.
"""

from __future__ import annotations

import json
import time
from typing import Any, Sequence

import numpy as np

from data.splitter import Partition
from harness.contract import SelectionContext
from harness.orchestrator import build_run_context
from probe import DecisionTreeProbe
from run_stage2_rl_selection import (
    _config_for_encoder,
    _summary,
    _write_csv,
    _write_json,
)
from stage2_rl_config import (
    DATASET,
    DT_TEST_ROOT,
    EXPECTED_CLEAN_FEATURES,
    INNER_CV_FOLDS,
    K_BEST,
    SEEDS,
    SELECTION_ROOT,
    TABLE_PREFIX,
    TABLE_ROOT,
)


def _load_rl_selection(
    seed: int,
    method: str,
    development_indices: np.ndarray,
) -> dict[str, Any]:
    """Read one existing RL selection and verify its inner-CV development rows."""
    path = SELECTION_ROOT / f"seed-{seed}" / method / "selection.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing {method} selection: {path}")
    with path.open("r", encoding="utf-8") as handle:
        artifact = json.load(handle)

    signature = artifact.get("experiment_signature", {})
    protocol = artifact.get("protocol", {})
    if (
        signature.get("dataset") != DATASET
        or signature.get("seed") != seed
        or signature.get("report_name") != method
    ):
        raise ValueError(f"RL selection identity mismatch in {path}")
    if (
        protocol.get("official_test_accessed") is not False
        or protocol.get("held_out_random_test_accessed") is not False
    ):
        raise ValueError(f"RL selection accessed test during search: {path}")

    saved_split = artifact.get("split_indices", {})
    if saved_split.get("development_inner_cv") != development_indices.astype(int).tolist():
        raise ValueError(f"development rows differ from RL run: {path}")
    if not artifact.get("selected_clean_indices"):
        raise ValueError(f"RL selection has no selected features: {path}")
    return artifact


def _development_partition(context: SelectionContext) -> Partition:
    """Combine search train and validation for final Decision-Tree fitting."""
    train = context.split.train
    validation = context.split.validation
    groups = None
    if train.groups is not None and validation.groups is not None:
        groups = np.concatenate((train.groups, validation.groups))
    return Partition(
        X=np.vstack((train.X, validation.X)),
        y=np.concatenate((train.y, validation.y)),
        indices=np.concatenate((train.indices, validation.indices)),
        feature_names=train.feature_names,
        groups=groups,
        metadata=train.metadata,
    )


def _mi_ranking(
    X_development: np.ndarray,
    y_development: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit Mutual Information on development rows and return descending ranking."""
    from sklearn.feature_selection import mutual_info_classif

    scores = mutual_info_classif(X_development, y_development, random_state=seed)
    ranked = np.argsort(-np.nan_to_num(scores, nan=-np.inf), kind="stable")
    return ranked, scores


def _top_k(ranked: np.ndarray, k: int) -> tuple[int, ...]:
    if not 1 <= k <= ranked.size:
        raise ValueError(f"k must be in [1, {ranked.size}], got {k}")
    return tuple(sorted(int(index) for index in ranked[:k]))


def _run_seed(seed: int) -> dict[str, Any]:
    config = _config_for_encoder("fixed")
    context = build_run_context(config, seed=seed)
    if context.n_features != EXPECTED_CLEAN_FEATURES:
        raise ValueError(f"expected {EXPECTED_CLEAN_FEATURES} cleaned features, got {context.n_features}")
    metadata = context.split.train.metadata
    if metadata is None:
        raise ValueError("radar metadata is required")
    original_ids = [int(value) for value in metadata["final_feature_ids"]]

    development = _development_partition(context)
    rl_selections = {
        method: _load_rl_selection(seed, method, development.indices)
        for method in ("marlfs", "full_irfs_fixed", "full_irfs_trained_gcn")
    }
    rl_features = {
        method: tuple(int(index) for index in artifact["selected_clean_indices"])
        for method, artifact in rl_selections.items()
    }
    fixed_features = rl_features["full_irfs_fixed"]

    test = context.split.release_test_for_final_metrics()
    ranked, mi_scores = _mi_ranking(development.X, development.y, seed)
    final_probe = DecisionTreeProbe(development, config, context.rng)

    candidates = (
        (
            "all_features_54",
            tuple(range(context.n_features)),
            "all 54 features",
            None,
        ),
        (
            "kbest_mutual_info_27",
            _top_k(ranked, K_BEST),
            "MI fit on all development rows; k=27",
            None,
        ),
        (
            "marlfs_selected",
            rl_features["marlfs"],
            "features read unchanged from the inner-CV MARLFS selection",
            rl_selections["marlfs"]["best_dt_inner_cv_accuracy"],
        ),
        (
            "full_irfs_fixed_selected",
            rl_features["full_irfs_fixed"],
            "features read unchanged from the inner-CV Full-IRFS-fixed selection",
            rl_selections["full_irfs_fixed"]["best_dt_inner_cv_accuracy"],
        ),
        (
            "full_irfs_trained_gcn_selected",
            rl_features["full_irfs_trained_gcn"],
            "features read unchanged from the inner-CV Full-IRFS-trained-GCN selection",
            rl_selections["full_irfs_trained_gcn"]["best_dt_inner_cv_accuracy"],
        ),
        (
            "kbest_mutual_info_matched_fixed_size",
            _top_k(ranked, len(fixed_features)),
            "MI fit on all development rows; k matches Full-IRFS-fixed for this seed",
            None,
        ),
    )

    methods: list[dict[str, Any]] = []
    for name, features, selection_rule, selection_cv_accuracy in candidates:
        started = time.perf_counter()
        probe_result = final_probe.probe(features, test)
        elapsed = time.perf_counter() - started
        train_accuracy = float(
            probe_result.tree.score(
                development.X[:, np.asarray(features, dtype=int)],
                development.y,
            )
        )
        method = {
            "name": name,
            "selection_rule": selection_rule,
            "selected_clean_indices": list(features),
            "selected_original_feature_ids": [original_ids[index] for index in features],
            "selected_count": len(features),
            "compression_ratio": float(1.0 - len(features) / context.n_features),
            "dt_development_accuracy": train_accuracy,
            "dt_test_accuracy": float(probe_result.accuracy),
            "dt_fit_and_test_seconds": elapsed,
            "selection_best_dt_inner_cv_accuracy": selection_cv_accuracy,
        }
        methods.append(method)
        print(
            f"seed={seed} {name:<39} features={len(features):>2} "
            f"dt_train={train_accuracy:.4f} dt_test={probe_result.accuracy:.4f}",
            flush=True,
        )

    result = {
        "protocol": {
            "stage": "stage2_dt_final_test_comparison",
            "rl_retrained": False,
            "rl_selected_features_modified": False,
            "development_fit_rows": int(development.X.shape[0]),
            "held_out_random_test_rows": int(test.X.shape[0]),
            "test_role": "final_evaluation_only",
            "row_split": "merge source files; outer stratified 80/20; RL uses inner 5-fold CV",
            "final_model": "DecisionTreeClassifier through DecisionTreeProbe",
            "lr_final_called": False,
            "selection_sources": {
                method: str(SELECTION_ROOT / f"seed-{seed}" / method / "selection.json")
                for method in rl_selections
            },
        },
        "seed": seed,
        "config": {
            "inner_cv_folds": INNER_CV_FOLDS,
            "kbest_k": K_BEST,
            "matched_k": len(fixed_features),
            "test_fraction": config.test_fraction,
            "validation_fraction": config.validation_fraction,
        },
        "dataset_metadata": metadata,
        "split_indices": {
            "development_train_plus_validation": development.indices.astype(int).tolist(),
            "held_out_test": test.indices.astype(int).tolist(),
        },
        "mutual_information_scores": {
            str(original_ids[index]): float(mi_scores[index]) for index in range(context.n_features)
        },
        "methods": methods,
    }
    _write_json(result, DT_TEST_ROOT / f"seed-{seed}" / "results.json")
    return result


def _aggregate(
    seed_results: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    names = [method["name"] for method in seed_results[0]["methods"]]
    fixed_by_seed = {
        result["seed"]: next(
            method["dt_test_accuracy"]
            for method in result["methods"]
            if method["name"] == "full_irfs_fixed_selected"
        )
        for result in seed_results
    }
    method_summaries: list[dict[str, Any]] = []
    flat_rows: list[dict[str, Any]] = []

    for name in names:
        rows = [
            next(method for method in result["methods"] if method["name"] == name) for result in seed_results
        ]
        deltas = [
            float(row["dt_test_accuracy"] - fixed_by_seed[result["seed"]])
            for result, row in zip(seed_results, rows)
        ]
        cv_values = [
            float(row["selection_best_dt_inner_cv_accuracy"])
            for row in rows
            if row.get("selection_best_dt_inner_cv_accuracy") is not None
        ]
        method_summaries.append(
            {
                "name": name,
                "n_seeds": len(rows),
                "selected_count": _summary([row["selected_count"] for row in rows]),
                "compression_ratio": _summary([row["compression_ratio"] for row in rows]),
                "selection_best_dt_inner_cv_accuracy": _summary(cv_values) if cv_values else None,
                "dt_development_accuracy": _summary([row["dt_development_accuracy"] for row in rows]),
                "dt_test_accuracy": _summary([row["dt_test_accuracy"] for row in rows]),
                "delta_vs_full_irfs_fixed": _summary(deltas),
                "win_tie_loss_vs_full_irfs_fixed": {
                    "win": sum(delta > 1e-12 for delta in deltas),
                    "tie": sum(abs(delta) <= 1e-12 for delta in deltas),
                    "loss": sum(delta < -1e-12 for delta in deltas),
                },
            }
        )
        for result, row, delta in zip(seed_results, rows, deltas):
            flat_rows.append(
                {
                    "seed": result["seed"],
                    "method": name,
                    "selected_count": row["selected_count"],
                    "compression_ratio": row["compression_ratio"],
                    "selection_best_dt_inner_cv_accuracy": row.get("selection_best_dt_inner_cv_accuracy"),
                    "dt_development_accuracy": row["dt_development_accuracy"],
                    "dt_test_accuracy": row["dt_test_accuracy"],
                    "delta_vs_full_irfs_fixed": delta,
                    "dt_fit_and_test_seconds": row["dt_fit_and_test_seconds"],
                    "selected_original_feature_ids": ";".join(
                        str(value) for value in row["selected_original_feature_ids"]
                    ),
                }
            )

    aggregate = {
        "dataset": DATASET,
        "seeds": list(SEEDS),
        "protocol": ("RL selects by 5-fold inner-CV DT; final DT fits development and evaluates outer test"),
        "methods": method_summaries,
    }
    return aggregate, flat_rows


def _aggregate_csv_rows(aggregate: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "method": method["name"],
            "n_seeds": method["n_seeds"],
            "selected_count_mean": method["selected_count"]["mean"],
            "selected_count_std": method["selected_count"]["std"],
            "selection_best_dt_inner_cv_accuracy_mean": (
                method["selection_best_dt_inner_cv_accuracy"]["mean"]
                if method["selection_best_dt_inner_cv_accuracy"] is not None
                else ""
            ),
            "dt_development_accuracy_mean": method["dt_development_accuracy"]["mean"],
            "dt_test_accuracy_mean": method["dt_test_accuracy"]["mean"],
            "dt_test_accuracy_std": method["dt_test_accuracy"]["std"],
            "delta_vs_full_irfs_fixed_mean": method["delta_vs_full_irfs_fixed"]["mean"],
            **method["win_tie_loss_vs_full_irfs_fixed"],
        }
        for method in aggregate["methods"]
    ]


def main() -> None:
    print(
        f"held-out DT test: seeds={list(SEEDS)} kbest={K_BEST}; "
        "fit all development rows; evaluate outer test; RL not rerun; LR absent",
        flush=True,
    )
    seed_results = [_run_seed(seed) for seed in SEEDS]
    aggregate, flat_rows = _aggregate(seed_results)
    aggregate_rows = _aggregate_csv_rows(aggregate)

    _write_json(aggregate, DT_TEST_ROOT / "aggregate.json")
    _write_csv(flat_rows, DT_TEST_ROOT / "per_seed_results.csv")
    _write_csv(aggregate_rows, DT_TEST_ROOT / "aggregate.csv")
    _write_csv(flat_rows, TABLE_ROOT / f"{TABLE_PREFIX}_dt_test_per_seed.csv")
    _write_csv(
        aggregate_rows,
        TABLE_ROOT / f"{TABLE_PREFIX}_dt_test_aggregate.csv",
    )

    print("\naggregate:")
    for method in aggregate["methods"]:
        accuracy = method["dt_test_accuracy"]
        delta = method["delta_vs_full_irfs_fixed"]
        print(
            f"{method['name']:<39} features={method['selected_count']['mean']:.1f} "
            f"dt_test={accuracy['mean']:.4f}±{accuracy['std']:.4f} "
            f"delta_vs_fixed={delta['mean']:+.4f}",
            flush=True,
        )
    print(f"artifacts: {DT_TEST_ROOT / 'aggregate.json'}", flush=True)


if __name__ == "__main__":
    main()
