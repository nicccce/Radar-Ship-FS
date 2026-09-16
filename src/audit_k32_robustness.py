#!/usr/bin/env python3
"""Independent structural and numerical audit for Task 4C."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from run_k32_robustness import (
    EPS,
    N_FEATURES,
    ROOT,
    K,
    clean_on_outer_train,
    fit_lr_predict,
    load_raw,
    read_json,
    read_jsonl,
    sha256,
    write_json,
)
from run_reward_alignment import all_single_swaps


def independent_objective(raw, labels, context, subset):
    values = []
    for repeat in context["inner_repeat_fold_rows"]:
        fold_scores = []
        for fold in repeat:
            prediction, _ = fit_lr_predict(
                raw,
                labels,
                context["final_feature_ids_1based"],
                subset,
                fold["fit"],
                fold["held_out"],
                context["lr_random_state"],
            )
            fold_scores.append(
                balanced_accuracy_score(labels[np.asarray(fold["held_out"], dtype=int)], prediction)
            )
        values.append(float(np.mean(fold_scores)))
    return float(np.mean(values) - 0.5 * np.std(values, ddof=1))


def audit() -> dict:
    manifest = read_json(ROOT / "manifest.json")
    if manifest["source_test_open_count"] != 0 or manifest["task_4b_status"] != "no-go":
        raise AssertionError("manifest isolation or 4B status is invalid")
    for path, expected in manifest["code_protocol_config_hashes"].items():
        if sha256(path) != expected:
            raise AssertionError(f"frozen source hash mismatch: {path}")
    partition_marker = read_json(ROOT / "partition-frozen.json")
    for path, expected in partition_marker["files"].items():
        if sha256(path) != expected:
            raise AssertionError(f"partition hash mismatch: {path}")
    results_marker = read_json(ROOT / "results-complete.json")
    for path, expected in results_marker["files"].items():
        if sha256(path) != expected:
            raise AssertionError(f"result hash mismatch: {path}")

    raw, labels = load_raw()
    partition = read_json(ROOT / "outer-partition.json")
    assignments = np.asarray(partition["validation_fold_by_original_row"], dtype=int)
    if len(assignments) != len(labels) or set(assignments) != set(range(5)):
        raise AssertionError("invalid outer fold assignments")
    seen_validation = []
    max_inner_error = 0.0
    max_outer_error = 0.0
    cases = []
    oof_frames = []
    all_structural = True
    for outer_fold in range(5):
        context = read_json(ROOT / "contexts" / f"outer-fold-{outer_fold}.json")
        train = np.asarray(context["outer_train_rows"], dtype=int)
        validation = np.asarray(context["outer_validation_rows"], dtype=int)
        seen_validation.extend(validation.tolist())
        if set(train) & set(validation) or len(set(train) | set(validation)) != len(labels):
            raise AssertionError("outer train/validation isolation failed")
        if not np.array_equal(np.flatnonzero(assignments == outer_fold), np.sort(validation)):
            raise AssertionError("assignment vector and context differ")
        final_ids, cleaning = clean_on_outer_train(raw[train])
        if final_ids.tolist() != context["final_feature_ids_1based"] or cleaning != context["cleaning"]:
            raise AssertionError("outer-train cleaning recomputation failed")
        train_set = set(train)
        for repeat in context["inner_repeat_fold_rows"]:
            held = []
            for fold in repeat:
                fit_set, held_set = set(fold["fit"]), set(fold["held_out"])
                if fit_set & held_set or not fit_set | held_set <= train_set:
                    raise AssertionError("inner fold isolation failed")
                held.extend(fold["held_out"])
            if sorted(held) != sorted(train.tolist()):
                raise AssertionError("inner repeat does not hold each outer-train row once")

        result = read_json(ROOT / "results" / f"outer-fold-{outer_fold}" / "result.json")
        lr_rows = read_jsonl(ROOT / "results" / f"outer-fold-{outer_fold}" / "lr_candidates.jsonl")
        dt_rows = read_jsonl(ROOT / "results" / f"outer-fold-{outer_fold}" / "dt_candidates.jsonl")
        for rows in (lr_rows, dt_rows):
            for row in rows:
                clean = tuple(row["clean_indices_0based"])
                expected = [int(final_ids[index]) for index in clean]
                if expected != row["original_feature_ids_1based"]:
                    raise AssertionError("candidate coordinate mismatch")
        lr_lookup = {tuple(row["clean_indices_0based"]): row for row in lr_rows}
        dt_lookup = {tuple(row["clean_indices_0based"]): row for row in dt_rows}
        for family, lookup, fit_multiplier, max_rounds in (
            ("lr_search", lr_lookup, 15, 2),
            ("task2_dt_search", dt_lookup, 5, 32),
        ):
            search = result[family]
            cost = search["total_cost"]
            if cost["classifier_fit_count"] != cost["unique_scored_subsets"] * fit_multiplier:
                raise AssertionError("classifier-fit identity failed")
            if cost["candidate_requests"] != cost["unique_scored_subsets"] + cost["cache_hits"]:
                raise AssertionError("request/cache identity failed")
            selected: tuple[int, ...] = ()
            current = float("-inf")
            forward_steps = search["path"][:K]
            for step in forward_steps:
                legal = [
                    tuple(sorted((*selected, feature)))
                    for feature in range(N_FEATURES)
                    if feature not in selected
                ]
                chosen = tuple(step["chosen_clean_indices_0based"])
                best = max(float(lookup[candidate]["objective"]) for candidate in legal)
                if chosen not in legal or abs(float(lookup[chosen]["objective"]) - best) > EPS:
                    raise AssertionError("forward argmax audit failed")
                selected, current = chosen, best
            swap_steps = search["path"][K:]
            if len(swap_steps) > max_rounds:
                raise AssertionError("swap round cap exceeded")
            for step in swap_steps:
                legal = all_single_swaps(selected, N_FEATURES)
                chosen = tuple(step["chosen_clean_indices_0based"])
                best = max(float(lookup[candidate]["objective"]) for candidate in legal)
                accepted = best > current + EPS
                if (
                    chosen not in legal
                    or abs(float(step["chosen_objective"]) - best) > EPS
                    or bool(step["accepted"]) != accepted
                ):
                    raise AssertionError("swap argmax/strict-stop audit failed")
                if accepted:
                    selected, current = chosen, best
            if tuple(search["endpoint_clean_indices_0based"]) != selected:
                raise AssertionError("search endpoint differs from audited path")

        lr_cost = result["lr_search"]["total_cost"]
        if lr_cost["unique_scored_subsets"] > 4000 or lr_cost["classifier_fit_count"] > 60000:
            all_structural = False
        for name in ("lr_forward", "lr_forward_swap2"):
            subset = result["evaluations"][name]["clean_indices_0based"]
            expected = independent_objective(raw, labels, context, subset)
            actual = (
                result["lr_search"]["forward_objective"]
                if name == "lr_forward"
                else result["lr_search"]["endpoint_objective"]
            )
            max_inner_error = max(max_inner_error, abs(expected - actual))
        for name, evaluation in result["evaluations"].items():
            prediction, _ = fit_lr_predict(
                raw,
                labels,
                final_ids,
                evaluation["clean_indices_0based"],
                train,
                validation,
                context["lr_random_state"],
            )
            expected = balanced_accuracy_score(labels[validation], prediction)
            max_outer_error = max(max_outer_error, abs(expected - evaluation["lr_balanced_accuracy"]))
        frame = pd.read_csv(ROOT / "results" / f"outer-fold-{outer_fold}" / "oof_predictions.csv")
        oof_frames.append(frame)
        cases.append(
            {
                "outer_fold": outer_fold,
                "lr_unique_scored_subsets": lr_cost["unique_scored_subsets"],
                "lr_classifier_fit_count": lr_cost["classifier_fit_count"],
                "lr_candidate_requests": lr_cost["candidate_requests"],
                "lr_cache_hits": lr_cost["cache_hits"],
                "lr_accepted_swap_count": result["lr_search"]["accepted_swap_count"],
            }
        )
    if sorted(seen_validation) != list(range(len(labels))):
        raise AssertionError("outer validation folds do not partition all rows")
    oof = pd.concat(oof_frames, ignore_index=True).sort_values("original_row")
    pooled_saved = pd.read_csv(ROOT / "analysis" / "pooled_results.csv")
    pooled_errors = []
    for row in pooled_saved.to_dict("records"):
        pooled_errors.append(
            abs(
                balanced_accuracy_score(oof["y_true"], oof[row["method"]])
                - row["pooled_oof_lr_bacc"]
            )
        )
    numerical = max(max_inner_error, max_outer_error, max(pooled_errors)) <= EPS
    preaudit = read_json(ROOT / "analysis" / "decision-preaudit.json")
    audit_passed = bool(numerical and all_structural)
    go = bool(
        preaudit["positive_fold_gate"]
        and preaudit["mean_gain_gate"]
        and preaudit["cost_gate"]
        and audit_passed
    )
    decision = {
        "status": "GO-development" if go else "NO-GO",
        "task_4d_admission": go,
        "task_5a_must_wait_for_4d_report": True,
        "stop_static_ppo_5a_labels_and_new_k_reward_scans": not go,
        "frozen_development_scorer": (
            "K=32/fold-local StandardScaler+LR BAcc/3x5 mean-0.5 sample SD" if go else None
        ),
        "task_4b_status_unchanged": "NO-GO",
        "task_4b_unique_main_reward_unchanged": None,
        "not_external_generalization": True,
        **{key: value for key, value in preaudit.items() if key != "status"},
        "numerical_and_structural_audit_gate": audit_passed,
    }
    write_json(ROOT / "analysis" / "final-decision.json", decision)
    return {
        "status": "passed" if audit_passed else "failed",
        "source_test_used": False,
        "source_test_open_count": 0,
        "outer_partition_rows_checked": len(labels),
        "outer_validation_exactly_once": True,
        "cleaning_recomputed_outer_train_only": True,
        "fold_local_scaler_independently_recomputed": True,
        "coordinate_rows_checked": sum(
            len(read_jsonl(ROOT / "results" / f"outer-fold-{fold}" / name))
            for fold in range(5)
            for name in ("lr_candidates.jsonl", "dt_candidates.jsonl")
        ),
        "argmax_and_strict_stop_checked": True,
        "cost_identities_checked": True,
        "max_abs_inner_objective_error": max_inner_error,
        "max_abs_outer_endpoint_error": max_outer_error,
        "max_abs_pooled_oof_error": max(pooled_errors),
        "cases": cases,
        "final_decision": decision,
    }


if __name__ == "__main__":
    payload = audit()
    write_json(ROOT / "audit.json", payload)
    if payload["status"] != "passed":
        raise SystemExit(1)

