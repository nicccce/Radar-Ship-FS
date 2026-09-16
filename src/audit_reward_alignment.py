#!/usr/bin/env python3
"""Independent structural and sampled numerical audit for task 4B."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from run_reward_alignment import (
    AGGREGATIONS,
    EPS,
    KS,
    ROOT,
    SEEDS,
    TRAIN_SHA256,
    all_single_swaps,
    canonical_subset,
    load_source_train,
    read_json,
    read_jsonl,
    sha256,
    verify_frozen_files,
    write_json,
)


def independent_lr_bacc(
    raw: np.ndarray,
    labels: np.ndarray,
    columns: np.ndarray,
    fit: np.ndarray,
    held_out: np.ndarray,
    *,
    seed: int,
) -> float:
    scaler = StandardScaler()
    fit_values = scaler.fit_transform(raw[fit][:, columns])
    held_out_values = scaler.transform(raw[held_out][:, columns])
    classifier = LogisticRegression(
        C=1.0,
        solver="liblinear",
        max_iter=5000,
        class_weight="balanced",
        random_state=seed,
    )
    classifier.fit(fit_values, labels[fit])
    prediction = classifier.predict(held_out_values)
    return float(balanced_accuracy_score(labels[held_out], prediction))


def selected_objective(repeat_means: list[float], aggregation: str) -> float:
    values = np.asarray(repeat_means, dtype=float)
    if aggregation == "single_5fold":
        return float(values[0])
    if aggregation == "repeated_mean":
        return float(np.mean(values))
    if aggregation == "mean_minus_0_5_sd":
        return float(np.mean(values) - 0.5 * np.std(values, ddof=1))
    raise ValueError(aggregation)


def audit() -> dict:
    manifest = read_json(ROOT / "manifest.json")
    if manifest["source_train_sha256"] != TRAIN_SHA256 or manifest["source_test_open_count"] != 0:
        raise AssertionError("data isolation manifest is invalid")
    for path, expected in manifest["code_protocol_hashes"].items():
        if sha256(path) != expected:
            raise AssertionError(f"frozen code/protocol hash mismatch: {path}")
    for marker in (
        ROOT / "candidate-bank-complete.json",
        ROOT / "bank-score-complete.json",
        ROOT / "bank-validation-complete.json",
        ROOT / "aligned-search-complete.json",
    ):
        verify_frozen_files(marker)

    raw, labels = load_source_train()
    selection = read_json(ROOT / "analysis" / "selected-scorer.json")
    aggregation = selection["selected"]["aggregation"]
    if selection["selected"]["signal"] != "lr_bacc" or aggregation not in AGGREGATIONS:
        raise AssertionError("selected scorer is outside the preregistered LR BAcc set")
    multiplier = 1 if aggregation == "single_5fold" else 3
    max_bank_score_error = 0.0
    max_bank_outer_error = 0.0
    max_search_outer_error = 0.0
    audited_bank_candidates = 0
    audited_search_endpoints = 0
    cases = []

    for seed in SEEDS:
        scoring_context = read_json(ROOT / "contexts" / f"seed-{seed}.json")
        repeat_rows = scoring_context["repeat_fold_rows"]
        final_ids = np.asarray(scoring_context["final_feature_ids"], dtype=int)
        for k in KS:
            bank = read_json(ROOT / "candidate_bank" / f"seed-{seed}" / f"k{k}.json")
            candidates = bank["candidates"]
            if set(bank["outer_train_rows"]) & set(bank["outer_validation_rows"]):
                raise AssertionError("outer rows overlap")
            if len(set(bank["outer_train_rows"]) | set(bank["outer_validation_rows"])) != 3897:
                raise AssertionError("outer rows do not cover source-train")
            if bank["cleaning_fit_scope"] != "search_train_only":
                raise AssertionError("cleaning was not fit inside outer-train")
            forward = tuple(candidates[int(bank["forward_candidate_id"])]["clean_indices_0based"])
            complete_actual = {
                tuple(row["clean_indices_0based"])
                for row in candidates
                if "complete_forward_single_swap" in row["strata"]
            }
            if complete_actual != set(all_single_swaps(forward, 65)):
                raise AssertionError("candidate bank does not contain the complete forward neighborhood")
            if sum("global_random" in row["strata"] for row in candidates) != 32:
                raise AssertionError("global random bank stratum has the wrong size")
            for distance in (2, 4, 8):
                label = f"forward_perturbation_distance_{distance}"
                if sum(label in row["strata"] for row in candidates) != 16:
                    raise AssertionError(f"perturbation stratum {distance} has the wrong size")
            for row in candidates:
                clean = canonical_subset(row["clean_indices_0based"], 65)
                expected_original = final_ids[list(clean)].astype(int).tolist()
                if expected_original != row["original_feature_ids_1based"]:
                    raise AssertionError("bank coordinate mapping failed")

            score_rows = read_jsonl(ROOT / "bank_scores" / f"seed-{seed}" / f"k{k}" / "scores.jsonl")
            outer_rows = read_jsonl(
                ROOT / "bank_outer_validation" / f"seed-{seed}" / f"k{k}" / "scores.jsonl"
            )
            if len(score_rows) != len(candidates) or len(outer_rows) != len(candidates):
                raise AssertionError("bank score/validation coverage is incomplete")
            sampled_id = int(bank["forward_candidate_id"])
            sampled = candidates[sampled_id]
            columns = final_ids[np.asarray(sampled["clean_indices_0based"], dtype=int)] - 1
            repeat_means = []
            use_repeats = repeat_rows[:1] if aggregation == "single_5fold" else repeat_rows
            for repeat in use_repeats:
                fold_values = []
                for fold in repeat:
                    fold_values.append(
                        independent_lr_bacc(
                            raw,
                            labels,
                            columns,
                            np.asarray(fold["fit"], dtype=int),
                            np.asarray(fold["held_out"], dtype=int),
                            seed=seed,
                        )
                    )
                repeat_means.append(float(np.mean(fold_values)))
            expected_objective = selected_objective(repeat_means, aggregation)
            actual_objective = float(score_rows[sampled_id]["aggregates"]["lr_bacc"][aggregation])
            max_bank_score_error = max(max_bank_score_error, abs(expected_objective - actual_objective))
            expected_outer = independent_lr_bacc(
                raw,
                labels,
                columns,
                np.asarray(bank["outer_train_rows"], dtype=int),
                np.asarray(bank["outer_validation_rows"], dtype=int),
                seed=seed,
            )
            max_bank_outer_error = max(
                max_bank_outer_error, abs(expected_outer - float(outer_rows[sampled_id]["lr_bacc"]))
            )
            audited_bank_candidates += 1

            result = read_json(ROOT / "aligned_search" / f"seed-{seed}" / f"k{k}" / "result.json")
            search_candidates = read_jsonl(
                ROOT / "aligned_search" / f"seed-{seed}" / f"k{k}" / "candidates.jsonl"
            )
            lookup = {tuple(row["clean_indices_0based"]): row for row in search_candidates}
            cost = result["cost"]
            if cost["classifier_fit_count"] != cost["unique_scored_subsets"] * 5 * multiplier:
                raise AssertionError("aligned search fit count identity failed")
            if cost["candidate_requests"] != cost["unique_scored_subsets"] + cost["cache_hits"]:
                raise AssertionError("aligned search request/cache identity failed")
            if cost["unique_scored_subsets"] > 4000 or cost["classifier_fit_count"] > 60000:
                raise AssertionError("aligned search exceeded its frozen cost ceiling")
            if result["reward_archive_stop_objective"] != f"lr_bacc/{aggregation}":
                raise AssertionError("reward/archive/stop objective is inconsistent")
            previous: tuple[int, ...] = ()
            for step in result["path"][:k]:
                chosen = tuple(step["chosen_clean_indices_0based"])
                legal = [
                    tuple(sorted((*previous, feature))) for feature in range(65) if feature not in previous
                ]
                if chosen not in legal:
                    raise AssertionError("forward chose an illegal candidate")
                best_value = max(float(lookup[candidate]["objective"]) for candidate in legal)
                if abs(float(lookup[chosen]["objective"]) - best_value) > EPS:
                    raise AssertionError("forward did not choose the selected objective maximum")
                previous = chosen
            current = tuple(result["forward_clean_indices_0based"])
            current_value = float(result["forward_objective"])
            archive_values = [float(result["archive"][0]["objective"])]
            for step in result["path"][k:]:
                legal = all_single_swaps(current, 65)
                chosen = tuple(step["chosen_clean_indices_0based"])
                if chosen not in legal:
                    raise AssertionError("single-swap chose an illegal candidate")
                best_value = max(float(lookup[candidate]["objective"]) for candidate in legal)
                if abs(float(step["chosen_objective"]) - best_value) > EPS:
                    raise AssertionError("single-swap did not choose the objective maximum")
                expected_accept = best_value > current_value + EPS
                if bool(step["accepted"]) != expected_accept:
                    raise AssertionError("single-swap stop rule is inconsistent")
                if expected_accept:
                    current = chosen
                    current_value = best_value
                    archive_values.append(current_value)
            if any(right <= left + EPS for left, right in zip(archive_values, archive_values[1:])):
                raise AssertionError("archive contains a non-improving accepted candidate")
            endpoint = tuple(result["endpoint_clean_indices_0based"])
            endpoint_columns = final_ids[np.asarray(endpoint, dtype=int)] - 1
            expected_endpoint_outer = independent_lr_bacc(
                raw,
                labels,
                endpoint_columns,
                np.asarray(bank["outer_train_rows"], dtype=int),
                np.asarray(bank["outer_validation_rows"], dtype=int),
                seed=seed,
            )
            max_search_outer_error = max(
                max_search_outer_error,
                abs(expected_endpoint_outer - float(result["outer_endpoint"]["lr_bacc"])),
            )
            audited_search_endpoints += 1
            cases.append(
                {
                    "seed": seed,
                    "k": k,
                    "candidate_count": len(candidates),
                    "complete_single_swap_count": len(complete_actual),
                    "search_unique_subsets": cost["unique_scored_subsets"],
                    "search_classifier_fits": cost["classifier_fit_count"],
                }
            )

    preaudit = read_json(ROOT / "analysis" / "decision-preaudit.json")
    numerical_passed = max(max_bank_score_error, max_bank_outer_error, max_search_outer_error) <= EPS
    status = "passed" if numerical_passed else "failed"
    freeze_reward = bool(preaudit["both_k_pre_audit_passed"] and numerical_passed)
    final_decision = {
        "status": "go" if freeze_reward else "no-go",
        "unique_main_reward": preaudit["unique_main_reward_if_passed"] if freeze_reward else None,
        "selected_diagnostic_scorer": preaudit["selected_diagnostic_scorer"],
        "reason": (
            "all preregistered bank, search-gain, cost, and audit gates passed"
            if freeze_reward
            else "one or more preregistered bank, search-gain, cost, or audit gates failed"
        ),
        "task_4c_reward_admission": freeze_reward,
    }
    write_json(ROOT / "analysis" / "final-decision.json", final_decision)
    return {
        "status": status,
        "source_test_used": False,
        "source_train_sha256": TRAIN_SHA256,
        "selected_signal_is_lr_bacc": True,
        "fold_local_scaler_recomputed": True,
        "candidate_banks_structurally_complete": True,
        "reward_archive_stop_aligned": True,
        "audited_bank_candidates": audited_bank_candidates,
        "audited_search_endpoints": audited_search_endpoints,
        "max_abs_bank_score_error": max_bank_score_error,
        "max_abs_bank_outer_error": max_bank_outer_error,
        "max_abs_search_outer_error": max_search_outer_error,
        "cases": cases,
        "final_decision": final_decision,
    }


if __name__ == "__main__":
    payload = audit()
    write_json(ROOT / "audit.json", payload)
    if payload["status"] != "passed":
        raise SystemExit(1)
