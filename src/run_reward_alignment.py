#!/usr/bin/env python3
"""Task 4B: frozen, source-train-only reward reliability and LR-aligned search."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed, parallel_config
from scipy.stats import spearmanr
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path("experiments/reward_alignment_v1")
TASK3 = Path("experiments/search_diagnosis_v1")
TRAIN = Path("../dataset/sim_ship_cr_v16n_2x_noise.train.svm")
CONFIG = Path("configs/v16n/reward_alignment_v1.toml")
PROTOCOL = Path("documents/research-plan/reward-alignment-protocol.md")
TRAIN_SHA256 = "2191fdc297aa49ce6a700df956ec44f68dc13e4c737937b31167543a203d22bb"
SEEDS = (42, 43, 44, 45, 46)
KS = (16, 32)
SIGNALS = ("dt_accuracy", "dt_bacc", "lr_accuracy", "lr_bacc")
AGGREGATIONS = ("single_5fold", "repeated_mean", "mean_minus_0_5_sd")
TOP_B = 32
EPS = 1e-12
PENALTY_WEIGHT = 0.5


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(target, index=False)


def canonical_subset(subset: Iterable[int], n_features: int) -> tuple[int, ...]:
    values = tuple(sorted(int(value) for value in subset))
    if not values or len(values) != len(set(values)):
        raise ValueError("subset must be non-empty and contain unique features")
    if values[0] < 0 or values[-1] >= n_features:
        raise ValueError("subset contains a feature outside the cleaned feature space")
    return values


def all_single_swaps(subset: Sequence[int], n_features: int) -> list[tuple[int, ...]]:
    selected = canonical_subset(subset, n_features)
    chosen = set(selected)
    return [
        tuple(sorted((chosen - {removed}) | {added}))
        for removed in selected
        for added in range(n_features)
        if added not in chosen
    ]


def aggregate_repeat_scores(repeat_means: Sequence[float]) -> dict[str, float]:
    values = np.asarray(repeat_means, dtype=float)
    if values.shape != (3,):
        raise ValueError("exactly three repeat means are required")
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    return {
        "single_5fold": float(values[0]),
        "repeated_mean": mean,
        "mean_minus_0_5_sd": mean - PENALTY_WEIGHT * sd,
    }


def _add_bank_candidate(
    store: dict[tuple[int, ...], dict[str, Any]],
    subset: Iterable[int],
    *,
    n_features: int,
    stratum: str,
    swap_distance: int | None,
) -> None:
    key = canonical_subset(subset, n_features)
    item = store.setdefault(
        key,
        {"clean_indices_0based": list(key), "strata": [], "swap_distances_from_forward": []},
    )
    if stratum not in item["strata"]:
        item["strata"].append(stratum)
    if swap_distance is not None and swap_distance not in item["swap_distances_from_forward"]:
        item["swap_distances_from_forward"].append(int(swap_distance))


def build_candidate_bank(
    forward: Sequence[int],
    mi_topk: Sequence[int],
    *,
    n_features: int,
    k: int,
    seed: int,
    global_random_count: int = 32,
    perturbation_count: int = 16,
) -> list[dict[str, Any]]:
    """Build the preregistered exact-K bank without any validation information."""

    forward_key = canonical_subset(forward, n_features)
    mi_key = canonical_subset(mi_topk, n_features)
    if len(forward_key) != k or len(mi_key) != k:
        raise ValueError("bank anchors must match K")
    store: dict[tuple[int, ...], dict[str, Any]] = {}
    _add_bank_candidate(
        store, forward_key, n_features=n_features, stratum="forward_checkpoint", swap_distance=0
    )
    _add_bank_candidate(store, mi_key, n_features=n_features, stratum="mi_topk", swap_distance=None)
    for candidate in all_single_swaps(forward_key, n_features):
        _add_bank_candidate(
            store,
            candidate,
            n_features=n_features,
            stratum="complete_forward_single_swap",
            swap_distance=1,
        )

    rng = np.random.default_rng(400000 + 100 * seed + k)
    global_seen: set[tuple[int, ...]] = set()
    while len(global_seen) < global_random_count:
        candidate = tuple(sorted(rng.choice(n_features, size=k, replace=False).astype(int).tolist()))
        if candidate in global_seen:
            continue
        global_seen.add(candidate)
        _add_bank_candidate(
            store, candidate, n_features=n_features, stratum="global_random", swap_distance=None
        )

    selected = set(forward_key)
    available = sorted(set(range(n_features)) - selected)
    for distance in (2, 4, 8):
        distance_seen: set[tuple[int, ...]] = set()
        while len(distance_seen) < perturbation_count:
            removed = set(rng.choice(forward_key, size=distance, replace=False).astype(int).tolist())
            added = set(rng.choice(available, size=distance, replace=False).astype(int).tolist())
            candidate = tuple(sorted((selected - removed) | added))
            if candidate in distance_seen:
                continue
            distance_seen.add(candidate)
            _add_bank_candidate(
                store,
                candidate,
                n_features=n_features,
                stratum=f"forward_perturbation_distance_{distance}",
                swap_distance=distance,
            )

    rows = list(store.values())
    for candidate_id, row in enumerate(rows):
        row["candidate_id"] = candidate_id
    return rows


def load_source_train() -> tuple[np.ndarray, np.ndarray]:
    if sha256(TRAIN) != TRAIN_SHA256:
        raise RuntimeError("source-train hash differs from the frozen protocol")
    sparse, labels = load_svmlight_file(TRAIN, n_features=75)
    return sparse.toarray().astype(np.float32), labels.astype(np.int64)


def _task3_case(seed: int) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    case = TASK3 / "search" / f"nested_dev-seed-{seed}"
    context_path = case / "context.json"
    frozen = read_json(TASK3 / "search-complete.json")["files"]
    if frozen.get(str(context_path)) != sha256(context_path):
        raise RuntimeError(f"task 3 context hash mismatch for seed {seed}")
    return (
        read_json(context_path),
        read_json(case / "complete.json"),
        read_jsonl(case / "candidates.jsonl"),
    )


def prepare_manifest() -> None:
    sources = [
        Path(__file__),
        CONFIG,
        PROTOCOL,
        Path("src/data/loader.py"),
        Path("src/audit_reward_alignment.py"),
        Path("src/summarize_reward_alignment.py"),
        Path("tests/test_reward_alignment.py"),
    ]
    hashes = {str(path): sha256(path) for path in sources}
    if (ROOT / "manifest.json").exists():
        current = read_json(ROOT / "manifest.json")
        if current["code_protocol_hashes"] != hashes:
            raise RuntimeError("frozen task 4B source/protocol changed after manifest creation")
        return
    ROOT.mkdir(parents=True, exist_ok=False)
    write_json(
        ROOT / "manifest.json",
        {
            "version": "reward-alignment-v1",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "code_protocol_hashes": hashes,
            "source_train_sha256": sha256(TRAIN),
            "source_test_open_count": 0,
            "rl_training_count": 0,
            "seeds": list(SEEDS),
            "ks": list(KS),
            "signals": list(SIGNALS),
            "aggregations": list(AGGREGATIONS),
            "python": sys.version,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "hardware": platform.platform(),
            "cpu_count": os.cpu_count(),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "git_status": subprocess.check_output(["git", "status", "--short"], text=True),
        },
    )


def freeze_banks() -> None:
    if (ROOT / "candidate-bank-complete.json").exists():
        return
    bank_root = ROOT / "candidate_bank"
    if bank_root.exists():
        raise RuntimeError("partial candidate bank refuses overwrite")
    for seed in SEEDS:
        context, complete, candidates = _task3_case(seed)
        if context["outer_split_random_state"] != 10000 + seed:
            raise AssertionError("outer split does not match the frozen task 4B rule")
        fit_rows = set(context["fit_original_rows"])
        validation_rows = set(context["validation_original_rows"])
        if fit_rows & validation_rows or len(fit_rows | validation_rows) != 3897:
            raise AssertionError("outer rows are not disjoint and exhaustive")
        final_ids = np.asarray(context["final_feature_ids"], dtype=int)
        mi_order = np.argsort(np.asarray(context["mi_rank"], dtype=int), kind="stable")
        for k in KS:
            forward_id = int(complete["starts"][str(k)])
            forward = tuple(candidates[forward_id]["clean_indices_0based"])
            bank = build_candidate_bank(
                forward,
                tuple(mi_order[:k].astype(int).tolist()),
                n_features=len(final_ids),
                k=k,
                seed=seed,
            )
            for row in bank:
                clean = tuple(row["clean_indices_0based"])
                original = final_ids[list(clean)].astype(int).tolist()
                if tuple(np.flatnonzero(np.isin(final_ids, original)).tolist()) != clean:
                    raise AssertionError("clean/original coordinate cross-check failed")
                row["original_feature_ids_1based"] = original
                row["coordinate_crosscheck_passed"] = True
            payload = {
                "version": "reward-alignment-v1",
                "seed": seed,
                "k": k,
                "n_features": len(final_ids),
                "outer_train_rows": context["development_original_rows"],
                "outer_validation_rows": context["validation_original_rows"],
                "cleaning": context["cleaning"],
                "cleaning_fit_scope": context["cleaning_fit_scope"],
                "final_feature_ids": final_ids.tolist(),
                "forward_candidate_id": next(
                    row["candidate_id"] for row in bank if "forward_checkpoint" in row["strata"]
                ),
                "mi_candidate_id": next(row["candidate_id"] for row in bank if "mi_topk" in row["strata"]),
                "candidate_count": len(bank),
                "candidates": bank,
                "generation": {
                    "random_state": 400000 + 100 * seed + k,
                    "global_random_count": 32,
                    "perturbation_distances": [2, 4, 8],
                    "perturbation_count_per_distance": 16,
                    "complete_single_swap_count": k * (len(final_ids) - k),
                },
            }
            write_json(bank_root / f"seed-{seed}" / f"k{k}.json", payload)
    files = sorted(path for path in bank_root.rglob("*") if path.is_file())
    write_json(
        ROOT / "candidate-bank-complete.json",
        {
            "all_candidates_frozen_before_scoring_and_outer_validation": True,
            "files": {str(path): sha256(path) for path in files},
        },
    )


def verify_frozen_files(marker: str | Path) -> None:
    payload = read_json(marker)
    for path, expected in payload["files"].items():
        if sha256(path) != expected:
            raise RuntimeError(f"frozen artifact changed: {path}")


def repeated_fold_rows(context: dict[str, Any], labels: np.ndarray) -> list[list[dict[str, list[int]]]]:
    repeats = [context["fold_original_rows"]]
    outer_rows = np.asarray(context["development_original_rows"], dtype=int)
    for random_state in (420000 + int(context["seed"]), 430000 + int(context["seed"])):
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        repeats.append(
            [
                {
                    "fit": outer_rows[fit].astype(int).tolist(),
                    "held_out": outer_rows[held_out].astype(int).tolist(),
                }
                for fit, held_out in splitter.split(outer_rows, labels[outer_rows])
            ]
        )
    return repeats


def fit_fold_local_lr(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_held_out: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """Fit the scaler on this scoring fold only and return held-out predictions."""

    scaler = StandardScaler().fit(X_fit)
    transformed_fit = scaler.transform(X_fit)
    transformed_held_out = scaler.transform(X_held_out)
    classifier = LogisticRegression(
        C=1.0,
        solver="liblinear",
        max_iter=5000,
        class_weight="balanced",
        random_state=seed,
    )
    classifier.fit(transformed_fit, y_fit)
    return classifier.predict(transformed_held_out)


def score_candidate_all_signals(
    raw: np.ndarray,
    labels: np.ndarray,
    final_ids: Sequence[int],
    subset: Sequence[int],
    repeats: Sequence[Sequence[dict[str, Sequence[int]]]],
    *,
    seed: int,
    tree_random_state: int,
) -> dict[str, Any]:
    columns = np.asarray(final_ids, dtype=int)[np.asarray(subset, dtype=int)] - 1
    fold_scores: dict[str, list[list[float]]] = {signal: [] for signal in SIGNALS}
    started = time.perf_counter()
    for repeat in repeats:
        current = {signal: [] for signal in SIGNALS}
        for fold in repeat:
            fit = np.asarray(fold["fit"], dtype=int)
            held_out = np.asarray(fold["held_out"], dtype=int)
            tree = DecisionTreeClassifier(random_state=tree_random_state)
            tree.fit(raw[fit][:, columns], labels[fit])
            tree_prediction = tree.predict(raw[held_out][:, columns])
            lr_prediction = fit_fold_local_lr(
                raw[fit][:, columns],
                labels[fit],
                raw[held_out][:, columns],
                seed=seed,
            )
            current["dt_accuracy"].append(float(accuracy_score(labels[held_out], tree_prediction)))
            current["dt_bacc"].append(float(balanced_accuracy_score(labels[held_out], tree_prediction)))
            current["lr_accuracy"].append(float(accuracy_score(labels[held_out], lr_prediction)))
            current["lr_bacc"].append(float(balanced_accuracy_score(labels[held_out], lr_prediction)))
        for signal in SIGNALS:
            fold_scores[signal].append(current[signal])
    repeat_means = {signal: [float(np.mean(scores)) for scores in fold_scores[signal]] for signal in SIGNALS}
    return {
        "fold_scores": fold_scores,
        "repeat_means": repeat_means,
        "aggregates": {signal: aggregate_repeat_scores(repeat_means[signal]) for signal in SIGNALS},
        "classifier_fit_count": 30,
        "dt_fit_count": 15,
        "lr_fit_count": 15,
        "fit_seconds": time.perf_counter() - started,
        "standard_scaler_fit_scope": "each_scoring_training_fold",
    }


def _scoring_context(
    seed: int, labels: np.ndarray
) -> tuple[dict[str, Any], list[list[dict[str, list[int]]]]]:
    context, _, _ = _task3_case(seed)
    context = {**context, "seed": seed}
    return context, repeated_fold_rows(context, labels)


def score_banks(jobs: int) -> None:
    verify_frozen_files(ROOT / "candidate-bank-complete.json")
    raw, labels = load_source_train()
    score_root = ROOT / "bank_scores"
    for seed in SEEDS:
        context, repeats = _scoring_context(seed, labels)
        write_json(
            ROOT / "contexts" / f"seed-{seed}.json",
            {
                "seed": seed,
                "outer_train_rows": context["development_original_rows"],
                "outer_validation_rows": context["validation_original_rows"],
                "final_feature_ids": context["final_feature_ids"],
                "cleaning": context["cleaning"],
                "cleaning_fit_scope": context["cleaning_fit_scope"],
                "cv_tree_random_state": context["cv_tree_random_state"],
                "repeat_random_states": [context["cv_tree_random_state"], 420000 + seed, 430000 + seed],
                "repeat_fold_rows": repeats,
            },
        )
        for k in KS:
            out = score_root / f"seed-{seed}" / f"k{k}"
            if (out / "complete.json").exists():
                continue
            if out.exists():
                raise RuntimeError(f"partial bank scoring refuses overwrite: {out}")
            bank = read_json(ROOT / "candidate_bank" / f"seed-{seed}" / f"k{k}.json")
            final_ids = bank["final_feature_ids"]
            started = time.perf_counter()
            results = Parallel(n_jobs=jobs, prefer="processes")(
                delayed(score_candidate_all_signals)(
                    raw,
                    labels,
                    final_ids,
                    candidate["clean_indices_0based"],
                    repeats,
                    seed=seed,
                    tree_random_state=int(context["cv_tree_random_state"]),
                )
                for candidate in bank["candidates"]
            )
            rows = [
                {"candidate_id": candidate["candidate_id"], **result}
                for candidate, result in zip(bank["candidates"], results)
            ]
            write_jsonl(out / "scores.jsonl", rows)
            write_json(
                out / "complete.json",
                {
                    "seed": seed,
                    "k": k,
                    "candidate_count": len(rows),
                    "classifier_fit_count": sum(row["classifier_fit_count"] for row in rows),
                    "dt_fit_count": sum(row["dt_fit_count"] for row in rows),
                    "lr_fit_count": sum(row["lr_fit_count"] for row in rows),
                    "sum_fit_seconds": sum(row["fit_seconds"] for row in rows),
                    "wall_seconds": time.perf_counter() - started,
                },
            )
            print(f"bank score seed={seed} K={k}: {len(rows)} candidates", flush=True)
    files = sorted(path for path in score_root.rglob("*") if path.is_file())
    write_json(ROOT / "bank-score-complete.json", {"files": {str(path): sha256(path) for path in files}})


def evaluate_outer_subset(
    raw: np.ndarray,
    labels: np.ndarray,
    final_ids: Sequence[int],
    subset: Sequence[int],
    outer_train: Sequence[int],
    outer_validation: Sequence[int],
    *,
    seed: int,
) -> dict[str, float | int]:
    columns = np.asarray(final_ids, dtype=int)[np.asarray(subset, dtype=int)] - 1
    fit = np.asarray(outer_train, dtype=int)
    held_out = np.asarray(outer_validation, dtype=int)
    started = time.perf_counter()
    lr_prediction = fit_fold_local_lr(raw[fit][:, columns], labels[fit], raw[held_out][:, columns], seed=seed)
    tree = DecisionTreeClassifier(random_state=seed)
    tree.fit(raw[fit][:, columns], labels[fit])
    tree_prediction = tree.predict(raw[held_out][:, columns])
    return {
        "lr_bacc": float(balanced_accuracy_score(labels[held_out], lr_prediction)),
        "lr_accuracy": float(accuracy_score(labels[held_out], lr_prediction)),
        "dt_bacc": float(balanced_accuracy_score(labels[held_out], tree_prediction)),
        "dt_accuracy": float(accuracy_score(labels[held_out], tree_prediction)),
        "classifier_fit_count": 2,
        "fit_seconds": time.perf_counter() - started,
    }


def validate_banks(jobs: int) -> None:
    verify_frozen_files(ROOT / "candidate-bank-complete.json")
    raw, labels = load_source_train()
    validation_root = ROOT / "bank_outer_validation"
    for seed in SEEDS:
        for k in KS:
            out = validation_root / f"seed-{seed}" / f"k{k}"
            if (out / "complete.json").exists():
                continue
            if out.exists():
                raise RuntimeError(f"partial bank validation refuses overwrite: {out}")
            bank = read_json(ROOT / "candidate_bank" / f"seed-{seed}" / f"k{k}.json")
            started = time.perf_counter()
            results = Parallel(n_jobs=jobs, prefer="processes")(
                delayed(evaluate_outer_subset)(
                    raw,
                    labels,
                    bank["final_feature_ids"],
                    candidate["clean_indices_0based"],
                    bank["outer_train_rows"],
                    bank["outer_validation_rows"],
                    seed=seed,
                )
                for candidate in bank["candidates"]
            )
            rows = [
                {"candidate_id": candidate["candidate_id"], **result}
                for candidate, result in zip(bank["candidates"], results)
            ]
            write_jsonl(out / "scores.jsonl", rows)
            write_json(
                out / "complete.json",
                {
                    "seed": seed,
                    "k": k,
                    "candidate_count": len(rows),
                    "classifier_fit_count": 2 * len(rows),
                    "sum_fit_seconds": sum(float(row["fit_seconds"]) for row in rows),
                    "wall_seconds": time.perf_counter() - started,
                    "primary_metric": "lr_bacc",
                },
            )
            print(f"outer validate seed={seed} K={k}: {len(rows)} candidates", flush=True)
    files = sorted(path for path in validation_root.rglob("*") if path.is_file())
    write_json(ROOT / "bank-validation-complete.json", {"files": {str(path): sha256(path) for path in files}})


def sign(value: float) -> int:
    return 1 if value > EPS else -1 if value < -EPS else 0


def safe_spearman(left: Sequence[float], right: Sequence[float]) -> float:
    value = float(spearmanr(left, right).statistic)
    return value if np.isfinite(value) else 0.0


def _top_b_metrics(scores: np.ndarray, validation: np.ndarray, b: int) -> tuple[int, float]:
    n = len(scores)
    count = min(b, n)
    score_top = set(np.argsort(-scores, kind="stable")[:count].astype(int).tolist())
    validation_top = set(np.argsort(-validation, kind="stable")[:count].astype(int).tolist())
    overlap = len(score_top & validation_top)
    enrichment = overlap * n / float(count * count)
    return overlap, enrichment


def choose_lr_bacc_protocol(summary_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    candidates = []
    priority = {"repeated_mean": 3, "mean_minus_0_5_sd": 2, "single_5fold": 1}
    multiplier = {"single_5fold": 1, "repeated_mean": 3, "mean_minus_0_5_sd": 3}
    for aggregation in AGGREGATIONS:
        rows = [
            row for row in summary_rows if row["signal"] == "lr_bacc" and row["aggregation"] == aggregation
        ]
        by_k = {k: [row for row in rows if int(row["k"]) == k] for k in KS}
        enrich = {k: float(np.mean([row["top_b_enrichment"] for row in by_k[k]])) for k in KS}
        rho = {k: float(np.mean([row["spearman"] for row in by_k[k]])) for k in KS}
        stability = {k: float(np.mean([row["repeat_rank_stability"] for row in by_k[k]])) for k in KS}
        key = (
            min(enrich.values()),
            float(np.mean(list(enrich.values()))),
            min(rho.values()),
            float(np.mean(list(rho.values()))),
            float(np.mean(list(stability.values()))),
            -multiplier[aggregation],
            priority[aggregation],
        )
        gates = {
            str(k): {
                "mean_top_b_enrichment": enrich[k],
                "partitions_enrichment_gt_1": int(sum(row["top_b_enrichment"] > 1.0 for row in by_k[k])),
                "mean_repeat_rank_stability": stability[k],
                "passed": enrich[k] >= 2.0
                and sum(row["top_b_enrichment"] > 1.0 for row in by_k[k]) >= 4
                and stability[k] >= 0.5,
            }
            for k in KS
        }
        candidates.append(
            {
                "signal": "lr_bacc",
                "aggregation": aggregation,
                "fit_multiplier": multiplier[aggregation],
                "selection_key": list(key),
                "bank_gate_by_k": gates,
                "bank_gate_passed": all(gate["passed"] for gate in gates.values()),
            }
        )
    selected = max(candidates, key=lambda row: tuple(row["selection_key"]))
    return {"selected": selected, "eligible_protocols": candidates}


def analyze_banks() -> None:
    verify_frozen_files(ROOT / "bank-score-complete.json")
    verify_frozen_files(ROOT / "bank-validation-complete.json")
    reliability_rows: list[dict[str, Any]] = []
    gain_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for k in KS:
            bank = read_json(ROOT / "candidate_bank" / f"seed-{seed}" / f"k{k}.json")
            scored = read_jsonl(ROOT / "bank_scores" / f"seed-{seed}" / f"k{k}" / "scores.jsonl")
            outer_rows = read_jsonl(
                ROOT / "bank_outer_validation" / f"seed-{seed}" / f"k{k}" / "scores.jsonl"
            )
            scores = {int(row["candidate_id"]): row for row in scored}
            outer = {int(row["candidate_id"]): row for row in outer_rows}
            ids = [int(candidate["candidate_id"]) for candidate in bank["candidates"]]
            anchor = int(bank["forward_candidate_id"])
            for signal in SIGNALS:
                repeat_ranks = [
                    [float(scores[candidate_id]["repeat_means"][signal][repeat]) for candidate_id in ids]
                    for repeat in range(3)
                ]
                rank_stability = float(
                    np.mean(
                        [
                            safe_spearman(repeat_ranks[0], repeat_ranks[1]),
                            safe_spearman(repeat_ranks[0], repeat_ranks[2]),
                            safe_spearman(repeat_ranks[1], repeat_ranks[2]),
                        ]
                    )
                )
                anchor_repeats = np.asarray(scores[anchor]["repeat_means"][signal], dtype=float)
                for candidate_id in ids:
                    gains = (
                        np.asarray(scores[candidate_id]["repeat_means"][signal], dtype=float) - anchor_repeats
                    )
                    signs = [sign(float(value)) for value in gains]
                    gain_rows.append(
                        {
                            "seed": seed,
                            "k": k,
                            "candidate_id": candidate_id,
                            "signal": signal,
                            "gain_mean": float(np.mean(gains)),
                            "gain_sample_variance": float(np.var(gains, ddof=1)),
                            "positive_repeats": signs.count(1),
                            "zero_repeats": signs.count(0),
                            "negative_repeats": signs.count(-1),
                            "sign_stability": max(signs.count(1), signs.count(0), signs.count(-1)) / 3.0,
                        }
                    )
                validation = np.asarray([outer[candidate_id]["lr_bacc"] for candidate_id in ids])
                validation_gain = validation - float(outer[anchor]["lr_bacc"])
                for aggregation in AGGREGATIONS:
                    scalar = np.asarray(
                        [scores[candidate_id]["aggregates"][signal][aggregation] for candidate_id in ids]
                    )
                    scalar_gain = scalar - float(scores[anchor]["aggregates"][signal][aggregation])
                    overlap, enrichment = _top_b_metrics(scalar, validation, TOP_B)
                    reliability_rows.append(
                        {
                            "seed": seed,
                            "k": k,
                            "signal": signal,
                            "aggregation": aggregation,
                            "n_candidates": len(ids),
                            "spearman": safe_spearman(scalar, validation),
                            "gain_sign_agreement": float(
                                np.mean(
                                    [
                                        sign(float(left)) == sign(float(right))
                                        for left, right in zip(scalar_gain, validation_gain)
                                    ]
                                )
                            ),
                            "top_b": min(TOP_B, len(ids)),
                            "top_b_overlap": overlap,
                            "top_b_enrichment": enrichment,
                            "repeat_rank_stability": rank_stability,
                            "fit_multiplier": 1 if aggregation == "single_5fold" else 3,
                        }
                    )
    write_csv(ROOT / "analysis" / "candidate_gain_stats.csv", gain_rows)
    write_csv(ROOT / "analysis" / "reliability.csv", reliability_rows)
    summary_rows: list[dict[str, Any]] = []
    for (signal, aggregation, k), frame in pd.DataFrame(reliability_rows).groupby(
        ["signal", "aggregation", "k"], sort=True
    ):
        summary_rows.append(
            {
                "signal": signal,
                "aggregation": aggregation,
                "k": int(k),
                "outer_partitions": len(frame),
                "spearman_mean": float(frame["spearman"].mean()),
                "spearman_sd": float(frame["spearman"].std(ddof=1)),
                "gain_sign_agreement_mean": float(frame["gain_sign_agreement"].mean()),
                "top_b_overlap_sum": int(frame["top_b_overlap"].sum()),
                "top_b_enrichment_mean": float(frame["top_b_enrichment"].mean()),
                "partitions_enrichment_gt_1": int((frame["top_b_enrichment"] > 1.0).sum()),
                "repeat_rank_stability_mean": float(frame["repeat_rank_stability"].mean()),
                "fit_multiplier": int(frame["fit_multiplier"].iloc[0]),
            }
        )
    write_csv(ROOT / "analysis" / "reliability_summary.csv", summary_rows)
    selection = choose_lr_bacc_protocol(reliability_rows)
    selection["selection_frozen_before_new_search"] = True
    selection["selection_rule"] = (
        "lexicographic min-K enrichment, pooled enrichment, min-K Spearman, pooled Spearman, "
        "rank stability, lower fit multiplier, fixed tie priority"
    )
    write_json(ROOT / "analysis" / "selected-scorer.json", selection)


def score_candidate_selected(
    raw: np.ndarray,
    labels: np.ndarray,
    final_ids: Sequence[int],
    subset: Sequence[int],
    repeats: Sequence[Sequence[dict[str, Sequence[int]]]],
    *,
    seed: int,
    aggregation: str,
) -> dict[str, Any]:
    use_repeats = repeats[:1] if aggregation == "single_5fold" else repeats
    columns = np.asarray(final_ids, dtype=int)[np.asarray(subset, dtype=int)] - 1
    fold_scores: list[list[float]] = []
    started = time.perf_counter()
    for repeat in use_repeats:
        current: list[float] = []
        for fold in repeat:
            fit = np.asarray(fold["fit"], dtype=int)
            held_out = np.asarray(fold["held_out"], dtype=int)
            prediction = fit_fold_local_lr(
                raw[fit][:, columns], labels[fit], raw[held_out][:, columns], seed=seed
            )
            current.append(float(balanced_accuracy_score(labels[held_out], prediction)))
        fold_scores.append(current)
    repeat_means = [float(np.mean(values)) for values in fold_scores]
    if aggregation == "single_5fold":
        objective = repeat_means[0]
    elif aggregation == "repeated_mean":
        objective = float(np.mean(repeat_means))
    elif aggregation == "mean_minus_0_5_sd":
        objective = float(np.mean(repeat_means) - PENALTY_WEIGHT * np.std(repeat_means, ddof=1))
    else:
        raise ValueError(f"unknown aggregation: {aggregation}")
    return {
        "objective": objective,
        "repeat_means": repeat_means,
        "fold_scores": fold_scores,
        "classifier_fit_count": 5 * len(use_repeats),
        "fit_seconds": time.perf_counter() - started,
    }


class SelectedScorer:
    def __init__(
        self,
        raw: np.ndarray,
        labels: np.ndarray,
        final_ids: Sequence[int],
        repeats: Sequence[Sequence[dict[str, Sequence[int]]]],
        *,
        seed: int,
        aggregation: str,
        jobs: int,
    ) -> None:
        self.raw = raw
        self.labels = labels
        self.final_ids = tuple(int(value) for value in final_ids)
        self.repeats = repeats
        self.seed = seed
        self.aggregation = aggregation
        self.jobs = jobs
        self.cache: dict[tuple[int, ...], dict[str, Any]] = {}
        self.requests = 0
        self.cache_hits = 0
        self.fit_count = 0
        self.fit_seconds = 0.0

    def score_many(self, subsets: Sequence[Sequence[int]]) -> list[float]:
        keys = [canonical_subset(subset, len(self.final_ids)) for subset in subsets]
        self.requests += len(keys)
        missing: list[tuple[int, ...]] = []
        seen: set[tuple[int, ...]] = set()
        for key in keys:
            if key in self.cache or key in seen:
                self.cache_hits += 1
            else:
                seen.add(key)
                missing.append(key)
        results = Parallel(n_jobs=self.jobs, prefer="processes")(
            delayed(score_candidate_selected)(
                self.raw,
                self.labels,
                self.final_ids,
                subset,
                self.repeats,
                seed=self.seed,
                aggregation=self.aggregation,
            )
            for subset in missing
        )
        for subset, result in zip(missing, results):
            self.cache[subset] = result
            self.fit_count += int(result["classifier_fit_count"])
            self.fit_seconds += float(result["fit_seconds"])
        return [float(self.cache[key]["objective"]) for key in keys]

    def score(self, subset: Sequence[int]) -> float:
        return self.score_many([subset])[0]

    def costs(self) -> dict[str, Any]:
        return {
            "candidate_requests": self.requests,
            "unique_scored_subsets": len(self.cache),
            "cache_hits": self.cache_hits,
            "classifier_fit_count": self.fit_count,
            "fit_seconds": self.fit_seconds,
        }


def choose_best(candidates: Sequence[tuple[int, ...]], scores: Sequence[float]) -> tuple[int, ...]:
    return candidates[int(np.argmax(np.asarray(scores, dtype=float)))]


def aligned_search_case(seed: int, k: int, aggregation: str, jobs: int) -> None:
    out = ROOT / "aligned_search" / f"seed-{seed}" / f"k{k}"
    if (out / "complete.json").exists():
        return
    if out.exists():
        raise RuntimeError(f"partial aligned search refuses overwrite: {out}")
    raw, labels = load_source_train()
    context, repeats = _scoring_context(seed, labels)
    final_ids = tuple(int(value) for value in context["final_feature_ids"])
    scorer = SelectedScorer(raw, labels, final_ids, repeats, seed=seed, aggregation=aggregation, jobs=jobs)
    started = time.perf_counter()
    selected: tuple[int, ...] = ()
    current_score = float("-inf")
    path: list[dict[str, Any]] = []
    for step in range(1, k + 1):
        candidates = [
            tuple(sorted((*selected, feature)))
            for feature in range(len(final_ids))
            if feature not in selected
        ]
        scores = scorer.score_many(candidates)
        chosen = choose_best(candidates, scores)
        current_score = float(scores[candidates.index(chosen)])
        selected = chosen
        path.append(
            {
                "phase": "forward",
                "step": step,
                "candidate_count": len(candidates),
                "chosen_clean_indices_0based": list(chosen),
                "chosen_objective": current_score,
                "accepted": True,
                "cumulative_cost": scorer.costs(),
            }
        )
    forward = selected
    forward_score = current_score
    archive = [
        {
            "source": "forward_exact_k",
            "clean_indices_0based": list(forward),
            "objective": forward_score,
        }
    ]
    termination = "two_swap_round_budget_exhausted"
    accepted_swaps = 0
    for round_index in range(1, 3):
        candidates = all_single_swaps(selected, len(final_ids))
        scores = scorer.score_many(candidates)
        chosen = choose_best(candidates, scores)
        chosen_score = float(scores[candidates.index(chosen)])
        accepted = chosen_score > current_score + EPS
        path.append(
            {
                "phase": "single_swap",
                "round": round_index,
                "candidate_count": len(candidates),
                "chosen_clean_indices_0based": list(chosen),
                "chosen_objective": chosen_score,
                "objective_gain": chosen_score - current_score,
                "accepted": accepted,
                "cumulative_cost": scorer.costs(),
            }
        )
        if not accepted:
            termination = "no_strict_single_swap_improvement"
            break
        selected = chosen
        current_score = chosen_score
        accepted_swaps += 1
        archive.append(
            {
                "source": "accepted_best_single_swap",
                "round": round_index,
                "clean_indices_0based": list(selected),
                "objective": current_score,
            }
        )

    forward_outer = evaluate_outer_subset(
        raw,
        labels,
        final_ids,
        forward,
        context["development_original_rows"],
        context["validation_original_rows"],
        seed=seed,
    )
    endpoint_outer = evaluate_outer_subset(
        raw,
        labels,
        final_ids,
        selected,
        context["development_original_rows"],
        context["validation_original_rows"],
        seed=seed,
    )
    result = {
        "version": "reward-alignment-v1",
        "seed": seed,
        "k": k,
        "scoring_signal": "lr_bacc",
        "scoring_aggregation": aggregation,
        "reward_archive_stop_objective": f"lr_bacc/{aggregation}",
        "correlation_penalty": 0.0,
        "strict_epsilon": EPS,
        "forward_clean_indices_0based": list(forward),
        "forward_original_feature_ids_1based": [final_ids[index] for index in forward],
        "forward_objective": forward_score,
        "endpoint_clean_indices_0based": list(selected),
        "endpoint_original_feature_ids_1based": [final_ids[index] for index in selected],
        "endpoint_objective": current_score,
        "accepted_swap_count": accepted_swaps,
        "termination": termination,
        "path": path,
        "archive": archive,
        "cost": scorer.costs(),
        "wall_seconds": time.perf_counter() - started,
        "outer_evaluation_fit_count": 4,
        "outer_forward": forward_outer,
        "outer_endpoint": endpoint_outer,
        "outer_lr_bacc_gain": float(endpoint_outer["lr_bacc"] - forward_outer["lr_bacc"]),
        "coordinate_crosscheck_passed": True,
    }
    cache_rows = []
    for candidate_id, (subset, score) in enumerate(scorer.cache.items()):
        cache_rows.append(
            {
                "candidate_id": candidate_id,
                "clean_indices_0based": list(subset),
                "original_feature_ids_1based": [final_ids[index] for index in subset],
                **score,
            }
        )
    write_jsonl(out / "candidates.jsonl", cache_rows)
    write_json(out / "result.json", result)
    write_json(out / "complete.json", {"result_sha256": sha256(out / "result.json")})
    print(
        f"aligned search seed={seed} K={k}: swaps={accepted_swaps} "
        f"outer_gain={result['outer_lr_bacc_gain']:.6f}",
        flush=True,
    )


def run_aligned_search(jobs: int) -> None:
    selection = read_json(ROOT / "analysis" / "selected-scorer.json")
    aggregation = selection["selected"]["aggregation"]
    for seed in SEEDS:
        for k in KS:
            aligned_search_case(seed, k, aggregation, jobs)
    search_root = ROOT / "aligned_search"
    files = sorted(path for path in search_root.rglob("*") if path.is_file())
    write_json(
        ROOT / "aligned-search-complete.json",
        {
            "selected_signal": "lr_bacc",
            "selected_aggregation": aggregation,
            "files": {str(path): sha256(path) for path in files},
        },
    )


def finalize() -> None:
    verify_frozen_files(ROOT / "aligned-search-complete.json")
    selection = read_json(ROOT / "analysis" / "selected-scorer.json")
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for k in KS:
            result = read_json(ROOT / "aligned_search" / f"seed-{seed}" / f"k{k}" / "result.json")
            cost = result["cost"]
            rows.append(
                {
                    "seed": seed,
                    "k": k,
                    "accepted_swap_count": result["accepted_swap_count"],
                    "termination": result["termination"],
                    "forward_objective": result["forward_objective"],
                    "endpoint_objective": result["endpoint_objective"],
                    "objective_gain": result["endpoint_objective"] - result["forward_objective"],
                    "outer_forward_lr_bacc": result["outer_forward"]["lr_bacc"],
                    "outer_endpoint_lr_bacc": result["outer_endpoint"]["lr_bacc"],
                    "outer_lr_bacc_gain": result["outer_lr_bacc_gain"],
                    "candidate_requests": cost["candidate_requests"],
                    "unique_scored_subsets": cost["unique_scored_subsets"],
                    "cache_hits": cost["cache_hits"],
                    "classifier_fit_count": cost["classifier_fit_count"],
                    "fit_seconds": cost["fit_seconds"],
                    "wall_seconds": result["wall_seconds"],
                    "within_cost_ceiling": cost["unique_scored_subsets"] <= 4000
                    and cost["classifier_fit_count"] <= 60000,
                }
            )
    write_csv(ROOT / "analysis" / "aligned_search_results.csv", rows)
    summary: list[dict[str, Any]] = []
    for k, frame in pd.DataFrame(rows).groupby("k", sort=True):
        positive = int((frame["outer_lr_bacc_gain"] > EPS).sum())
        mean_gain = float(frame["outer_lr_bacc_gain"].mean())
        search_gate = positive >= 4 and mean_gain >= 0.002
        cost_gate = bool(frame["within_cost_ceiling"].all())
        bank_gate = bool(selection["selected"]["bank_gate_by_k"][str(int(k))]["passed"])
        summary.append(
            {
                "k": int(k),
                "outer_lr_bacc_gain_mean": mean_gain,
                "outer_lr_bacc_gain_sd": float(frame["outer_lr_bacc_gain"].std(ddof=1)),
                "positive_partitions": positive,
                "zero_partitions": int((frame["outer_lr_bacc_gain"].abs() <= EPS).sum()),
                "negative_partitions": int((frame["outer_lr_bacc_gain"] < -EPS).sum()),
                "bank_gate_passed": bank_gate,
                "search_gain_gate_passed": search_gate,
                "cost_gate_passed": cost_gate,
                "preaudit_k_passed": bank_gate and search_gate and cost_gate,
            }
        )
    write_csv(ROOT / "analysis" / "aligned_search_summary.csv", summary)
    preaudit_pass = all(row["preaudit_k_passed"] for row in summary)
    write_json(
        ROOT / "analysis" / "decision-preaudit.json",
        {
            "selected_diagnostic_scorer": selection["selected"],
            "both_k_pre_audit_passed": preaudit_pass,
            "decision_if_audit_passes": "freeze_unique_main_reward" if preaudit_pass else "no-go",
            "unique_main_reward_if_passed": (
                f"fold-local StandardScaler+LR Balanced Accuracy / {selection['selected']['aggregation']}"
                if preaudit_pass
                else None
            ),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("bank", "score", "validate", "analyze", "search", "finalize"), required=True
    )
    parser.add_argument("--jobs", type=int, default=24)
    args = parser.parse_args()
    prepare_manifest()
    with parallel_config(backend="loky", inner_max_num_threads=1):
        if args.stage == "bank":
            freeze_banks()
        elif args.stage == "score":
            score_banks(args.jobs)
        elif args.stage == "validate":
            validate_banks(args.jobs)
        elif args.stage == "analyze":
            analyze_banks()
        elif args.stage == "search":
            run_aligned_search(args.jobs)
        else:
            finalize()


if __name__ == "__main__":
    main()
