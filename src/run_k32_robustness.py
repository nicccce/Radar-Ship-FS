#!/usr/bin/env python3
"""Task 4C: locked K=32 post-selection development robustness check."""

from __future__ import annotations

import argparse
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
from joblib import Parallel, delayed
from sklearn.datasets import load_svmlight_file
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from data.loader import _find_unique_columns
from run_reward_alignment import all_single_swaps, canonical_subset, sha256

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 environment
    import tomli as tomllib

ROOT = Path("experiments/k32_reward_robustness_v1")
TRAIN = Path("../dataset/sim_ship_cr_v16n_2x_noise.train.svm")
CONFIG = Path("configs/v16n/k32_reward_robustness_v1.toml")
PROTOCOL = Path("documents/research-plan/04c-k32-robustness-protocol.md")
TRAIN_SHA256 = "2191fdc297aa49ce6a700df956ec44f68dc13e4c737937b31167543a203d22bb"
K = 32
N_FEATURES = 65
EPS = 1e-12
OUTER_RANDOM_STATE = 20260915
INNER_BASE_STATES = (202609151, 202609152, 202609153)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def write_csv(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(target, index=False)


def load_raw() -> tuple[np.ndarray, np.ndarray]:
    if sha256(TRAIN) != TRAIN_SHA256:
        raise RuntimeError("source-train hash differs from the frozen protocol")
    sparse, labels = load_svmlight_file(TRAIN, n_features=75)
    raw = sparse.toarray().astype(np.float32)
    labels = labels.astype(np.int64)
    if raw.shape != (3897, 75):
        raise AssertionError(f"unexpected source-train shape: {raw.shape}")
    return raw, labels


def clean_on_outer_train(values: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    constant = np.isclose(values.min(axis=0), values.max(axis=0))
    nonconstant = np.flatnonzero(~constant)
    unique, duplicates = _find_unique_columns(values[:, nonconstant])
    final_ids = nonconstant[unique] + 1
    metadata = {
        "constant_feature_ids_1based": (np.flatnonzero(constant) + 1).astype(int).tolist(),
        "duplicate_feature_mapping_1based": {
            str(int(nonconstant[removed] + 1)): int(nonconstant[kept] + 1)
            for removed, kept in duplicates.items()
        },
        "final_feature_ids_1based": final_ids.astype(int).tolist(),
        "fit_scope": "outer_train_only",
    }
    return final_ids.astype(int), metadata


def inner_repeats(
    outer_train: np.ndarray, labels: np.ndarray, outer_fold: int
) -> list[list[dict[str, list[int]]]]:
    repeats = []
    for base in INNER_BASE_STATES:
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=base + outer_fold)
        repeats.append(
            [
                {
                    "fit": outer_train[fit].astype(int).tolist(),
                    "held_out": outer_train[held].astype(int).tolist(),
                }
                for fit, held in splitter.split(outer_train, labels[outer_train])
            ]
        )
    return repeats


def _frozen_sources() -> list[Path]:
    return [
        Path(__file__),
        Path("src/audit_k32_robustness.py"),
        Path("src/run_reward_alignment.py"),
        Path("src/data/loader.py"),
        Path("tests/test_k32_robustness.py"),
        CONFIG,
        PROTOCOL,
    ]


def verify_manifest() -> None:
    manifest = read_json(ROOT / "manifest.json")
    for path, expected in manifest["code_protocol_config_hashes"].items():
        if sha256(path) != expected:
            raise RuntimeError(f"frozen source changed after prepare: {path}")
    marker = read_json(ROOT / "partition-frozen.json")
    for path, expected in marker["files"].items():
        if sha256(path) != expected:
            raise RuntimeError(f"frozen partition/context changed: {path}")


def prepare() -> None:
    with CONFIG.open("rb") as handle:
        config = tomllib.load(handle)
    if config["outer"]["random_state"] != OUTER_RANDOM_STATE or config["search"]["k"] != K:
        raise AssertionError("checked-in config differs from locked constants")
    if Path("documents/research-plan/03a-data-lineage-audit.md").exists():
        raise RuntimeError(
            "3A appeared; split choice must be reviewed before running this row-split protocol"
        )
    if ROOT.exists():
        verify_manifest()
        return
    raw, labels = load_raw()
    ROOT.mkdir(parents=True, exist_ok=False)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=OUTER_RANDOM_STATE)
    assignments = np.full(len(labels), -1, dtype=int)
    contexts = []
    for outer_fold, (train, validation) in enumerate(splitter.split(raw, labels)):
        train = train.astype(int)
        validation = validation.astype(int)
        assignments[validation] = outer_fold
        final_ids, cleaning = clean_on_outer_train(raw[train])
        if len(final_ids) != N_FEATURES:
            raise AssertionError(f"outer fold {outer_fold} produced {len(final_ids)} cleaned features")
        context = {
            "outer_fold": outer_fold,
            "outer_kind": "StratifiedKFold",
            "outer_random_state": OUTER_RANDOM_STATE,
            "outer_train_rows": train.tolist(),
            "outer_validation_rows": validation.tolist(),
            "outer_train_label_counts": {
                str(int(label)): int((labels[train] == label).sum()) for label in np.unique(labels)
            },
            "outer_validation_label_counts": {
                str(int(label)): int((labels[validation] == label).sum()) for label in np.unique(labels)
            },
            "cleaning": cleaning,
            "final_feature_ids_1based": final_ids.tolist(),
            "inner_repeat_random_states": [base + outer_fold for base in INNER_BASE_STATES],
            "inner_repeat_fold_rows": inner_repeats(train, labels, outer_fold),
            "lr_random_state": OUTER_RANDOM_STATE + outer_fold,
            "dt_random_state": OUTER_RANDOM_STATE + outer_fold,
            "mi_random_state": OUTER_RANDOM_STATE + outer_fold,
        }
        path = ROOT / "contexts" / f"outer-fold-{outer_fold}.json"
        write_json(path, context)
        contexts.append({"outer_fold": outer_fold, "path": str(path)})
    if np.any(assignments < 0):
        raise AssertionError("outer partition does not validate every row exactly once")
    partition = {
        "version": "k32-reward-robustness-v1",
        "created_before_any_4c_scoring": True,
        "group_metadata_status": "unavailable_03a_absent",
        "splitter": {
            "kind": "StratifiedKFold",
            "n_splits": 5,
            "shuffle": True,
            "random_state": OUTER_RANDOM_STATE,
        },
        "source_train_rows": len(labels),
        "validation_fold_by_original_row": assignments.tolist(),
        "contexts": contexts,
    }
    write_json(ROOT / "outer-partition.json", partition)
    frozen_files = [ROOT / "outer-partition.json", *sorted((ROOT / "contexts").glob("*.json"))]
    write_json(
        ROOT / "partition-frozen.json",
        {"files": {str(path): sha256(path) for path in frozen_files}},
    )
    write_json(
        ROOT / "manifest.json",
        {
            "version": "k32-reward-robustness-v1",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "post_selection_development_check": True,
            "task_4b_status": "no-go",
            "task_4b_unique_main_reward": None,
            "k16_closed": True,
            "source_test_open_count": 0,
            "rl_training_count": 0,
            "source_train_sha256": sha256(TRAIN),
            "outer_partition_sha256": sha256(ROOT / "outer-partition.json"),
            "code_protocol_config_hashes": {str(path): sha256(path) for path in _frozen_sources()},
            "python": sys.version,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "git_status_before_results": subprocess.check_output(
                ["git", "status", "--short"], text=True
            ),
        },
    )
    print(f"frozen outer partition: {sha256(ROOT / 'outer-partition.json')}")


def fit_lr_predict(
    raw: np.ndarray,
    labels: np.ndarray,
    final_ids: Sequence[int],
    subset: Sequence[int],
    fit_rows: Sequence[int],
    held_rows: Sequence[int],
    random_state: int,
) -> tuple[np.ndarray, float]:
    columns = np.asarray(final_ids, dtype=int)[np.asarray(subset, dtype=int)] - 1
    fit = np.asarray(fit_rows, dtype=int)
    held = np.asarray(held_rows, dtype=int)
    started = time.perf_counter()
    scaler = StandardScaler().fit(raw[fit][:, columns])
    classifier = LogisticRegression(
        C=1.0,
        solver="liblinear",
        class_weight="balanced",
        max_iter=5000,
        random_state=random_state,
    )
    classifier.fit(scaler.transform(raw[fit][:, columns]), labels[fit])
    prediction = classifier.predict(scaler.transform(raw[held][:, columns]))
    return prediction.astype(int), time.perf_counter() - started


class LRScorer:
    def __init__(self, raw: np.ndarray, labels: np.ndarray, context: dict[str, Any], jobs: int) -> None:
        self.raw = raw
        self.labels = labels
        self.context = context
        self.final_ids = tuple(context["final_feature_ids_1based"])
        self.jobs = jobs
        self.cache: dict[tuple[int, ...], dict[str, Any]] = {}
        self.requests = 0
        self.cache_hits = 0
        self.fit_count = 0
        self.fit_seconds = 0.0

    @staticmethod
    def _score_one(
        raw: np.ndarray,
        labels: np.ndarray,
        final_ids: tuple[int, ...],
        subset: tuple[int, ...],
        repeats: Sequence[Sequence[dict[str, Sequence[int]]]],
        random_state: int,
    ) -> dict[str, Any]:
        fold_scores: list[list[float]] = []
        started = time.perf_counter()
        for repeat in repeats:
            values = []
            for fold in repeat:
                prediction, _ = fit_lr_predict(
                    raw,
                    labels,
                    final_ids,
                    subset,
                    fold["fit"],
                    fold["held_out"],
                    random_state,
                )
                values.append(
                    float(
                        balanced_accuracy_score(
                            labels[np.asarray(fold["held_out"], dtype=int)], prediction
                        )
                    )
                )
            fold_scores.append(values)
        repeat_means = [float(np.mean(values)) for values in fold_scores]
        objective = float(np.mean(repeat_means) - 0.5 * np.std(repeat_means, ddof=1))
        return {
            "objective": objective,
            "repeat_means": repeat_means,
            "fold_scores": fold_scores,
            "classifier_fit_count": 15,
            "fit_seconds": time.perf_counter() - started,
            "scaler_fit_scope": "each_inner_training_fold",
        }

    def score_many(self, subsets: Sequence[Sequence[int]]) -> list[float]:
        keys = [canonical_subset(subset, len(self.final_ids)) for subset in subsets]
        self.requests += len(keys)
        missing = []
        seen = set()
        for key in keys:
            if key in self.cache or key in seen:
                self.cache_hits += 1
            else:
                seen.add(key)
                missing.append(key)
        computed = Parallel(n_jobs=self.jobs, prefer="processes")(
            delayed(self._score_one)(
                self.raw,
                self.labels,
                self.final_ids,
                subset,
                self.context["inner_repeat_fold_rows"],
                int(self.context["lr_random_state"]),
            )
            for subset in missing
        )
        for subset, payload in zip(missing, computed):
            self.cache[subset] = payload
            self.fit_count += int(payload["classifier_fit_count"])
            self.fit_seconds += float(payload["fit_seconds"])
        return [float(self.cache[key]["objective"]) for key in keys]

    def costs(self) -> dict[str, Any]:
        return {
            "candidate_requests": self.requests,
            "unique_scored_subsets": len(self.cache),
            "cache_hits": self.cache_hits,
            "classifier_fit_count": self.fit_count,
            "fit_seconds": self.fit_seconds,
        }


class DTScorer:
    def __init__(self, raw: np.ndarray, labels: np.ndarray, context: dict[str, Any], jobs: int) -> None:
        self.raw = raw
        self.labels = labels
        self.context = context
        self.final_ids = tuple(context["final_feature_ids_1based"])
        self.folds = context["inner_repeat_fold_rows"][0]
        self.jobs = jobs
        self.cache: dict[tuple[int, ...], dict[str, Any]] = {}
        self.requests = 0
        self.cache_hits = 0
        self.fit_count = 0
        self.fit_seconds = 0.0

    @staticmethod
    def _score_one(raw, labels, final_ids, subset, folds, random_state):
        columns = np.asarray(final_ids, dtype=int)[np.asarray(subset, dtype=int)] - 1
        values = []
        started = time.perf_counter()
        for fold in folds:
            fit = np.asarray(fold["fit"], dtype=int)
            held = np.asarray(fold["held_out"], dtype=int)
            tree = DecisionTreeClassifier(random_state=random_state)
            tree.fit(raw[fit][:, columns], labels[fit])
            values.append(float(accuracy_score(labels[held], tree.predict(raw[held][:, columns]))))
        return {
            "objective": float(np.mean(values)),
            "fold_scores": values,
            "classifier_fit_count": 5,
            "fit_seconds": time.perf_counter() - started,
        }

    def score_many(self, subsets):
        keys = [canonical_subset(subset, len(self.final_ids)) for subset in subsets]
        self.requests += len(keys)
        missing = []
        seen = set()
        for key in keys:
            if key in self.cache or key in seen:
                self.cache_hits += 1
            else:
                seen.add(key)
                missing.append(key)
        computed = Parallel(n_jobs=self.jobs, prefer="processes")(
            delayed(self._score_one)(
                self.raw,
                self.labels,
                self.final_ids,
                subset,
                self.folds,
                int(self.context["dt_random_state"]),
            )
            for subset in missing
        )
        for subset, payload in zip(missing, computed):
            self.cache[subset] = payload
            self.fit_count += 5
            self.fit_seconds += float(payload["fit_seconds"])
        return [float(self.cache[key]["objective"]) for key in keys]

    def costs(self):
        return {
            "candidate_requests": self.requests,
            "unique_scored_subsets": len(self.cache),
            "cache_hits": self.cache_hits,
            "classifier_fit_count": self.fit_count,
            "fit_seconds": self.fit_seconds,
        }


def exact_k_forward(scorer, n_features: int, k: int, label: str):
    selected: tuple[int, ...] = ()
    path = []
    for step in range(1, k + 1):
        candidates = [
            tuple(sorted((*selected, feature)))
            for feature in range(n_features)
            if feature not in selected
        ]
        scores = scorer.score_many(candidates)
        best_index = int(np.argmax(np.asarray(scores, dtype=float)))
        selected = candidates[best_index]
        path.append(
            {
                "phase": f"{label}_forward",
                "step": step,
                "candidate_count": len(candidates),
                "chosen_clean_indices_0based": list(selected),
                "chosen_objective": float(scores[best_index]),
                "accepted": True,
                "cumulative_cost": dict(scorer.costs()),
            }
        )
    return selected, float(path[-1]["chosen_objective"]), path


def best_improvement_swaps(scorer, start, start_score, max_rounds: int, label: str):
    selected = tuple(start)
    current = float(start_score)
    path = []
    archive = [{"round": 0, "clean_indices_0based": list(selected), "objective": current}]
    termination = "max_swap_rounds_reached_without_optimality_confirmation"
    for round_index in range(1, max_rounds + 1):
        candidates = all_single_swaps(selected, N_FEATURES)
        scores = scorer.score_many(candidates)
        best_index = int(np.argmax(np.asarray(scores, dtype=float)))
        chosen = candidates[best_index]
        chosen_score = float(scores[best_index])
        accepted = chosen_score > current + EPS
        path.append(
            {
                "phase": f"{label}_single_swap",
                "round": round_index,
                "candidate_count": len(candidates),
                "chosen_clean_indices_0based": list(chosen),
                "chosen_objective": chosen_score,
                "objective_gain": chosen_score - current,
                "accepted": accepted,
                "reason": "strict_improvement" if accepted else "no_strict_improvement",
                "cumulative_cost": dict(scorer.costs()),
            }
        )
        if not accepted:
            termination = "no_strict_single_swap_improvement"
            break
        selected, current = chosen, chosen_score
        archive.append(
            {"round": round_index, "clean_indices_0based": list(selected), "objective": current}
        )
    return selected, current, path, archive, termination


def _cache_rows(cache, final_ids):
    rows = []
    for candidate_id, (subset, payload) in enumerate(cache.items()):
        original = [int(final_ids[index]) for index in subset]
        if tuple(np.flatnonzero(np.isin(final_ids, original)).tolist()) != subset:
            raise AssertionError("clean/original coordinate cross-check failed")
        rows.append(
            {
                "candidate_id": candidate_id,
                "clean_indices_0based": list(subset),
                "original_feature_ids_1based": original,
                "coordinate_crosscheck_passed": True,
                **payload,
            }
        )
    return rows


def run_case(outer_fold: int, jobs: int) -> None:
    out = ROOT / "results" / f"outer-fold-{outer_fold}"
    if (out / "complete.json").exists():
        return
    if out.exists():
        raise RuntimeError(f"partial result refuses overwrite: {out}")
    verify_manifest()
    raw, labels = load_raw()
    context = read_json(ROOT / "contexts" / f"outer-fold-{outer_fold}.json")
    final_ids = np.asarray(context["final_feature_ids_1based"], dtype=int)
    outer_train = np.asarray(context["outer_train_rows"], dtype=int)
    outer_validation = np.asarray(context["outer_validation_rows"], dtype=int)
    started = time.perf_counter()

    lr_scorer = LRScorer(raw, labels, context, jobs)
    lr_forward, lr_forward_score, lr_path = exact_k_forward(lr_scorer, N_FEATURES, K, "lr")
    lr_forward_cost = dict(lr_scorer.costs())
    lr_endpoint, lr_endpoint_score, lr_swap_path, lr_archive, lr_termination = best_improvement_swaps(
        lr_scorer, lr_forward, lr_forward_score, 2, "lr"
    )

    dt_scorer = DTScorer(raw, labels, context, jobs)
    dt_forward, dt_forward_score, dt_path = exact_k_forward(dt_scorer, N_FEATURES, K, "dt")
    dt_forward_cost = dict(dt_scorer.costs())
    dt_endpoint, dt_endpoint_score, dt_swap_path, dt_archive, dt_termination = best_improvement_swaps(
        dt_scorer, dt_forward, dt_forward_score, 32, "dt"
    )

    mi_started = time.perf_counter()
    mi_values = mutual_info_classif(
        raw[outer_train][:, final_ids - 1], labels[outer_train], random_state=context["mi_random_state"]
    )
    mi_subset = tuple(np.argsort(-mi_values, kind="stable")[:K].astype(int).tolist())
    mi_seconds = time.perf_counter() - mi_started
    methods = {
        "all_features": tuple(range(N_FEATURES)),
        "mi_32": tuple(sorted(mi_subset)),
        "lr_forward": lr_forward,
        "lr_forward_swap2": lr_endpoint,
        "task2_dt_forward_k32": dt_forward,
        "task2_dt_forward_single_swap_k32": dt_endpoint,
    }
    prediction_rows = [
        {"original_row": int(row), "outer_fold": outer_fold, "y_true": int(labels[row])}
        for row in outer_validation
    ]
    evaluations = {}
    for method, subset in methods.items():
        prediction, seconds = fit_lr_predict(
            raw,
            labels,
            final_ids,
            subset,
            outer_train,
            outer_validation,
            int(context["lr_random_state"]),
        )
        for row, value in zip(prediction_rows, prediction):
            row[method] = int(value)
        evaluations[method] = {
            "clean_indices_0based": list(subset),
            "original_feature_ids_1based": [int(final_ids[index]) for index in subset],
            "lr_balanced_accuracy": float(balanced_accuracy_score(labels[outer_validation], prediction)),
            "lr_accuracy": float(accuracy_score(labels[outer_validation], prediction)),
            "endpoint_classifier_fit_count": 1,
            "endpoint_fit_seconds": seconds,
        }

    result = {
        "version": "k32-reward-robustness-v1",
        "outer_fold": outer_fold,
        "k": K,
        "source_test_open_count": 0,
        "rl_training_count": 0,
        "outer_validation_used_during_selection": False,
        "scorer": {
            "signal": "lr_balanced_accuracy",
            "aggregation": "mean_minus_0_5_sample_sd",
            "repeat_count": 3,
            "fold_count": 5,
            "reward_candidate_archive_stop": "same_J",
        },
        "lr_search": {
            "forward_clean_indices_0based": list(lr_forward),
            "forward_objective": lr_forward_score,
            "endpoint_clean_indices_0based": list(lr_endpoint),
            "endpoint_objective": lr_endpoint_score,
            "accepted_swap_count": sum(step["accepted"] for step in lr_swap_path),
            "termination": lr_termination,
            "path": lr_path + lr_swap_path,
            "archive": lr_archive,
            "forward_cost": lr_forward_cost,
            "total_cost": lr_scorer.costs(),
        },
        "task2_dt_search": {
            "forward_clean_indices_0based": list(dt_forward),
            "forward_objective": dt_forward_score,
            "endpoint_clean_indices_0based": list(dt_endpoint),
            "endpoint_objective": dt_endpoint_score,
            "accepted_swap_count": sum(step["accepted"] for step in dt_swap_path),
            "termination": dt_termination,
            "path": dt_path + dt_swap_path,
            "archive": dt_archive,
            "forward_cost": dt_forward_cost,
            "total_cost": dt_scorer.costs(),
        },
        "mi_cost": {"fit_count": 1, "fit_seconds": mi_seconds},
        "evaluations": evaluations,
        "lr_forward_to_swap_gain": float(
            evaluations["lr_forward_swap2"]["lr_balanced_accuracy"]
            - evaluations["lr_forward"]["lr_balanced_accuracy"]
        ),
        "wall_seconds": time.perf_counter() - started,
        "dfs": "unavailable / protocol provenance incomplete",
    }
    write_jsonl(out / "lr_candidates.jsonl", _cache_rows(lr_scorer.cache, final_ids))
    write_jsonl(out / "dt_candidates.jsonl", _cache_rows(dt_scorer.cache, final_ids))
    write_csv(out / "oof_predictions.csv", prediction_rows)
    write_json(out / "result.json", result)
    write_json(
        out / "complete.json",
        {
            "result_sha256": sha256(out / "result.json"),
            "lr_candidates_sha256": sha256(out / "lr_candidates.jsonl"),
            "dt_candidates_sha256": sha256(out / "dt_candidates.jsonl"),
            "oof_predictions_sha256": sha256(out / "oof_predictions.csv"),
        },
    )
    print(
        f"outer fold {outer_fold}: LR swaps={result['lr_search']['accepted_swap_count']} "
        f"gain={100 * result['lr_forward_to_swap_gain']:+.4f} pp",
        flush=True,
    )


def run_all(jobs: int) -> None:
    verify_manifest()
    for outer_fold in range(5):
        run_case(outer_fold, jobs)
    files = sorted(path for path in (ROOT / "results").rglob("*") if path.is_file())
    write_json(ROOT / "results-complete.json", {"files": {str(path): sha256(path) for path in files}})


def finalize() -> None:
    verify_manifest()
    marker = read_json(ROOT / "results-complete.json")
    for path, expected in marker["files"].items():
        if sha256(path) != expected:
            raise RuntimeError(f"result changed after completion: {path}")
    raw, labels = load_raw()
    fold_rows = []
    oof_frames = []
    costs = []
    methods = None
    for outer_fold in range(5):
        result = read_json(ROOT / "results" / f"outer-fold-{outer_fold}" / "result.json")
        frame = pd.read_csv(ROOT / "results" / f"outer-fold-{outer_fold}" / "oof_predictions.csv")
        oof_frames.append(frame)
        methods = list(result["evaluations"])
        fold_rows.append(
            {
                "outer_fold": outer_fold,
                **{
                    f"{method}_lr_bacc": result["evaluations"][method]["lr_balanced_accuracy"]
                    for method in methods
                },
                "lr_forward_to_swap_gain": result["lr_forward_to_swap_gain"],
                "lr_accepted_swaps": result["lr_search"]["accepted_swap_count"],
                "lr_termination": result["lr_search"]["termination"],
            }
        )
        for family in ("lr_search", "task2_dt_search"):
            costs.append(
                {
                    "outer_fold": outer_fold,
                    "family": family,
                    **result[family]["total_cost"],
                    "wall_seconds_total_case": result["wall_seconds"],
                }
            )
    oof = pd.concat(oof_frames, ignore_index=True).sort_values("original_row")
    if len(oof) != len(labels) or oof["original_row"].nunique() != len(labels):
        raise AssertionError("OOF rows are not exhaustive and unique")
    pooled = []
    for method in methods or []:
        pooled.append(
            {
                "method": method,
                "pooled_oof_lr_bacc": float(balanced_accuracy_score(oof["y_true"], oof[method])),
                "mean_fold_lr_bacc": float(
                    np.mean([row[f"{method}_lr_bacc"] for row in fold_rows])
                ),
                "sd_fold_lr_bacc": float(
                    np.std([row[f"{method}_lr_bacc"] for row in fold_rows], ddof=1)
                ),
            }
        )
    forward_pooled = next(row for row in pooled if row["method"] == "lr_forward")
    swap_pooled = next(row for row in pooled if row["method"] == "lr_forward_swap2")
    gains = np.asarray([row["lr_forward_to_swap_gain"] for row in fold_rows], dtype=float)
    lr_cost_ok = all(
        row["family"] != "lr_search"
        or (row["unique_scored_subsets"] <= 4000 and row["classifier_fit_count"] <= 60000)
        for row in costs
    )
    preaudit = {
        "status": "pending-independent-audit",
        "positive_outer_folds": int((gains > EPS).sum()),
        "zero_outer_folds": int((np.abs(gains) <= EPS).sum()),
        "negative_outer_folds": int((gains < -EPS).sum()),
        "mean_fold_gain": float(np.mean(gains)),
        "sd_fold_gain": float(np.std(gains, ddof=1)),
        "pooled_oof_gain": float(
            swap_pooled["pooled_oof_lr_bacc"] - forward_pooled["pooled_oof_lr_bacc"]
        ),
        "positive_fold_gate": int((gains > EPS).sum()) >= 4,
        "mean_gain_gate": float(np.mean(gains)) >= 0.002,
        "cost_gate": lr_cost_ok,
        "numerical_and_structural_audit_gate": None,
        "task_4b_status_unchanged": "no-go",
        "task_4b_unique_main_reward_unchanged": None,
    }
    write_csv(ROOT / "analysis" / "fold_results.csv", fold_rows)
    write_csv(ROOT / "analysis" / "pooled_results.csv", pooled)
    write_csv(ROOT / "analysis" / "costs.csv", costs)
    oof.to_csv(ROOT / "analysis" / "oof_predictions.csv", index=False)
    write_json(ROOT / "analysis" / "decision-preaudit.json", preaudit)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare", "run", "finalize", "all"), required=True)
    parser.add_argument("--jobs", type=int, default=24)
    args = parser.parse_args()
    if args.stage in ("prepare", "all"):
        prepare()
    if args.stage in ("run", "all"):
        run_all(args.jobs)
    if args.stage in ("finalize", "all"):
        finalize()


if __name__ == "__main__":
    main()

