#!/usr/bin/env python3
"""Protocol-frozen, classifier-aligned strong feature-selection baselines.

Selection is a train-only stage. Final source-test diagnostics are a separate, resumable stage that
loads the frozen test file once after every seed/subset is complete.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
import sklearn
from joblib import Parallel, delayed
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 experiment environment
    import tomli as tomllib

from data.loader import load_radar_ship_source_test, load_radar_ship_source_train
from radar_ship_fs.feature_mapping import FeatureIndexMap

REQUIRED_METHODS = (
    "all_features",
    "mi_topk",
    "mi_ordered_accept",
    "forward_greedy",
    "forward_greedy_single_swap",
)


def _canonical_subset(values: Sequence[int], n_features: int) -> tuple[int, ...]:
    subset = tuple(sorted(int(value) for value in values))
    if not subset:
        raise ValueError("a scored subset must contain at least one feature")
    if len(subset) != len(set(subset)):
        raise ValueError("a subset must not contain duplicate features")
    if subset[0] < 0 or subset[-1] >= n_features:
        raise ValueError("a subset contains an out-of-range feature")
    return subset


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(temporary, path)


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git(args: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip()


@dataclass(frozen=True)
class UnifiedConfig:
    path: Path
    raw_text: str
    raw: dict[str, Any]
    config_hash: str

    @property
    def protocol_version(self) -> str:
        return str(self.raw["protocol_version"])

    @property
    def dataset(self) -> dict[str, Any]:
        return self.raw["dataset"]

    @property
    def selection(self) -> dict[str, Any]:
        return self.raw["selection"]

    @property
    def final_classifier(self) -> dict[str, Any]:
        return self.raw["final_classifier"]

    @property
    def output_root(self) -> Path:
        return Path(self.raw["output"]["root"])

    @property
    def resume(self) -> bool:
        return bool(self.raw["output"]["resume"])


def load_unified_config(path: str | Path) -> UnifiedConfig:
    source = Path(path)
    raw_bytes = source.read_bytes()
    raw = tomllib.loads(raw_bytes.decode("utf-8"))
    required_sections = {
        "schema_version",
        "protocol_version",
        "dataset",
        "selection",
        "final_classifier",
        "output",
    }
    if set(raw) != required_sections:
        raise ValueError(
            f"config top-level fields must be exactly {sorted(required_sections)}, got {sorted(raw)}"
        )
    if int(raw["schema_version"]) != 1:
        raise ValueError("schema_version must be 1")
    if str(raw["protocol_version"]) not in {
        "unified-strong-baselines-v1",
        "unified-strong-baselines-smoke-v1",
    }:
        raise ValueError("unsupported protocol_version")

    dataset = raw["dataset"]
    required_dataset = {
        "name",
        "version",
        "data_dir",
        "seeds",
        "validation_fraction",
        "inner_cv_folds",
        "expected_original_features",
        "expected_clean_features",
        "expected_source_train_rows",
        "expected_source_test_rows",
        "source_train_sha256",
        "source_test_sha256",
    }
    if set(dataset) != required_dataset:
        raise ValueError("dataset fields do not match the unified baseline schema")
    seeds = tuple(int(seed) for seed in dataset["seeds"])
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("dataset.seeds must be a non-empty unique list")
    if not 0.0 < float(dataset["validation_fraction"]) < 1.0:
        raise ValueError("dataset.validation_fraction must be in (0, 1)")
    if int(dataset["inner_cv_folds"]) < 2:
        raise ValueError("dataset.inner_cv_folds must be at least 2")

    selection = raw["selection"]
    required_selection = {
        "methods",
        "fixed_k",
        "k_max",
        "strict_improvement_tolerance",
        "max_swap_rounds",
        "n_jobs",
    }
    if set(selection) != required_selection:
        raise ValueError("selection fields do not match the unified baseline schema")
    methods = tuple(str(name) for name in selection["methods"])
    if methods != REQUIRED_METHODS:
        raise ValueError(f"selection.methods must be exactly {list(REQUIRED_METHODS)}")
    fixed_k = tuple(int(value) for value in selection["fixed_k"])
    if fixed_k != tuple(sorted(set(fixed_k))) or not fixed_k:
        raise ValueError("selection.fixed_k must be a non-empty strictly increasing list")
    if fixed_k[-1] != int(selection["k_max"]):
        raise ValueError("largest fixed K must equal selection.k_max")
    if fixed_k[0] <= 0 or fixed_k[-1] > int(dataset["expected_clean_features"]):
        raise ValueError("fixed K values are outside the cleaned feature space")
    if float(selection["strict_improvement_tolerance"]) < 0.0:
        raise ValueError("strict_improvement_tolerance must be non-negative")
    if int(selection["max_swap_rounds"]) <= 0:
        raise ValueError("max_swap_rounds must be positive")
    if int(selection["n_jobs"]) == 0:
        raise ValueError("n_jobs cannot be zero")

    final_classifier = raw["final_classifier"]
    required_final = {
        "primary",
        "C",
        "solver",
        "max_iter",
        "class_weight",
        "secondary",
    }
    if set(final_classifier) != required_final:
        raise ValueError("final_classifier fields do not match the unified baseline schema")
    if final_classifier["primary"] != "logistic_regression_pipeline":
        raise ValueError("primary final classifier must remain logistic_regression_pipeline")
    if final_classifier["secondary"] != "decision_tree":
        raise ValueError("secondary final classifier must remain decision_tree")

    output = raw["output"]
    if set(output) != {"root", "resume"} or not output["root"]:
        raise ValueError("output requires non-empty root and resume")
    canonical = json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return UnifiedConfig(
        path=source,
        raw_text=raw_bytes.decode("utf-8"),
        raw=raw,
        config_hash=_sha256_bytes(canonical.encode("utf-8")),
    )


@dataclass(frozen=True)
class SeedContext:
    seed: int
    X: np.ndarray
    y: np.ndarray
    development_original_rows: np.ndarray
    initial_train_rows: np.ndarray
    initial_validation_rows: np.ndarray
    folds: tuple[tuple[np.ndarray, np.ndarray], ...]
    fold_original_rows: tuple[dict[str, list[int]], ...]
    split_random_state: int
    cv_tree_random_state: int
    mi_random_state: int


def build_seed_context(
    X_source_train: np.ndarray,
    y_source_train: np.ndarray,
    *,
    seed: int,
    validation_fraction: float,
    n_splits: int,
) -> SeedContext:
    """Reproduce the protocol's split-then-recombine RNG and row-order contract."""

    rng = np.random.default_rng(seed)
    split_random_state = int(rng.integers(0, 2**32))
    source_rows = np.arange(X_source_train.shape[0], dtype=int)
    initial_train, initial_validation = train_test_split(
        source_rows,
        test_size=validation_fraction,
        random_state=split_random_state,
        stratify=y_source_train,
    )
    development_rows = np.concatenate((initial_train, initial_validation))
    X = X_source_train[development_rows]
    y = y_source_train[development_rows]

    cv_tree_random_state = int(rng.integers(0, 2**32))
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=cv_tree_random_state,
    )
    folds = tuple((fit.astype(int), held_out.astype(int)) for fit, held_out in splitter.split(X, y))
    fold_original_rows = tuple(
        {
            "fit": development_rows[fit].astype(int).tolist(),
            "held_out": development_rows[held_out].astype(int).tolist(),
        }
        for fit, held_out in folds
    )
    mi_random_state = int(rng.integers(0, 2**32))
    return SeedContext(
        seed=seed,
        X=X,
        y=y,
        development_original_rows=development_rows,
        initial_train_rows=initial_train,
        initial_validation_rows=initial_validation,
        folds=folds,
        fold_original_rows=fold_original_rows,
        split_random_state=split_random_state,
        cv_tree_random_state=cv_tree_random_state,
        mi_random_state=mi_random_state,
    )


def _fit_fold_scores(
    X: np.ndarray,
    y: np.ndarray,
    folds: tuple[tuple[np.ndarray, np.ndarray], ...],
    subset: tuple[int, ...],
    random_state: int,
) -> tuple[float, tuple[float, ...]]:
    feature_indices = np.asarray(subset, dtype=int)
    fold_scores: list[float] = []
    for fit_rows, held_out_rows in folds:
        classifier = DecisionTreeClassifier(random_state=random_state)
        classifier.fit(X[fit_rows][:, feature_indices], y[fit_rows])
        fold_scores.append(float(classifier.score(X[held_out_rows][:, feature_indices], y[held_out_rows])))
    return float(np.mean(fold_scores)), tuple(fold_scores)


class CountingSubsetScorer:
    """One method-local cache around the frozen five-fold Decision-Tree scorer."""

    def __init__(self, context: SeedContext, *, n_jobs: int) -> None:
        self.context = context
        self.n_jobs = int(n_jobs)
        self.candidate_requests = 0
        self.cache_hits = 0
        self.classifier_fit_count = 0
        self.fit_elapsed_seconds = 0.0
        self._cache: dict[tuple[int, ...], tuple[float, tuple[float, ...]]] = {}

    def score_many(self, candidates: Sequence[Sequence[int]]) -> list[float]:
        canonical = [_canonical_subset(candidate, self.context.X.shape[1]) for candidate in candidates]
        self.candidate_requests += len(canonical)
        misses: list[tuple[int, ...]] = []
        seen_misses: set[tuple[int, ...]] = set()
        for subset in canonical:
            if subset in self._cache or subset in seen_misses:
                self.cache_hits += 1
            else:
                seen_misses.add(subset)
                misses.append(subset)

        started = time.perf_counter()
        if len(misses) == 1 or self.n_jobs == 1:
            computed = [
                _fit_fold_scores(
                    self.context.X,
                    self.context.y,
                    self.context.folds,
                    subset,
                    self.context.cv_tree_random_state,
                )
                for subset in misses
            ]
        else:
            computed = Parallel(n_jobs=self.n_jobs, prefer="processes")(
                delayed(_fit_fold_scores)(
                    self.context.X,
                    self.context.y,
                    self.context.folds,
                    subset,
                    self.context.cv_tree_random_state,
                )
                for subset in misses
            )
        self.fit_elapsed_seconds += time.perf_counter() - started
        for subset, result in zip(misses, computed):
            self._cache[subset] = result
        self.classifier_fit_count += len(misses) * len(self.context.folds)
        return [self._cache[subset][0] for subset in canonical]

    def score(self, subset: Sequence[int]) -> float:
        return self.score_many([subset])[0]

    def fold_scores(self, subset: Sequence[int]) -> tuple[float, ...]:
        canonical = _canonical_subset(subset, self.context.X.shape[1])
        if canonical not in self._cache:
            self.score(canonical)
        return self._cache[canonical][1]

    def counters(self) -> dict[str, Any]:
        return {
            "candidate_subset_requests": int(self.candidate_requests),
            "unique_scored_subsets": int(len(self._cache)),
            "scorer_cache_hits": int(self.cache_hits),
            "classifier_fit_count": int(self.classifier_fit_count),
            "cv_fits_per_unique_subset": int(len(self.context.folds)),
            "scorer_fit_elapsed_seconds": float(self.fit_elapsed_seconds),
        }


def _archive_item(
    source: str,
    subset: Sequence[int],
    score: float,
    *,
    order: int,
) -> dict[str, Any]:
    return {
        "order": int(order),
        "source": source,
        "clean_indices_0based": list(subset),
        "feature_count": int(len(subset)),
        "inner_cv_accuracy": float(score),
    }


def _result_payload(
    *,
    config: UnifiedConfig,
    context: SeedContext,
    feature_map: FeatureIndexMap,
    run_id: str,
    method: str,
    track: str,
    target_k: int | None,
    subset: Sequence[int],
    score: float,
    fold_scores: Sequence[float],
    counters: dict[str, Any],
    elapsed_seconds: float,
    mi_fit_count: int,
    termination: str,
    selection_path: list[dict[str, Any]],
    archive: list[dict[str, Any]],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    clean = _canonical_subset(subset, feature_map.clean_feature_count)
    original = feature_map.clean_indices_to_original_ids_1based(clean)
    if feature_map.original_ids_1based_to_clean_indices(original) != clean:
        raise AssertionError("clean/original feature coordinate cross-check failed")
    return {
        "artifact_schema_version": 1,
        "protocol_version": config.protocol_version,
        "config_hash": config.config_hash,
        "seed": int(context.seed),
        "run_id": run_id,
        "method": method,
        "comparison_track": track,
        "target_k": target_k,
        "selected_k": len(clean),
        "clean_indices_0based": list(clean),
        "original_feature_ids_1based": list(original),
        "selected_clean_indices": list(clean),
        "selected_original_feature_ids": list(original),
        "coordinate_crosscheck_passed": True,
        "inner_cv_accuracy": float(score),
        "inner_cv_fold_accuracies": [float(value) for value in fold_scores],
        "selection_cost": {
            **counters,
            "mi_fit_count": int(mi_fit_count),
            "total_elapsed_seconds": float(elapsed_seconds),
            "includes_initialization": True,
            "includes_candidate_review": True,
            "final_evaluation_fits_included": False,
        },
        "termination": termination,
        "selection_path": selection_path,
        "archive": archive,
        "source_test_read_during_selection": False,
        **(extra or {}),
    }


def _run_all_features(
    config: UnifiedConfig,
    context: SeedContext,
    feature_map: FeatureIndexMap,
) -> dict[str, Any]:
    scorer = CountingSubsetScorer(context, n_jobs=int(config.selection["n_jobs"]))
    started = time.perf_counter()
    subset = tuple(range(context.X.shape[1]))
    score = scorer.score(subset)
    elapsed = time.perf_counter() - started
    return _result_payload(
        config=config,
        context=context,
        feature_map=feature_map,
        run_id="all_features__reference",
        method="all_features",
        track="reference",
        target_k=len(subset),
        subset=subset,
        score=score,
        fold_scores=scorer.fold_scores(subset),
        counters=scorer.counters(),
        elapsed_seconds=elapsed,
        mi_fit_count=0,
        termination="all_clean_features",
        selection_path=[],
        archive=[_archive_item("all_features", subset, score, order=1)],
    )


def _compute_mi_ranking(context: SeedContext) -> tuple[tuple[int, ...], float]:
    started = time.perf_counter()
    relevance = mutual_info_classif(
        context.X,
        context.y,
        random_state=context.mi_random_state,
    )
    ranked = tuple(np.argsort(-relevance, kind="stable").astype(int).tolist())
    return ranked, time.perf_counter() - started


def _run_mi_topk(
    config: UnifiedConfig,
    context: SeedContext,
    feature_map: FeatureIndexMap,
    ranking: tuple[int, ...],
    mi_elapsed: float,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    candidates: list[tuple[int, ...]] = []
    for k in config.selection["fixed_k"]:
        subset = tuple(sorted(ranking[: int(k)]))
        candidates.append(subset)
        scorer = CountingSubsetScorer(context, n_jobs=int(config.selection["n_jobs"]))
        started = time.perf_counter()
        score = scorer.score(subset)
        elapsed = mi_elapsed + (time.perf_counter() - started)
        results.append(
            _result_payload(
                config=config,
                context=context,
                feature_map=feature_map,
                run_id=f"mi_topk__fixed__k{int(k)}",
                method="mi_topk",
                track="fixed_k",
                target_k=int(k),
                subset=subset,
                score=score,
                fold_scores=scorer.fold_scores(subset),
                counters=scorer.counters(),
                elapsed_seconds=elapsed,
                mi_fit_count=1,
                termination="fixed_k_reached",
                selection_path=[],
                archive=[_archive_item("mi_topk", subset, score, order=1)],
                extra={"mi_random_state": context.mi_random_state},
            )
        )

    scorer = CountingSubsetScorer(context, n_jobs=int(config.selection["n_jobs"]))
    started = time.perf_counter()
    scores = scorer.score_many(candidates)
    best_index = max(
        range(len(candidates)),
        key=lambda index: (scores[index], -len(candidates[index])),
    )
    elapsed = mi_elapsed + (time.perf_counter() - started)
    archive: list[dict[str, Any]] = []
    best_so_far = -np.inf
    for subset, score in zip(candidates, scores):
        if score > best_so_far:
            archive.append(_archive_item("mi_topk_k_grid", subset, score, order=len(archive) + 1))
            best_so_far = score
    selected = candidates[best_index]
    results.append(
        _result_payload(
            config=config,
            context=context,
            feature_map=feature_map,
            run_id="mi_topk__auto_grid",
            method="mi_topk",
            track="automatic_k",
            target_k=None,
            subset=selected,
            score=scores[best_index],
            fold_scores=scorer.fold_scores(selected),
            counters=scorer.counters(),
            elapsed_seconds=elapsed,
            mi_fit_count=1,
            termination="best_inner_cv_over_preregistered_k_grid",
            selection_path=[
                {
                    "candidate_k": len(subset),
                    "clean_indices_0based": list(subset),
                    "inner_cv_accuracy": float(score),
                }
                for subset, score in zip(candidates, scores)
            ],
            archive=archive,
            extra={
                "mi_random_state": context.mi_random_state,
                "automatic_k_candidates": [int(value) for value in config.selection["fixed_k"]],
            },
        )
    )
    return results


def _run_mi_ordered_accept(
    config: UnifiedConfig,
    context: SeedContext,
    feature_map: FeatureIndexMap,
    ranking: tuple[int, ...],
    mi_elapsed: float,
) -> dict[str, Any]:
    scorer = CountingSubsetScorer(context, n_jobs=int(config.selection["n_jobs"]))
    started = time.perf_counter()
    tolerance = float(config.selection["strict_improvement_tolerance"])
    selected: list[int] = []
    best_score = -np.inf
    path: list[dict[str, Any]] = []
    archive: list[dict[str, Any]] = []
    termination = "mi_order_exhausted"
    for rank, feature in enumerate(ranking, start=1):
        if len(selected) >= int(config.selection["k_max"]):
            termination = "k_max_reached"
            break
        candidate = tuple(sorted((*selected, feature)))
        score = scorer.score(candidate)
        accepted = score > best_score + tolerance
        path.append(
            {
                "rank": rank,
                "feature": int(feature),
                "candidate_k": len(candidate),
                "inner_cv_accuracy": float(score),
                "accepted": bool(accepted),
            }
        )
        if accepted:
            selected.append(feature)
            best_score = score
            archive.append(
                _archive_item("mi_ordered_strict_improvement", candidate, score, order=len(archive) + 1)
            )
    elapsed = mi_elapsed + (time.perf_counter() - started)
    subset = tuple(sorted(selected))
    return _result_payload(
        config=config,
        context=context,
        feature_map=feature_map,
        run_id="mi_ordered_accept__automatic",
        method="mi_ordered_accept",
        track="automatic_k",
        target_k=None,
        subset=subset,
        score=best_score,
        fold_scores=scorer.fold_scores(subset),
        counters=scorer.counters(),
        elapsed_seconds=elapsed,
        mi_fit_count=1,
        termination=termination,
        selection_path=path,
        archive=archive,
        extra={"mi_random_state": context.mi_random_state},
    )


def _choose_best_candidate(
    candidates: list[tuple[int, ...]],
    scores: list[float],
) -> tuple[tuple[int, ...], float]:
    best_index = max(range(len(candidates)), key=lambda index: scores[index])
    return candidates[best_index], float(scores[best_index])


def _forward_path(
    config: UnifiedConfig,
    context: SeedContext,
    feature_map: FeatureIndexMap,
) -> list[dict[str, Any]]:
    """Run one exact-K greedy path and emit fixed checkpoints plus native automatic stop."""

    scorer = CountingSubsetScorer(context, n_jobs=int(config.selection["n_jobs"]))
    started = time.perf_counter()
    tolerance = float(config.selection["strict_improvement_tolerance"])
    fixed_k = tuple(int(value) for value in config.selection["fixed_k"])
    selected: tuple[int, ...] = ()
    current_score = -np.inf
    remaining = set(range(context.X.shape[1]))
    path: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    auto_result: dict[str, Any] | None = None

    while len(selected) < int(config.selection["k_max"]):
        candidates = [tuple(sorted((*selected, feature))) for feature in sorted(remaining)]
        scores = scorer.score_many(candidates)
        chosen, chosen_score = _choose_best_candidate(candidates, scores)
        improvement = None if not np.isfinite(current_score) else chosen_score - current_score
        step = {
            "step": len(selected) + 1,
            "candidate_count": len(candidates),
            "chosen_clean_indices_0based": list(chosen),
            "chosen_inner_cv_accuracy": chosen_score,
            "improvement": improvement,
            "cumulative_cost": scorer.counters(),
        }
        path.append(step)

        if auto_result is None and np.isfinite(current_score) and chosen_score <= current_score + tolerance:
            elapsed = time.perf_counter() - started
            auto_result = _result_payload(
                config=config,
                context=context,
                feature_map=feature_map,
                run_id="forward_greedy__automatic",
                method="forward_greedy",
                track="automatic_k",
                target_k=None,
                subset=selected,
                score=current_score,
                fold_scores=scorer.fold_scores(selected),
                counters=scorer.counters(),
                elapsed_seconds=elapsed,
                mi_fit_count=0,
                termination="no_strict_addition_improvement",
                selection_path=list(path),
                archive=[
                    _archive_item(
                        "forward_strict_improvement",
                        entry["chosen_clean_indices_0based"],
                        entry["chosen_inner_cv_accuracy"],
                        order=index + 1,
                    )
                    for index, entry in enumerate(path[:-1])
                ],
            )

        selected = chosen
        current_score = chosen_score
        remaining = set(range(context.X.shape[1])) - set(selected)
        print(
            f"seed={context.seed} family=forward step={len(selected):>2} "
            f"candidates={len(candidates):>2} score={current_score:.6f}",
            flush=True,
        )
        if len(selected) in fixed_k:
            elapsed = time.perf_counter() - started
            results.append(
                _result_payload(
                    config=config,
                    context=context,
                    feature_map=feature_map,
                    run_id=f"forward_greedy__fixed__k{len(selected)}",
                    method="forward_greedy",
                    track="fixed_k",
                    target_k=len(selected),
                    subset=selected,
                    score=current_score,
                    fold_scores=scorer.fold_scores(selected),
                    counters=scorer.counters(),
                    elapsed_seconds=elapsed,
                    mi_fit_count=0,
                    termination="fixed_k_reached",
                    selection_path=list(path),
                    archive=[
                        _archive_item(
                            f"forward_fixed_k{len(selected)}",
                            selected,
                            current_score,
                            order=1,
                        )
                    ],
                )
            )

    if auto_result is None:
        auto_result = _result_payload(
            config=config,
            context=context,
            feature_map=feature_map,
            run_id="forward_greedy__automatic",
            method="forward_greedy",
            track="automatic_k",
            target_k=None,
            subset=selected,
            score=current_score,
            fold_scores=scorer.fold_scores(selected),
            counters=scorer.counters(),
            elapsed_seconds=time.perf_counter() - started,
            mi_fit_count=0,
            termination="k_max_reached_with_strict_improvements",
            selection_path=list(path),
            archive=[
                _archive_item(
                    "forward_strict_improvement",
                    entry["chosen_clean_indices_0based"],
                    entry["chosen_inner_cv_accuracy"],
                    order=index + 1,
                )
                for index, entry in enumerate(path)
            ],
        )
    results.append(auto_result)
    return results


def _forward_initialize(
    config: UnifiedConfig,
    context: SeedContext,
    scorer: CountingSubsetScorer,
    *,
    target_k: int,
    automatic: bool,
) -> tuple[tuple[int, ...], float, list[dict[str, Any]], list[dict[str, Any]], str]:
    tolerance = float(config.selection["strict_improvement_tolerance"])
    selected: tuple[int, ...] = ()
    current_score = -np.inf
    remaining = set(range(context.X.shape[1]))
    path: list[dict[str, Any]] = []
    archive: list[dict[str, Any]] = []
    termination = "fixed_k_reached"

    while len(selected) < target_k:
        candidates = [tuple(sorted((*selected, feature))) for feature in sorted(remaining)]
        scores = scorer.score_many(candidates)
        chosen, chosen_score = _choose_best_candidate(candidates, scores)
        improvement = None if not np.isfinite(current_score) else chosen_score - current_score
        path.append(
            {
                "phase": "forward_initialization",
                "step": len(selected) + 1,
                "candidate_count": len(candidates),
                "chosen_clean_indices_0based": list(chosen),
                "chosen_inner_cv_accuracy": chosen_score,
                "improvement": improvement,
                "cumulative_cost": scorer.counters(),
            }
        )
        if automatic and np.isfinite(current_score) and chosen_score <= current_score + tolerance:
            termination = "forward_no_strict_addition_improvement"
            break
        selected = chosen
        current_score = chosen_score
        remaining = set(range(context.X.shape[1])) - set(selected)
        if automatic:
            archive.append(
                _archive_item(
                    "forward_strict_improvement",
                    selected,
                    current_score,
                    order=len(archive) + 1,
                )
            )
        print(
            f"seed={context.seed} family=single_swap init_k={len(selected):>2} "
            f"target={target_k:>2} score={current_score:.6f}",
            flush=True,
        )

    if not automatic:
        archive = [_archive_item("forward_fixed_k_initialization", selected, current_score, order=1)]
    return selected, current_score, path, archive, termination


def _run_single_swap(
    config: UnifiedConfig,
    context: SeedContext,
    feature_map: FeatureIndexMap,
    *,
    target_k: int,
    automatic: bool,
) -> dict[str, Any]:
    scorer = CountingSubsetScorer(context, n_jobs=int(config.selection["n_jobs"]))
    started = time.perf_counter()
    selected, current_score, path, archive, init_termination = _forward_initialize(
        config,
        context,
        scorer,
        target_k=target_k,
        automatic=automatic,
    )
    tolerance = float(config.selection["strict_improvement_tolerance"])
    max_rounds = int(config.selection["max_swap_rounds"])
    termination = "single_swap_local_optimum"
    accepted_swaps = 0

    for round_index in range(1, max_rounds + 1):
        selected_set = set(selected)
        remaining = sorted(set(range(context.X.shape[1])) - selected_set)
        candidates = [
            tuple(sorted((selected_set - {removed}) | {added})) for removed in selected for added in remaining
        ]
        scores = scorer.score_many(candidates)
        chosen, chosen_score = _choose_best_candidate(candidates, scores)
        improvement = chosen_score - current_score
        accepted = chosen_score > current_score + tolerance
        path.append(
            {
                "phase": "single_swap",
                "round": round_index,
                "candidate_count": len(candidates),
                "chosen_clean_indices_0based": list(chosen),
                "chosen_inner_cv_accuracy": chosen_score,
                "improvement": improvement,
                "accepted": accepted,
                "cumulative_cost": scorer.counters(),
            }
        )
        print(
            f"seed={context.seed} family=single_swap k={len(selected):>2} "
            f"round={round_index:>2} candidates={len(candidates):>4} "
            f"score={chosen_score:.6f} accepted={accepted}",
            flush=True,
        )
        if not accepted:
            termination = "single_swap_local_optimum"
            break
        selected = chosen
        current_score = chosen_score
        accepted_swaps += 1
        archive.append(
            _archive_item(
                "accepted_best_single_swap",
                selected,
                current_score,
                order=len(archive) + 1,
            )
        )
    else:
        termination = "max_swap_rounds_reached_without_optimality_confirmation"

    track = "automatic_k" if automatic else "fixed_k"
    run_id = (
        "forward_greedy_single_swap__automatic"
        if automatic
        else f"forward_greedy_single_swap__fixed__k{target_k}"
    )
    return _result_payload(
        config=config,
        context=context,
        feature_map=feature_map,
        run_id=run_id,
        method="forward_greedy_single_swap",
        track=track,
        target_k=None if automatic else target_k,
        subset=selected,
        score=current_score,
        fold_scores=scorer.fold_scores(selected),
        counters=scorer.counters(),
        elapsed_seconds=time.perf_counter() - started,
        mi_fit_count=0,
        termination=termination,
        selection_path=path,
        archive=archive,
        extra={
            "initialization_termination": init_termination,
            "accepted_swap_count": accepted_swaps,
            "max_swap_rounds": max_rounds,
        },
    )


def _selection_flat_row(result: dict[str, Any], result_path: Path) -> dict[str, Any]:
    cost = result["selection_cost"]
    return {
        "seed": result["seed"],
        "run_id": result["run_id"],
        "method": result["method"],
        "comparison_track": result["comparison_track"],
        "target_k": "" if result["target_k"] is None else result["target_k"],
        "selected_k": result["selected_k"],
        "inner_cv_accuracy": result["inner_cv_accuracy"],
        "candidate_subset_requests": cost["candidate_subset_requests"],
        "unique_scored_subsets": cost["unique_scored_subsets"],
        "scorer_cache_hits": cost["scorer_cache_hits"],
        "classifier_fit_count": cost["classifier_fit_count"],
        "mi_fit_count": cost["mi_fit_count"],
        "total_elapsed_seconds": cost["total_elapsed_seconds"],
        "termination": result["termination"],
        "result_path": str(result_path),
    }


def _mean_sd(values: Sequence[float]) -> tuple[float, float]:
    return float(mean(values)), float(stdev(values)) if len(values) > 1 else 0.0


def summarize_selection(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = sorted({(row["run_id"], row["method"], row["comparison_track"], row["target_k"]) for row in rows})
    summary: list[dict[str, Any]] = []
    for run_id, method, track, target_k in keys:
        group = [row for row in rows if row["run_id"] == run_id]
        score_mean, score_sd = _mean_sd([float(row["inner_cv_accuracy"]) for row in group])
        k_mean, k_sd = _mean_sd([float(row["selected_k"]) for row in group])
        unique_mean, unique_sd = _mean_sd([float(row["unique_scored_subsets"]) for row in group])
        fits_mean, fits_sd = _mean_sd([float(row["classifier_fit_count"]) for row in group])
        elapsed_mean, elapsed_sd = _mean_sd([float(row["total_elapsed_seconds"]) for row in group])
        summary.append(
            {
                "run_id": run_id,
                "method": method,
                "comparison_track": track,
                "target_k": target_k,
                "seeds": len(group),
                "selected_k_mean": k_mean,
                "selected_k_sd": k_sd,
                "inner_cv_accuracy_mean": score_mean,
                "inner_cv_accuracy_sd": score_sd,
                "unique_scored_subsets_mean": unique_mean,
                "unique_scored_subsets_sd": unique_sd,
                "classifier_fit_count_mean": fits_mean,
                "classifier_fit_count_sd": fits_sd,
                "total_elapsed_seconds_mean": elapsed_mean,
                "total_elapsed_seconds_sd": elapsed_sd,
                "nonlocal_termination_count": sum(
                    "without_optimality_confirmation" in str(row["termination"]) for row in group
                ),
            }
        )
    return summary


def paired_selection_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id_seed = {(row["run_id"], int(row["seed"])): row for row in rows}
    comparisons: list[tuple[str, str]] = []
    fixed_k = sorted(
        {
            int(row["target_k"])
            for row in rows
            if row["comparison_track"] == "fixed_k" and row["target_k"] != ""
        }
    )
    for k in fixed_k:
        baseline = f"mi_topk__fixed__k{k}"
        for contender in (
            f"forward_greedy__fixed__k{k}",
            f"forward_greedy_single_swap__fixed__k{k}",
        ):
            comparisons.append((contender, baseline))
    for contender in (
        "mi_ordered_accept__automatic",
        "forward_greedy__automatic",
        "forward_greedy_single_swap__automatic",
    ):
        comparisons.append((contender, "mi_topk__auto_grid"))

    output: list[dict[str, Any]] = []
    seeds = sorted({int(row["seed"]) for row in rows})
    for contender, baseline in comparisons:
        deltas = [
            float(by_id_seed[(contender, seed)]["inner_cv_accuracy"])
            - float(by_id_seed[(baseline, seed)]["inner_cv_accuracy"])
            for seed in seeds
        ]
        delta_mean, delta_sd = _mean_sd(deltas)
        output.append(
            {
                "contender_run_id": contender,
                "baseline_run_id": baseline,
                "seeds": len(deltas),
                "inner_cv_accuracy_delta_mean": delta_mean,
                "inner_cv_accuracy_delta_sd": delta_sd,
                "wins": sum(delta > 0.0 for delta in deltas),
                "ties": sum(delta == 0.0 for delta in deltas),
                "losses": sum(delta < 0.0 for delta in deltas),
            }
        )
    return output


def _root_manifest(config: UnifiedConfig) -> dict[str, Any]:
    return {
        "artifact_schema_version": 1,
        "protocol_version": config.protocol_version,
        "config_hash": config.config_hash,
        "config_raw_sha256": _sha256_bytes(config.raw_text.encode("utf-8")),
        "config_path": str(config.path),
    }


def _prepare_root(config: UnifiedConfig) -> None:
    root = config.output_root
    marker = root / "experiment-root.json"
    expected = _root_manifest(config)
    if marker.is_file():
        if _read_json(marker) != expected:
            raise ValueError("output root is reserved for a different config/protocol identity")
        return
    if root.is_dir() and any(root.iterdir()):
        raise ValueError("refusing to write into a non-empty unversioned experiment root")
    root.mkdir(parents=True, exist_ok=True)
    _write_json(marker, expected)
    (root / "config.toml").write_text(config.raw_text, encoding="utf-8")


def _validate_train_metadata(config: UnifiedConfig, X: np.ndarray, metadata: dict[str, Any]) -> None:
    expected = config.dataset
    checks = {
        "expected_source_train_rows": X.shape[0],
        "expected_original_features": metadata["original_feature_count"],
        "expected_clean_features": X.shape[1],
    }
    for field, actual in checks.items():
        if int(expected[field]) != int(actual):
            raise ValueError(f"{field} mismatch: expected {expected[field]}, got {actual}")
    actual_hash = metadata["source_files"]["train"]["sha256"]
    if actual_hash != expected["source_train_sha256"]:
        raise ValueError("source-train SHA-256 does not match the preregistered config")


def _seed_context_payload(
    config: UnifiedConfig,
    context: SeedContext,
    metadata: dict[str, Any],
    setup_elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "artifact_schema_version": 1,
        "protocol_version": config.protocol_version,
        "config_hash": config.config_hash,
        "seed": context.seed,
        "source_test_loaded": False,
        "source_train": metadata,
        "split_random_state": context.split_random_state,
        "cv_tree_random_state": context.cv_tree_random_state,
        "mi_random_state": context.mi_random_state,
        "initial_train_original_rows": context.initial_train_rows.tolist(),
        "initial_validation_original_rows": context.initial_validation_rows.tolist(),
        "development_row_order": context.development_original_rows.tolist(),
        "folds": list(context.fold_original_rows),
        "shared_setup_elapsed_seconds": setup_elapsed_seconds,
    }


def _run_seed_selection(
    config: UnifiedConfig,
    X: np.ndarray,
    y: np.ndarray,
    metadata: dict[str, Any],
    seed: int,
) -> list[dict[str, Any]]:
    seed_root = config.output_root / "selection" / f"seed-{seed}"
    complete_path = seed_root / "complete.json"
    if config.resume and complete_path.is_file():
        complete = _read_json(complete_path)
        if complete.get("config_hash") != config.config_hash:
            raise ValueError(f"seed {seed} completion marker has a config mismatch")
        print(f"seed={seed} selection resume: complete", flush=True)
        return [_read_json(config.output_root / entry) for entry in complete["result_paths"]]
    if seed_root.exists() and any(seed_root.iterdir()):
        raise ValueError(
            f"partial seed directory exists without a completion marker: {seed_root}; "
            "use a new protocol/output root instead of overwriting"
        )

    setup_started = time.perf_counter()
    context = build_seed_context(
        X,
        y,
        seed=seed,
        validation_fraction=float(config.dataset["validation_fraction"]),
        n_splits=int(config.dataset["inner_cv_folds"]),
    )
    setup_elapsed = time.perf_counter() - setup_started
    feature_map = FeatureIndexMap.from_metadata(metadata)
    _write_json(
        seed_root / "context.json",
        _seed_context_payload(config, context, metadata, setup_elapsed),
    )

    results: list[dict[str, Any]] = [_run_all_features(config, context, feature_map)]
    ranking, mi_elapsed = _compute_mi_ranking(context)
    results.extend(_run_mi_topk(config, context, feature_map, ranking, mi_elapsed))
    results.append(_run_mi_ordered_accept(config, context, feature_map, ranking, mi_elapsed))
    results.extend(_forward_path(config, context, feature_map))
    for k in config.selection["fixed_k"]:
        results.append(
            _run_single_swap(
                config,
                context,
                feature_map,
                target_k=int(k),
                automatic=False,
            )
        )
    results.append(
        _run_single_swap(
            config,
            context,
            feature_map,
            target_k=int(config.selection["k_max"]),
            automatic=True,
        )
    )

    result_paths: list[str] = []
    for result in results:
        result_path = seed_root / result["run_id"] / "result.json"
        _write_json(result_path, result)
        result_paths.append(str(result_path.relative_to(config.output_root)))
    _write_json(
        complete_path,
        {
            "artifact_schema_version": 1,
            "protocol_version": config.protocol_version,
            "config_hash": config.config_hash,
            "seed": seed,
            "result_count": len(results),
            "result_paths": result_paths,
        },
    )
    return results


def run_selection(config: UnifiedConfig) -> list[dict[str, Any]]:
    _prepare_root(config)
    completion_path = config.output_root / "selection-complete.json"
    if config.resume and completion_path.is_file():
        completion = _read_json(completion_path)
        if completion.get("config_hash") != config.config_hash:
            raise ValueError("selection completion marker has a config mismatch")
        results: list[dict[str, Any]] = []
        for seed in config.dataset["seeds"]:
            seed_completion = _read_json(
                config.output_root / "selection" / f"seed-{int(seed)}" / "complete.json"
            )
            results.extend(_read_json(config.output_root / path) for path in seed_completion["result_paths"])
        print("selection resume: complete", flush=True)
        return results
    X, y, _, metadata = load_radar_ship_source_train(
        str(config.dataset["data_dir"]),
        str(config.dataset["version"]),
    )
    _validate_train_metadata(config, X, metadata)
    all_results: list[dict[str, Any]] = []
    for seed in config.dataset["seeds"]:
        all_results.extend(_run_seed_selection(config, X, y, metadata, int(seed)))

    rows: list[dict[str, Any]] = []
    for result in all_results:
        path = config.output_root / "selection" / f"seed-{result['seed']}" / result["run_id"] / "result.json"
        rows.append(_selection_flat_row(result, path))
    _write_csv(config.output_root / "selection" / "runs.csv", rows)
    _write_csv(
        config.output_root / "selection" / "summary.csv",
        summarize_selection(rows),
    )
    _write_csv(
        config.output_root / "selection" / "paired_deltas.csv",
        paired_selection_deltas(rows),
    )
    _write_json(
        config.output_root / "dfs_status.json",
        {
            "status": "unavailable / protocol provenance incomplete",
            "eligible_for_numeric_ranking": False,
            "blocking_other_baselines": False,
            "reason": (
                "No executable DFS training code, dependency environment, licensed source/commit, "
                "checkpoint-selection provenance, or protocol-generated seed trajectories are present."
            ),
            "oracle_diagnostic_scripts_used": False,
        },
    )
    _write_json(
        config.output_root / "selection-complete.json",
        {
            **_root_manifest(config),
            "all_seeds_complete": True,
            "seed_count": len(config.dataset["seeds"]),
            "result_count": len(all_results),
            "source_test_loaded": False,
        },
    )
    return all_results


def _binary_metrics(
    y_true: np.ndarray,
    prediction: np.ndarray,
    probability: np.ndarray,
    classes: np.ndarray,
) -> dict[str, Any]:
    positive_label = classes[-1]
    binary_target = (y_true == positive_label).astype(int)
    recalls = recall_score(
        y_true,
        prediction,
        labels=classes,
        average=None,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "macro_f1": float(f1_score(y_true, prediction, average="macro", zero_division=0)),
        "per_class_recall": {str(int(label)): float(value) for label, value in zip(classes, recalls)},
        "roc_auc": float(roc_auc_score(binary_target, probability)),
        "confusion_matrix": confusion_matrix(y_true, prediction, labels=classes).astype(int).tolist(),
        "class_order": [int(value) for value in classes],
        "positive_label": int(positive_label),
    }


def _evaluate_subset(
    config: UnifiedConfig,
    seed: int,
    subset: tuple[int, ...],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    indices = np.asarray(subset, dtype=int)
    final = config.final_classifier
    lr = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logistic_regression",
                LogisticRegression(
                    C=float(final["C"]),
                    solver=str(final["solver"]),
                    max_iter=int(final["max_iter"]),
                    class_weight=str(final["class_weight"]),
                    random_state=seed,
                ),
            ),
        ]
    )
    lr.fit(X_train[:, indices], y_train)
    lr_prediction = lr.predict(X_test[:, indices])
    lr_probability_matrix = lr.predict_proba(X_test[:, indices])
    lr_classes = lr.named_steps["logistic_regression"].classes_
    lr_positive_column = int(np.flatnonzero(lr_classes == lr_classes[-1])[0])
    lr_metrics = _binary_metrics(
        y_test,
        lr_prediction,
        lr_probability_matrix[:, lr_positive_column],
        lr_classes,
    )

    tree = DecisionTreeClassifier(random_state=seed)
    tree.fit(X_train[:, indices], y_train)
    tree_prediction = tree.predict(X_test[:, indices])
    tree_probability_matrix = tree.predict_proba(X_test[:, indices])
    tree_classes = tree.classes_
    tree_positive_column = int(np.flatnonzero(tree_classes == tree_classes[-1])[0])
    tree_metrics = _binary_metrics(
        y_test,
        tree_prediction,
        tree_probability_matrix[:, tree_positive_column],
        tree_classes,
    )
    return {
        "primary_logistic_regression": lr_metrics,
        "secondary_decision_tree": tree_metrics,
    }, time.perf_counter() - started


def _final_flat_row(result: dict[str, Any], path: Path) -> dict[str, Any]:
    lr = result["final_metrics"]["primary_logistic_regression"]
    tree = result["final_metrics"]["secondary_decision_tree"]
    return {
        "seed": result["seed"],
        "run_id": result["run_id"],
        "method": result["method"],
        "comparison_track": result["comparison_track"],
        "target_k": "" if result["target_k"] is None else result["target_k"],
        "selected_k": result["selected_k"],
        "inner_cv_accuracy": result["inner_cv_accuracy"],
        "lr_accuracy": lr["accuracy"],
        "lr_balanced_accuracy": lr["balanced_accuracy"],
        "lr_macro_f1": lr["macro_f1"],
        "lr_recall_class_-1": lr["per_class_recall"].get("-1"),
        "lr_recall_class_1": lr["per_class_recall"].get("1"),
        "lr_roc_auc": lr["roc_auc"],
        "dt_accuracy": tree["accuracy"],
        "dt_balanced_accuracy": tree["balanced_accuracy"],
        "dt_macro_f1": tree["macro_f1"],
        "dt_recall_class_-1": tree["per_class_recall"].get("-1"),
        "dt_recall_class_1": tree["per_class_recall"].get("1"),
        "dt_roc_auc": tree["roc_auc"],
        "final_classifier_fit_count": result["final_classifier_fit_count"],
        "final_evaluation_elapsed_seconds": result["final_evaluation_elapsed_seconds"],
        "result_path": str(path),
    }


def summarize_final(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_fields = (
        "selected_k",
        "inner_cv_accuracy",
        "lr_accuracy",
        "lr_balanced_accuracy",
        "lr_macro_f1",
        "lr_roc_auc",
        "dt_accuracy",
        "dt_balanced_accuracy",
        "dt_macro_f1",
        "dt_roc_auc",
        "final_evaluation_elapsed_seconds",
    )
    output: list[dict[str, Any]] = []
    for run_id in sorted({row["run_id"] for row in rows}):
        group = [row for row in rows if row["run_id"] == run_id]
        item: dict[str, Any] = {
            "run_id": run_id,
            "method": group[0]["method"],
            "comparison_track": group[0]["comparison_track"],
            "target_k": group[0]["target_k"],
            "seeds": len(group),
        }
        for field in metric_fields:
            field_mean, field_sd = _mean_sd([float(row[field]) for row in group])
            item[f"{field}_mean"] = field_mean
            item[f"{field}_sd"] = field_sd
        output.append(item)
    return output


def run_final_evaluation(config: UnifiedConfig) -> list[dict[str, Any]]:
    _prepare_root(config)
    completion = config.output_root / "selection-complete.json"
    if not completion.is_file():
        raise FileNotFoundError("selection must complete for all seeds before final evaluation")
    if _read_json(completion).get("config_hash") != config.config_hash:
        raise ValueError("selection completion marker has a config mismatch")

    final_completion = config.output_root / "final-evaluation-complete.json"
    if config.resume and final_completion.is_file():
        marker = _read_json(final_completion)
        if marker.get("config_hash") != config.config_hash:
            raise ValueError("final completion marker has a config mismatch")
        print("final evaluation resume: complete", flush=True)
        return [_read_json(config.output_root / path) for path in marker["result_paths"]]
    final_root = config.output_root / "final"
    if final_root.is_dir() and any(final_root.rglob("metrics.json")):
        raise ValueError(
            "partial final-evaluation artifacts exist without a completion marker; "
            "use a new protocol/output root instead of overwriting"
        )

    X_train, y_train, _, train_metadata = load_radar_ship_source_train(
        str(config.dataset["data_dir"]),
        str(config.dataset["version"]),
    )
    _validate_train_metadata(config, X_train, train_metadata)
    X_test, y_test, test_metadata = load_radar_ship_source_test(
        str(config.dataset["data_dir"]),
        str(config.dataset["version"]),
        train_metadata,
    )
    if test_metadata["sha256"] != config.dataset["source_test_sha256"]:
        raise ValueError("source-test SHA-256 does not match the preregistered config")
    if int(test_metadata["rows"]) != int(config.dataset["expected_source_test_rows"]):
        raise ValueError("source-test row count does not match the preregistered config")
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("clean train/test feature dimensions differ")

    selection_results: list[dict[str, Any]] = []
    for seed in config.dataset["seeds"]:
        complete = _read_json(config.output_root / "selection" / f"seed-{int(seed)}" / "complete.json")
        selection_results.extend(_read_json(config.output_root / path) for path in complete["result_paths"])

    feature_map = FeatureIndexMap.from_metadata(train_metadata)
    output: list[dict[str, Any]] = []
    result_paths: list[str] = []
    for selection in selection_results:
        subset = feature_map.validate_artifact_selection(selection)
        if list(subset) != selection["clean_indices_0based"]:
            raise ValueError("selection coordinate aliases disagree")
        metrics, elapsed = _evaluate_subset(
            config,
            int(selection["seed"]),
            subset,
            X_train,
            y_train,
            X_test,
            y_test,
        )
        payload = {
            "artifact_schema_version": 1,
            "protocol_version": config.protocol_version,
            "config_hash": config.config_hash,
            "seed": selection["seed"],
            "run_id": selection["run_id"],
            "method": selection["method"],
            "comparison_track": selection["comparison_track"],
            "target_k": selection["target_k"],
            "selected_k": selection["selected_k"],
            "clean_indices_0based": selection["clean_indices_0based"],
            "original_feature_ids_1based": selection["original_feature_ids_1based"],
            "coordinate_crosscheck_passed": True,
            "inner_cv_accuracy": selection["inner_cv_accuracy"],
            "source_test_role": "reused_source_test_diagnostic_only",
            "historical_source_test_already_observed": True,
            "eligible_as_fresh_independent_validation": False,
            "source_test_used_for_subset_or_hyperparameter_selection": False,
            "source_test_metadata": test_metadata,
            "final_classifier_fit_count": 2,
            "final_evaluation_elapsed_seconds": elapsed,
            "final_metrics": metrics,
        }
        result_path = (
            config.output_root / "final" / f"seed-{selection['seed']}" / selection["run_id"] / "metrics.json"
        )
        _write_json(result_path, payload)
        result_paths.append(str(result_path.relative_to(config.output_root)))
        output.append(payload)
        print(
            f"final seed={selection['seed']} run={selection['run_id']:<50} "
            f"k={selection['selected_k']:>2} "
            f"lr_bacc={metrics['primary_logistic_regression']['balanced_accuracy']:.6f} "
            f"dt_bacc={metrics['secondary_decision_tree']['balanced_accuracy']:.6f}",
            flush=True,
        )

    rows = [
        _final_flat_row(
            result,
            config.output_root / "final" / f"seed-{result['seed']}" / result["run_id"] / "metrics.json",
        )
        for result in output
    ]
    _write_csv(config.output_root / "final" / "runs.csv", rows)
    _write_csv(config.output_root / "final" / "summary.csv", summarize_final(rows))
    _write_json(
        final_completion,
        {
            **_root_manifest(config),
            "all_selections_completed_before_test_evaluation": True,
            "test_file_loaded_once_in_evaluation_process": True,
            "test_used_for_selection": False,
            "source_test_role": "reused_source_test_diagnostic_only",
            "result_count": len(output),
            "result_paths": result_paths,
        },
    )
    return output


def write_environment_manifest(config: UnifiedConfig) -> None:
    environment_path = config.output_root / "environment.json"
    if environment_path.is_file():
        return
    _write_json(
        environment_path,
        {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "git_commit": _git(["rev-parse", "HEAD"]),
            "git_branch": _git(["branch", "--show-current"]),
            "git_dirty": bool(_git(["status", "--porcelain"])),
            "git_status_porcelain": _git(["status", "--porcelain"]),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/v16n/unified_strong_baselines.toml",
    )
    parser.add_argument(
        "--stage",
        choices=("select", "evaluate", "all"),
        default="select",
        help="'select' never loads source-test; 'evaluate' requires complete frozen selections.",
    )
    args = parser.parse_args()
    config = load_unified_config(args.config)
    _prepare_root(config)
    write_environment_manifest(config)
    if args.stage in {"select", "all"}:
        run_selection(config)
    if args.stage in {"evaluate", "all"}:
        run_final_evaluation(config)


if __name__ == "__main__":
    main()
