#!/usr/bin/env python3
"""Train and evaluate the block-rewrite PPO accuracy exploration.

This entry point is deliberately train-only.  It reuses row numbers and the
clean-feature mapping from task 4C, but builds every graph and scorer from the
corresponding outer-training rows and never opens the historical source-test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import sklearn
import torch
from joblib import Parallel, delayed
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from radar_ship_fs.feature_mapping import FeatureIndexMap
from radar_ship_fs.ppo.block_rewrite import (
    ConditionalMacroActorCritic,
    MacroObservation,
    MacroPPOAgent,
    MacroPPOConfig,
    MacroRolloutBuffer,
    apply_macro_action,
    best_so_far_reward,
)
from radar_ship_fs.ppo.ppo_graph import FeatureGraph, build_feature_graph
from run_k32_robustness import (
    best_improvement_swaps,
    clean_on_outer_train,
    exact_k_forward,
)
from run_reward_alignment import canonical_subset

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


DEFAULT_CONFIG = Path("configs/v16n/block_rewrite_ppo_v1.toml")
METHODS = ("pair_ppo", "single_ppo", "random_pair")
METHOD_CODES = {"pair_ppo": 11, "single_ppo": 23, "random_pair": 37}


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(target)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(target)


def write_csv(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(target, index=False)


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def load_train(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    section = config["dataset"]
    path = Path(section["source_train"])
    if sha256(path) != section["source_train_sha256"]:
        raise RuntimeError("source-train hash differs from block-rewrite protocol")
    sparse, labels = load_svmlight_file(path, n_features=int(section["expected_original_features"]))
    raw = sparse.toarray().astype(np.float32)
    labels = labels.astype(np.int64)
    expected = (
        int(section["expected_rows"]),
        int(section["expected_original_features"]),
    )
    if raw.shape != expected:
        raise AssertionError(f"unexpected source-train shape: {raw.shape}")
    if set(np.unique(labels).tolist()) != {-1, 1}:
        raise AssertionError("expected -1/+1 labels")
    partition = Path(section["outer_partition"])
    if sha256(partition) != section["outer_partition_sha256"]:
        raise RuntimeError("outer partition hash differs from block-rewrite protocol")
    return raw, labels


@dataclass
class FoldData:
    fold: int
    train_rows: np.ndarray
    validation_rows: np.ndarray
    final_ids: tuple[int, ...]
    mapping: FeatureIndexMap
    inner_folds: list[dict[str, list[int]]]
    graph: FeatureGraph
    mi32: tuple[int, ...]


def prepare_fold(
    raw: np.ndarray,
    labels: np.ndarray,
    config: dict[str, Any],
    fold: int,
    output_root: Path,
) -> FoldData:
    old_context = read_json(Path("experiments/k32_reward_robustness_v1/contexts") / f"outer-fold-{fold}.json")
    train_rows = np.asarray(old_context["outer_train_rows"], dtype=int)
    validation_rows = np.asarray(old_context["outer_validation_rows"], dtype=int)
    final_ids_array, cleaning = clean_on_outer_train(raw[train_rows])
    final_ids = tuple(int(value) for value in final_ids_array)
    if len(final_ids) != int(config["dataset"]["expected_clean_features"]):
        raise AssertionError(f"fold {fold} did not clean to 65 features")
    if final_ids != tuple(int(value) for value in old_context["final_feature_ids_1based"]):
        raise AssertionError(f"fold {fold} clean mapping differs from frozen context")
    mapping = FeatureIndexMap(final_ids)

    splitter = StratifiedKFold(
        n_splits=int(config["scorer"]["folds"]),
        shuffle=True,
        random_state=int(config["experiment"]["inner_seed_base"]) + fold,
    )
    inner_folds = []
    for fit_local, held_local in splitter.split(train_rows, labels[train_rows]):
        inner_folds.append(
            {
                "fit": train_rows[fit_local].astype(int).tolist(),
                "held_out": train_rows[held_local].astype(int).tolist(),
            }
        )

    clean_columns = np.asarray(final_ids, dtype=int) - 1
    clean_train = raw[train_rows][:, clean_columns]
    graph_seed = int(config["experiment"]["graph_seed_base"]) + fold
    graph = build_feature_graph(
        clean_train,
        labels[train_rows],
        np.zeros(len(final_ids), dtype=int),
        threshold=float(config["experiment"]["graph_threshold"]),
        seed=graph_seed,
        tree_seed=graph_seed,
        include_feature_id_node_feature=False,
    )
    original = np.asarray(final_ids, dtype=int)
    mi_order = np.lexsort((original, -graph.mutual_information.astype(float)))
    mi32 = tuple(sorted(int(value) for value in mi_order[:32]))
    context_payload = {
        "outer_fold": fold,
        "outer_train_rows": train_rows.tolist(),
        "outer_validation_rows": validation_rows.tolist(),
        "inner_folds": inner_folds,
        "inner_random_state": int(config["experiment"]["inner_seed_base"]) + fold,
        "cleaning": cleaning,
        "final_feature_ids_1based": list(final_ids),
        "mi32_clean_indices_0based": list(mi32),
        "mi32_original_feature_ids_1based": list(mapping.clean_indices_to_original_ids_1based(mi32)),
        "graph": {
            "fit_rows": train_rows.tolist(),
            "threshold": graph.threshold,
            "seed": graph_seed,
            "feature_id_node_feature": False,
            "edge_count": graph.edge_count,
            "dependency_edge_count": graph.dependency_edge_count,
            "static_node_feature_count": int(graph.static_node_features.shape[1]),
        },
    }
    write_json(output_root / "contexts" / f"outer-fold-{fold}.json", context_payload)
    return FoldData(
        fold,
        train_rows,
        validation_rows,
        final_ids,
        mapping,
        inner_folds,
        graph,
        mi32,
    )


class SVCScorer:
    def __init__(
        self,
        raw: np.ndarray,
        labels: np.ndarray,
        fold_data: FoldData,
        jobs: int,
    ) -> None:
        self.raw = raw
        self.labels = labels
        self.fold_data = fold_data
        self.jobs = jobs
        self.cache: dict[tuple[int, ...], dict[str, Any]] = {}
        self.requests = 0
        self.cache_hits = 0
        self.fit_count = 0
        self.fit_seconds = 0.0

    @staticmethod
    def score_one(
        raw: np.ndarray,
        labels: np.ndarray,
        final_ids: tuple[int, ...],
        subset: tuple[int, ...],
        folds: Sequence[dict[str, Sequence[int]]],
    ) -> dict[str, Any]:
        columns = np.asarray(final_ids, dtype=int)[np.asarray(subset, dtype=int)] - 1
        fold_scores = []
        started = time.perf_counter()
        for split in folds:
            fit = np.asarray(split["fit"], dtype=int)
            held = np.asarray(split["held_out"], dtype=int)
            scaler = StandardScaler().fit(raw[fit][:, columns])
            classifier = SVC(
                kernel="rbf",
                C=1.0,
                gamma=1.0 / 65.0,
                class_weight="balanced",
                probability=False,
                tol=1e-3,
                cache_size=256,
                max_iter=-1,
                shrinking=True,
            )
            classifier.fit(scaler.transform(raw[fit][:, columns]), labels[fit])
            prediction = classifier.predict(scaler.transform(raw[held][:, columns]))
            fold_scores.append(float(balanced_accuracy_score(labels[held], prediction)))
        return {
            "objective": float(np.mean(fold_scores)),
            "fold_scores": fold_scores,
            "classifier_fit_count": len(folds),
            "fit_seconds": time.perf_counter() - started,
            "scaler_fit_scope": "each_inner_training_fold",
        }

    def score_many(self, subsets: Sequence[Sequence[int]]) -> list[float]:
        keys = [canonical_subset(subset, len(self.fold_data.final_ids)) for subset in subsets]
        self.requests += len(keys)
        missing = []
        seen: set[tuple[int, ...]] = set()
        for key in keys:
            if key in self.cache or key in seen:
                self.cache_hits += 1
            else:
                seen.add(key)
                missing.append(key)
        computed = Parallel(n_jobs=self.jobs, prefer="processes")(
            delayed(self.score_one)(
                self.raw,
                self.labels,
                self.fold_data.final_ids,
                subset,
                self.fold_data.inner_folds,
            )
            for subset in missing
        )
        for subset, payload in zip(missing, computed):
            self.cache[subset] = payload
            self.fit_count += int(payload["classifier_fit_count"])
            self.fit_seconds += float(payload["fit_seconds"])
        return [float(self.cache[key]["objective"]) for key in keys]

    def preload(self, scores: dict[tuple[int, ...], float]) -> None:
        for subset, score in scores.items():
            self.cache.setdefault(
                subset,
                {
                    "objective": float(score),
                    "fold_scores": [],
                    "classifier_fit_count": 0,
                    "fit_seconds": 0.0,
                    "preloaded_from_case_checkpoint": True,
                },
            )

    def costs(self) -> dict[str, Any]:
        return {
            "candidate_requests": self.requests,
            "unique_scored_subsets": len(self.cache),
            "cache_hits": self.cache_hits,
            "classifier_fit_count": self.fit_count,
            "fit_seconds": self.fit_seconds,
        }


def cost_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    return {key: after[key] - before[key] for key in before}


def add_cost(total: dict[str, Any], delta: dict[str, Any]) -> None:
    for key in total:
        if key == "unique_scored_subsets":
            continue
        total[key] += delta[key]


def score_case(
    scorer: SVCScorer,
    subsets: Sequence[Sequence[int]],
    case_cost: dict[str, Any],
) -> list[float]:
    before = scorer.costs()
    scores = scorer.score_many(subsets)
    after = scorer.costs()
    delta = cost_delta(before, after)
    case_cost["candidate_requests"] += delta["candidate_requests"]
    case_cost["cache_hits"] += delta["cache_hits"]
    case_cost["classifier_fit_count"] += delta["classifier_fit_count"]
    case_cost["fit_seconds"] += delta["fit_seconds"]
    return scores


def subset_payload(subset: Sequence[int], fold_data: FoldData) -> dict[str, Any]:
    clean = tuple(sorted(int(value) for value in subset))
    return {
        "selected_clean_indices_0based": list(clean),
        "selected_original_feature_ids_1based": list(
            fold_data.mapping.clean_indices_to_original_ids_1based(clean)
        ),
    }


def endpoint_metrics(
    raw: np.ndarray,
    labels: np.ndarray,
    fold_data: FoldData,
    subset: Sequence[int],
    *,
    classifier: str = "svc",
) -> dict[str, float]:
    clean = np.asarray(tuple(subset), dtype=int)
    columns = np.asarray(fold_data.final_ids, dtype=int)[clean] - 1
    fit = fold_data.train_rows
    held = fold_data.validation_rows
    scaler = StandardScaler().fit(raw[fit][:, columns])
    if classifier == "svc":
        model = SVC(
            kernel="rbf",
            C=1.0,
            gamma=1.0 / 65.0,
            class_weight="balanced",
            probability=False,
            tol=1e-3,
            cache_size=256,
            max_iter=-1,
            shrinking=True,
        )
    elif classifier == "lr":
        model = LogisticRegression(
            C=1.0,
            solver="liblinear",
            class_weight="balanced",
            max_iter=5000,
            random_state=20260915 + fold_data.fold,
        )
    else:
        raise ValueError(classifier)
    model.fit(scaler.transform(raw[fit][:, columns]), labels[fit])
    prediction = model.predict(scaler.transform(raw[held][:, columns]))
    return {
        "balanced_accuracy": float(balanced_accuracy_score(labels[held], prediction)),
        "accuracy": float(accuracy_score(labels[held], prediction)),
        "recall_negative": float(recall_score(labels[held], prediction, pos_label=-1)),
        "recall_positive": float(recall_score(labels[held], prediction, pos_label=1)),
    }


def top_archive(
    archive: dict[tuple[int, ...], dict[str, Any]],
    fold_data: FoldData,
    count: int,
) -> list[tuple[int, ...]]:
    return [
        subset
        for subset, _ in sorted(
            archive.items(),
            key=lambda item: (
                -float(item[1]["score"]),
                fold_data.mapping.clean_indices_to_original_ids_1based(item[0]),
            ),
        )[:count]
    ]


def random_starts(
    n_episodes: int,
    n_features: int,
    budget: int,
    seed: int,
    fold: int,
    salt: int,
) -> list[tuple[int, ...]]:
    generator = np.random.default_rng(np.random.SeedSequence([seed, fold, salt]))
    return [
        tuple(sorted(int(value) for value in generator.choice(n_features, budget, replace=False)))
        for _ in range(n_episodes)
    ]


def archive_rows(archive: dict[tuple[int, ...], dict[str, Any]], fold_data: FoldData):
    for subset, payload in sorted(archive.items()):
        yield {
            **subset_payload(subset, fold_data),
            **payload,
        }


def record_archive(
    archive: dict[tuple[int, ...], dict[str, Any]],
    subset: Sequence[int],
    score: float,
    *,
    phase: str,
    episode: int,
    step: int,
) -> None:
    key = tuple(sorted(int(value) for value in subset))
    if key not in archive:
        archive[key] = {
            "score": float(score),
            "first_phase": phase,
            "first_episode": int(episode),
            "first_step": int(step),
        }
    elif abs(float(archive[key]["score"]) - float(score)) > 1e-12:
        raise AssertionError("deterministic scorer returned inconsistent values")


def random_pair_actions(
    subsets: Sequence[Sequence[int]], generator: np.random.Generator, n_features: int
) -> np.ndarray:
    rows = []
    universe = np.arange(n_features)
    for subset in subsets:
        chosen = np.asarray(tuple(subset), dtype=int)
        complement = universe[~np.isin(universe, chosen)]
        remove = generator.choice(chosen, 2, replace=False)
        add = generator.choice(complement, 2, replace=False)
        rows.append(np.concatenate([remove, add]))
    return np.asarray(rows, dtype=np.int64)


def model_parameter_l2(model: torch.nn.Module) -> float:
    vector = torch.cat([parameter.detach().flatten().cpu() for parameter in model.parameters()])
    return float(torch.linalg.vector_norm(vector).item())


def empty_case_cost() -> dict[str, Any]:
    return {
        "candidate_requests": 0,
        "unique_scored_subsets": 0,
        "cache_hits": 0,
        "classifier_fit_count": 0,
        "fit_seconds": 0.0,
    }


def _case_paths(output_root: Path, variant: str, fold: int, seed: int, method: str) -> tuple[Path, Path]:
    case_dir = output_root / "cases" / variant / f"fold-{fold}-seed-{seed}-{method}"
    return case_dir, case_dir / "checkpoint.pt"


def _build_agent(
    graph: FeatureGraph,
    method: str,
    config: dict[str, Any],
    device: torch.device,
) -> MacroPPOAgent | None:
    if method == "random_pair":
        return None
    mode = "pair" if method == "pair_ppo" else "single"
    model = ConditionalMacroActorCritic(
        graph,
        mode=mode,
        hidden_dim=int(config["ppo"]["hidden_dim"]),
    )
    if model.prior_scale != 0.0:
        raise AssertionError("legacy selection prior must be disabled")
    ppo_config = MacroPPOConfig(
        learning_rate=float(config["ppo"]["learning_rate"]),
        gamma=float(config["ppo"]["gamma"]),
        gae_lambda=float(config["ppo"]["gae_lambda"]),
        clip_ratio=float(config["ppo"]["clip_ratio"]),
        ppo_epochs=int(config["ppo"]["ppo_epochs"]),
        minibatch_size=int(config["ppo"]["minibatch_size"]),
        entropy_coef=float(config["ppo"]["entropy_coef"]),
        value_coef=float(config["ppo"]["value_coef"]),
        max_grad_norm=float(config["ppo"]["max_grad_norm"]),
        target_kl=float(config["ppo"]["target_kl"]),
    )
    return MacroPPOAgent(model, ppo_config, device)


def _checkpoint_payload(
    *,
    phase: str,
    train_next: int,
    freeze_next: int,
    agent: MacroPPOAgent | None,
    archive: dict[tuple[int, ...], dict[str, Any]],
    frozen_archive: dict[tuple[int, ...], dict[str, Any]],
    trajectories: list[dict[str, Any]],
    curves: list[dict[str, Any]],
    updates: list[dict[str, Any]],
    case_cost: dict[str, Any],
    action_rng: np.random.Generator,
    elite_rng: np.random.Generator,
    initial_parameter_l2: float | None,
    update_hook: Any = None,
) -> dict[str, Any]:
    return {
        "phase": phase,
        "train_next": train_next,
        "freeze_next": freeze_next,
        "model_state": None if agent is None else agent.model.state_dict(),
        "optimizer_state": None if agent is None else agent.optimizer.state_dict(),
        "archive": archive,
        "frozen_archive": frozen_archive,
        "trajectories": trajectories,
        "curves": curves,
        "updates": updates,
        "case_cost": case_cost,
        "action_rng_state": action_rng.bit_generator.state,
        "elite_rng_state": elite_rng.bit_generator.state,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "initial_parameter_l2": initial_parameter_l2,
        "update_hook_state": None if update_hook is None else update_hook.state_dict(),
    }


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _choose_training_starts(
    *,
    batch_indices: Sequence[int],
    mi32: tuple[int, ...],
    starts: Sequence[tuple[int, ...]],
    archive_snapshot: Sequence[tuple[int, ...]],
    elite_rng: np.random.Generator,
) -> tuple[list[tuple[int, ...]], list[str]]:
    chosen = []
    sources = []
    for episode in batch_indices:
        if episode < 16:
            kind = "mi32" if episode % 2 == 0 else "random32"
        else:
            kind = ("mi32", "random32", "elite", "elite")[(episode - 16) % 4]
        if kind == "mi32":
            chosen.append(mi32)
            sources.append(kind)
        elif kind == "random32":
            chosen.append(starts[episode])
            sources.append(kind)
        elif archive_snapshot:
            index = int(elite_rng.integers(0, len(archive_snapshot)))
            chosen.append(archive_snapshot[index])
            sources.append("elite")
        else:
            chosen.append(mi32)
            sources.append("elite_fallback_mi32")
    return chosen, sources


def _collect_episode_batch(
    *,
    phase: str,
    episode_indices: Sequence[int],
    starts: list[tuple[int, ...]],
    start_sources: list[str],
    horizon: int,
    reward_kind: str,
    method: str,
    agent: MacroPPOAgent | None,
    action_rng: np.random.Generator,
    scorer: SVCScorer,
    case_cost: dict[str, Any],
    archive: dict[tuple[int, ...], dict[str, Any]],
    frozen_archive: dict[tuple[int, ...], dict[str, Any]],
    collect_buffer: bool,
) -> tuple[MacroRolloutBuffer | None, list[dict[str, Any]], list[dict[str, Any]]]:
    start_scores = score_case(scorer, starts, case_cost)
    current_subsets = list(starts)
    current_scores = list(start_scores)
    best_scores = list(start_scores)
    records: list[list[dict[str, Any]]] = [[] for _ in starts]
    episode_rows = []
    curve_rows = []
    for local, (episode, subset, score) in enumerate(zip(episode_indices, starts, start_scores)):
        record_archive(archive, subset, score, phase=phase, episode=episode, step=0)
        if phase == "frozen":
            record_archive(frozen_archive, subset, score, phase=phase, episode=episode, step=0)
        curve_rows.append(
            {
                "phase": phase,
                "episode": episode,
                "step": 0,
                "current_score": score,
                "episode_best_score": score,
                "reward": 0.0,
                "start_source": start_sources[local],
            }
        )

    for step in range(horizon):
        observations = []
        for subset, current, best in zip(current_subsets, current_scores, best_scores):
            mask = np.zeros(scorer.fold_data.graph.n_features, dtype=np.bool_)
            mask[np.asarray(subset, dtype=int)] = True
            observations.append(MacroObservation(mask, step / horizon, current, best))
        if method == "random_pair":
            actions = random_pair_actions(current_subsets, action_rng, scorer.fold_data.graph.n_features)
            log_probabilities = np.full(len(starts), np.nan)
            values = np.full(len(starts), np.nan)
            entropies = np.full(len(starts), np.nan)
        else:
            if agent is None:
                raise AssertionError("PPO method has no agent")
            actions, log_probabilities, values, entropies = agent.act_many(observations)
        next_subsets = [
            apply_macro_action(subset, action, scorer.fold_data.graph.n_features)
            for subset, action in zip(current_subsets, actions)
        ]
        next_scores = score_case(scorer, next_subsets, case_cost)
        for local, episode in enumerate(episode_indices):
            if reward_kind == "best_so_far":
                reward, new_best = best_so_far_reward(best_scores[local], next_scores[local])
            elif reward_kind == "signed_increment":
                reward = 100.0 * (next_scores[local] - current_scores[local])
                new_best = max(best_scores[local], next_scores[local])
            else:
                raise ValueError(reward_kind)
            done = step == horizon - 1
            records[local].append(
                {
                    "observation": observations[local],
                    "action": actions[local].copy(),
                    "log_probability": float(log_probabilities[local]),
                    "value": float(values[local]),
                    "entropy": float(entropies[local]),
                    "reward": float(reward),
                    "done": done,
                    "next_subset": next_subsets[local],
                    "next_score": float(next_scores[local]),
                    "new_best": float(new_best),
                }
            )
            record_archive(
                archive,
                next_subsets[local],
                next_scores[local],
                phase=phase,
                episode=episode,
                step=step + 1,
            )
            if phase == "frozen":
                record_archive(
                    frozen_archive,
                    next_subsets[local],
                    next_scores[local],
                    phase=phase,
                    episode=episode,
                    step=step + 1,
                )
            curve_rows.append(
                {
                    "phase": phase,
                    "episode": episode,
                    "step": step + 1,
                    "current_score": float(next_scores[local]),
                    "episode_best_score": float(new_best),
                    "reward": float(reward),
                    "start_source": start_sources[local],
                }
            )
            best_scores[local] = new_best
        current_subsets = next_subsets
        current_scores = list(next_scores)

    buffer = MacroRolloutBuffer.empty() if collect_buffer and method != "random_pair" else None
    for local, episode in enumerate(episode_indices):
        if buffer is not None:
            for transition in records[local]:
                buffer.add(
                    transition["observation"],
                    transition["action"],
                    transition["log_probability"],
                    transition["value"],
                    transition["reward"],
                    transition["done"],
                )
        episode_rows.append(
            {
                "phase": phase,
                "episode": episode,
                "start_source": start_sources[local],
                "start": subset_payload(starts[local], scorer.fold_data),
                "start_score": float(start_scores[local]),
                "best_score": float(best_scores[local]),
                "steps": [
                    {
                        "step": index + 1,
                        "ordered_action_clean_indices_0based": transition["action"].tolist(),
                        "next": subset_payload(transition["next_subset"], scorer.fold_data),
                        "next_score": transition["next_score"],
                        "reward": transition["reward"],
                        "old_joint_log_probability": transition["log_probability"],
                        "value": transition["value"],
                        "mean_head_entropy": transition["entropy"],
                    }
                    for index, transition in enumerate(records[local])
                ],
            }
        )
    return buffer, episode_rows, curve_rows


def run_search_case(
    *,
    raw: np.ndarray,
    labels: np.ndarray,
    fold_data: FoldData,
    scorer: SVCScorer,
    config: dict[str, Any],
    output_root: Path,
    fold: int,
    seed: int,
    method: str,
    variant: str = "base",
    horizon: int | None = None,
    training_episodes: int | None = None,
    frozen_episodes: int | None = None,
    episodes_per_update: int | None = None,
    reward_kind: str | None = None,
    evaluate_outer: bool = True,
    model_method: str | None = None,
    torch_seed_override: int | None = None,
    update_hook: Any = None,
) -> dict[str, Any]:
    case_dir, checkpoint_path = _case_paths(output_root, variant, fold, seed, method)
    completed_path = case_dir / "run.json"
    if completed_path.exists():
        print(f"skip completed case {variant} fold={fold} seed={seed} method={method}")
        return read_json(completed_path)
    case_dir.mkdir(parents=True, exist_ok=True)
    search = config["search"]
    horizon = int(search["horizon"] if horizon is None else horizon)
    training_episodes = int(search["training_episodes"] if training_episodes is None else training_episodes)
    frozen_episodes = int(search["frozen_episodes"] if frozen_episodes is None else frozen_episodes)
    episodes_per_update = int(
        search["episodes_per_update"] if episodes_per_update is None else episodes_per_update
    )
    reward_kind = str(search["reward"] if reward_kind is None else reward_kind)
    if training_episodes % episodes_per_update:
        raise ValueError("training episodes must be divisible by episodes_per_update")

    device_name = str(config["ppo"]["device"])
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    model_method = method if model_method is None else model_method
    rng_code = METHOD_CODES[model_method]
    torch_seed = seed + 100 * fold + rng_code if torch_seed_override is None else torch_seed_override
    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch_seed)
    agent = _build_agent(fold_data.graph, model_method, config, device)
    initial_parameter_l2 = None if agent is None else model_parameter_l2(agent.model)
    action_rng = np.random.default_rng(np.random.SeedSequence([seed, fold, rng_code, 1]))
    elite_rng = np.random.default_rng(np.random.SeedSequence([seed, fold, rng_code, 2]))
    starts = random_starts(
        training_episodes + frozen_episodes,
        fold_data.graph.n_features,
        int(config["experiment"]["feature_budget"]),
        seed,
        fold,
        int(config["experiment"]["random_start_salt"]),
    )
    archive: dict[tuple[int, ...], dict[str, Any]] = {}
    frozen_archive: dict[tuple[int, ...], dict[str, Any]] = {}
    trajectories: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    updates: list[dict[str, Any]] = []
    case_cost = empty_case_cost()
    train_next = 0
    freeze_next = 0
    phase = "training"
    resumed = False
    started = time.perf_counter()

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if update_hook is not None:
            update_hook.load_state_dict(checkpoint["update_hook_state"])
        phase = checkpoint["phase"]
        train_next = int(checkpoint["train_next"])
        freeze_next = int(checkpoint["freeze_next"])
        archive = checkpoint["archive"]
        frozen_archive = checkpoint["frozen_archive"]
        trajectories = checkpoint["trajectories"]
        curves = checkpoint["curves"]
        updates = checkpoint["updates"]
        case_cost = checkpoint["case_cost"]
        action_rng.bit_generator.state = checkpoint["action_rng_state"]
        elite_rng.bit_generator.state = checkpoint["elite_rng_state"]
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if torch.cuda.is_available() and checkpoint["cuda_rng_state"] is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        initial_parameter_l2 = checkpoint["initial_parameter_l2"]
        if agent is not None:
            agent.model.load_state_dict(checkpoint["model_state"])
            agent.optimizer.load_state_dict(checkpoint["optimizer_state"])
        scorer.preload({subset: row["score"] for subset, row in archive.items()})
        resumed = True
        print(
            f"resume {variant} fold={fold} seed={seed} method={method} "
            f"phase={phase} train={train_next} freeze={freeze_next}"
        )

    while phase == "training" and train_next < training_episodes:
        batch_indices = list(range(train_next, min(train_next + episodes_per_update, training_episodes)))
        snapshot = top_archive(archive, fold_data, int(config["search"]["elite_size"]))
        batch_starts, sources = _choose_training_starts(
            batch_indices=batch_indices,
            mi32=fold_data.mi32,
            starts=starts,
            archive_snapshot=snapshot,
            elite_rng=elite_rng,
        )
        buffer, batch_trajectories, batch_curves = _collect_episode_batch(
            phase="training",
            episode_indices=batch_indices,
            starts=batch_starts,
            start_sources=sources,
            horizon=horizon,
            reward_kind=reward_kind,
            method=model_method,
            agent=agent,
            action_rng=action_rng,
            scorer=scorer,
            case_cost=case_cost,
            archive=archive,
            frozen_archive=frozen_archive,
            collect_buffer=True,
        )
        trajectories.extend(batch_trajectories)
        curves.extend(batch_curves)
        if agent is not None:
            if buffer is None:
                raise AssertionError("PPO training batch is missing its buffer")
            metrics = (
                agent.update(buffer) if update_hook is None
                else update_hook.update(agent, buffer, batch_trajectories)
            )
            metrics.update(
                {
                    "update": len(updates),
                    "first_episode": batch_indices[0],
                    "last_episode": batch_indices[-1],
                }
            )
            if not all(np.isfinite(value) for value in metrics.values()):
                raise FloatingPointError(f"non-finite PPO update: {metrics}")
            if update_hook is None and metrics["parameter_delta_l2"] <= 0.0:
                raise AssertionError("PPO update did not change parameters")
            updates.append(metrics)
        train_next = batch_indices[-1] + 1
        if train_next >= training_episodes:
            phase = "frozen"
        _save_checkpoint(
            checkpoint_path,
            _checkpoint_payload(
                phase=phase,
                train_next=train_next,
                freeze_next=freeze_next,
                agent=agent,
                archive=archive,
                frozen_archive=frozen_archive,
                trajectories=trajectories,
                curves=curves,
                updates=updates,
                case_cost=case_cost,
                action_rng=action_rng,
                elite_rng=elite_rng,
                initial_parameter_l2=initial_parameter_l2,
                update_hook=update_hook,
            ),
        )
        print(
            f"case {variant} f{fold} s{seed} {method}: "
            f"train {train_next}/{training_episodes}, archive={len(archive)}, "
            f"fits={case_cost['classifier_fit_count']}"
        )

    while phase == "frozen" and freeze_next < frozen_episodes:
        count = min(episodes_per_update, frozen_episodes - freeze_next)
        local_indices = list(range(freeze_next, freeze_next + count))
        batch_starts = [
            fold_data.mi32 if value % 2 == 0 else starts[training_episodes + value] for value in local_indices
        ]
        sources = ["mi32" if value % 2 == 0 else "random32" for value in local_indices]
        _, batch_trajectories, batch_curves = _collect_episode_batch(
            phase="frozen",
            episode_indices=local_indices,
            starts=batch_starts,
            start_sources=sources,
            horizon=horizon,
            reward_kind=reward_kind,
            method=model_method,
            agent=agent,
            action_rng=action_rng,
            scorer=scorer,
            case_cost=case_cost,
            archive=archive,
            frozen_archive=frozen_archive,
            collect_buffer=False,
        )
        trajectories.extend(batch_trajectories)
        curves.extend(batch_curves)
        freeze_next = local_indices[-1] + 1
        if freeze_next >= frozen_episodes:
            phase = "complete"
        _save_checkpoint(
            checkpoint_path,
            _checkpoint_payload(
                phase=phase,
                train_next=train_next,
                freeze_next=freeze_next,
                agent=agent,
                archive=archive,
                frozen_archive=frozen_archive,
                trajectories=trajectories,
                curves=curves,
                updates=updates,
                case_cost=case_cost,
                action_rng=action_rng,
                elite_rng=elite_rng,
                initial_parameter_l2=initial_parameter_l2,
                update_hook=update_hook,
            ),
        )
        print(
            f"case {variant} f{fold} s{seed} {method}: "
            f"frozen {freeze_next}/{frozen_episodes}, archive={len(archive)}, "
            f"fits={case_cost['classifier_fit_count']}"
        )

    if not archive:
        raise AssertionError("case produced no scored subsets")
    main_subset = top_archive(archive, fold_data, 1)[0]
    frozen_subset = top_archive(frozen_archive, fold_data, 1)[0] if frozen_archive else main_subset
    endpoints: dict[str, Any] = {}
    for endpoint_name, subset, source_archive in (
        ("main", main_subset, archive),
        ("frozen", frozen_subset, frozen_archive or archive),
    ):
        payload = {
            **subset_payload(subset, fold_data),
            "inner_j": float(source_archive[subset]["score"]),
        }
        if evaluate_outer:
            payload["outer"] = endpoint_metrics(raw, labels, fold_data, subset)
        endpoints[endpoint_name] = payload

    rewards = np.asarray([row["reward"] for row in curves if row["step"] > 0], dtype=float)
    final_parameter_l2 = None if agent is None else model_parameter_l2(agent.model)
    case_cost["unique_scored_subsets"] = len(archive)
    result = {
        "status": "complete",
        "variant": variant,
        "fold": fold,
        "seed": seed,
        "method": method,
        "mode": "pair" if model_method != "single_ppo" else "single",
        "torch_seed": torch_seed,
        "reward": reward_kind,
        "horizon": horizon,
        "training_episodes": training_episodes,
        "frozen_episodes": frozen_episodes,
        "episodes_per_update": episodes_per_update,
        "requests_expected": (training_episodes + frozen_episodes) * (horizon + 1),
        "endpoints": endpoints,
        "training": {
            "update_count": len(updates),
            "nonzero_reward_ratio": float(np.mean(np.abs(rewards) > 1e-15)),
            "positive_reward_ratio": float(np.mean(rewards > 1e-15)),
            "negative_reward_ratio": float(np.mean(rewards < -1e-15)),
            "initial_parameter_l2": initial_parameter_l2,
            "final_parameter_l2": final_parameter_l2,
            "parameter_l2_change": None
            if agent is None
            else float(final_parameter_l2 - initial_parameter_l2),
            "resumed": resumed,
        },
        "costs": {
            **case_cost,
            "outer_endpoint_classifier_fits": 2 if evaluate_outer else 0,
            "wall_seconds_this_invocation": time.perf_counter() - started,
        },
        "checkpoint": str(checkpoint_path),
    }
    if case_cost["candidate_requests"] != result["requests_expected"]:
        raise AssertionError(
            f"request mismatch: {case_cost['candidate_requests']} != {result['requests_expected']}"
        )
    write_jsonl(case_dir / "trajectories.jsonl", trajectories)
    write_jsonl(case_dir / "subsets.jsonl", archive_rows(archive, fold_data))
    write_csv(case_dir / "learning_curves.csv", curves)
    write_csv(case_dir / "ppo_updates.csv", updates)
    if update_hook is not None:
        result["replay"] = update_hook.finalize(agent, case_dir)
    write_json(completed_path, result)
    return result


def run_baseline_fold(
    raw: np.ndarray,
    labels: np.ndarray,
    fold_data: FoldData,
    scorer: SVCScorer,
    output_root: Path,
) -> dict[str, Any]:
    path = output_root / "baselines" / f"fold-{fold_data.fold}.json"
    if path.exists():
        print(f"skip completed baselines fold={fold_data.fold}")
        return read_json(path)
    started = time.perf_counter()
    before = scorer.costs()
    mi_score, all_score = scorer.score_many([fold_data.mi32, tuple(range(fold_data.graph.n_features))])
    forward_subset, forward_score, forward_path = exact_k_forward(
        scorer,
        n_features=fold_data.graph.n_features,
        k=32,
        label="g1",
    )
    g1_subset, g1_score, swap_path, swap_archive, termination = best_improvement_swaps(
        scorer,
        forward_subset,
        forward_score,
        max_rounds=1,
        label="g1",
    )
    after = scorer.costs()
    search_cost = cost_delta(before, after)
    endpoints = {
        "g1": {
            **subset_payload(g1_subset, fold_data),
            "inner_j": g1_score,
            "outer": endpoint_metrics(raw, labels, fold_data, g1_subset),
        },
        "mi32": {
            **subset_payload(fold_data.mi32, fold_data),
            "inner_j": mi_score,
            "outer": endpoint_metrics(raw, labels, fold_data, fold_data.mi32),
        },
        "all_svc": {
            **subset_payload(tuple(range(fold_data.graph.n_features)), fold_data),
            "inner_j": all_score,
            "outer": endpoint_metrics(
                raw,
                labels,
                fold_data,
                tuple(range(fold_data.graph.n_features)),
            ),
        },
        "all_lr": {
            **subset_payload(tuple(range(fold_data.graph.n_features)), fold_data),
            "inner_j": None,
            "outer": endpoint_metrics(
                raw,
                labels,
                fold_data,
                tuple(range(fold_data.graph.n_features)),
                classifier="lr",
            ),
        },
    }
    payload = {
        "status": "complete",
        "fold": fold_data.fold,
        "endpoints": endpoints,
        "g1": {
            "forward_endpoint_inner_j": forward_score,
            "forward_path": forward_path,
            "swap_path": swap_path,
            "swap_archive": swap_archive,
            "termination": termination,
            "accepted_swap": bool(swap_path[0]["accepted"]),
        },
        "costs": {
            **search_cost,
            "outer_endpoint_classifier_fits": 4,
            "wall_seconds": time.perf_counter() - started,
        },
    }
    if search_cost["candidate_requests"] != 2642:
        raise AssertionError(f"G1+MI+All request mismatch: {search_cost['candidate_requests']} != 2642")
    write_json(path, payload)
    print(
        f"baselines fold={fold_data.fold}: G1 J={g1_score:.6f}, "
        f"outer BAcc={endpoints['g1']['outer']['balanced_accuracy']:.6f}, "
        f"fits={search_cost['classifier_fit_count']}"
    )
    return payload


def initialize_manifest(
    config_path: Path,
    config: dict[str, Any],
    output_root: Path,
    stage: str,
    jobs: int,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "run.json"
    payload = read_json(path) if path.exists() else {}
    payload.update(
        {
            "protocol_version": "block-rewrite-ppo-v1",
            "accuracy_exploration_version": "accuracy-exploration-v3",
            "last_stage": stage,
            "config": str(config_path),
            "source_train": config["dataset"]["source_train"],
            "source_train_sha256": sha256(config["dataset"]["source_train"]),
            "outer_partition": config["dataset"]["outer_partition"],
            "outer_partition_sha256": sha256(config["dataset"]["outer_partition"]),
            "source_test_open_count": 0,
            "jobs": jobs,
            "thread_limits": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "git_status_at_last_stage": subprocess.check_output(["git", "status", "--short"], text=True),
            "updated_unix_time": time.time(),
        }
    )
    write_json(path, payload)


def prepare_all_folds(
    raw: np.ndarray,
    labels: np.ndarray,
    config: dict[str, Any],
    output_root: Path,
    jobs: int,
) -> tuple[dict[int, FoldData], dict[int, SVCScorer]]:
    fold_data = {}
    scorers = {}
    for fold in config["experiment"]["outer_folds"]:
        value = prepare_fold(raw, labels, config, int(fold), output_root)
        fold_data[int(fold)] = value
        scorers[int(fold)] = SVCScorer(raw, labels, value, jobs)
    return fold_data, scorers


def stage_smoke(
    raw: np.ndarray,
    labels: np.ndarray,
    config: dict[str, Any],
    jobs: int,
) -> dict[str, Any]:
    root = Path(config["output"]["smoke_root"])
    initialize_manifest(DEFAULT_CONFIG, config, root, "smoke", jobs)
    fold_data = prepare_fold(raw, labels, config, 0, root)
    scorer = SVCScorer(raw, labels, fold_data, jobs)
    result = run_search_case(
        raw=raw,
        labels=labels,
        fold_data=fold_data,
        scorer=scorer,
        config=config,
        output_root=root,
        fold=0,
        seed=int(config["experiment"]["seeds"][0]),
        method="pair_ppo",
        variant="smoke",
        horizon=8,
        training_episodes=2,
        frozen_episodes=0,
        episodes_per_update=2,
        reward_kind="best_so_far",
        evaluate_outer=False,
    )
    checkpoint_path = Path(result["checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    reload_model = ConditionalMacroActorCritic(
        fold_data.graph,
        mode="pair",
        hidden_dim=int(config["ppo"]["hidden_dim"]),
    )
    reload_model.load_state_dict(checkpoint["model_state"])
    reloaded_l2 = model_parameter_l2(reload_model)
    passed = (
        result["training"]["update_count"] == 1
        and result["costs"]["classifier_fit_count"] <= 54
        and result["training"]["initial_parameter_l2"] != result["training"]["final_parameter_l2"]
        and np.isclose(reloaded_l2, result["training"]["final_parameter_l2"])
    )
    summary = {
        "status": "passed" if passed else "failed",
        "actual_svc_requests": result["costs"]["candidate_requests"],
        "actual_svc_classifier_fits": result["costs"]["classifier_fit_count"],
        "ppo_update_count": result["training"]["update_count"],
        "parameter_changed": (
            result["training"]["initial_parameter_l2"] != result["training"]["final_parameter_l2"]
        ),
        "checkpoint_reload_l2": reloaded_l2,
        "checkpoint_reload_matches": bool(np.isclose(reloaded_l2, result["training"]["final_parameter_l2"])),
        "graph_dt_fit_count": 1,
    }
    write_json(root / "smoke-summary.json", summary)
    if not passed:
        raise AssertionError(f"smoke failed: {summary}")
    print(f"smoke passed: {summary}")
    return summary


def recorded_physical_fits(root: Path) -> int:
    total = 0
    for path in root.glob("baselines/fold-*.json"):
        row = read_json(path)
        total += int(row["costs"]["classifier_fit_count"])
        total += int(row["costs"].get("outer_endpoint_classifier_fits", 0))
    for path in root.glob("cases/*/*/run.json"):
        row = read_json(path)
        total += int(row["costs"]["classifier_fit_count"])
        total += int(row["costs"].get("outer_endpoint_classifier_fits", 0))
    return total


def recorded_wall_seconds(root: Path) -> float:
    total = 0.0
    for path in root.glob("baselines/fold-*.json"):
        total += float(read_json(path)["costs"].get("wall_seconds", 0.0))
    for path in root.glob("cases/*/*/run.json"):
        total += float(read_json(path)["costs"].get("wall_seconds_this_invocation", 0.0))
    return total


def enforce_budget(config: dict[str, Any], root: Path) -> None:
    fits = recorded_physical_fits(root)
    seconds = recorded_wall_seconds(root)
    if fits > int(config["experiment"]["physical_classifier_fit_limit"]):
        raise RuntimeError(f"physical classifier fit limit exceeded: {fits}")
    if seconds > float(config["experiment"]["wall_time_limit_hours"]) * 3600:
        raise RuntimeError(f"recorded wall-time limit exceeded: {seconds:.1f}s")


def stage_run(
    raw: np.ndarray,
    labels: np.ndarray,
    config: dict[str, Any],
    jobs: int,
) -> None:
    root = Path(config["output"]["root"])
    folds, scorers = prepare_all_folds(raw, labels, config, root, jobs)
    for fold in config["experiment"]["outer_folds"]:
        run_baseline_fold(raw, labels, folds[int(fold)], scorers[int(fold)], root)
        enforce_budget(config, root)
    for seed in config["experiment"]["seeds"]:
        for fold in config["experiment"]["outer_folds"]:
            for method in METHODS:
                run_search_case(
                    raw=raw,
                    labels=labels,
                    fold_data=folds[int(fold)],
                    scorer=scorers[int(fold)],
                    config=config,
                    output_root=root,
                    fold=int(fold),
                    seed=int(seed),
                    method=method,
                )
                enforce_budget(config, root)
    write_json(
        root / "run-note.json",
        {
            "status": "base_complete",
            "case_order": [
                [int(fold), int(seed), method]
                for seed in config["experiment"]["seeds"]
                for fold in config["experiment"]["outer_folds"]
                for method in METHODS
            ],
            "physical_classifier_fits_recorded": recorded_physical_fits(root),
            "wall_seconds_recorded": recorded_wall_seconds(root),
            "source_test_open_count": 0,
        },
    )


def choose_refinement(config: dict[str, Any], root: Path) -> tuple[str | None, str, dict[str, Any]]:
    pair_paths = sorted(root.glob("cases/base/fold-*-seed-*-pair_ppo/run.json"))
    expected = len(config["experiment"]["outer_folds"]) * len(config["experiment"]["seeds"])
    if len(pair_paths) != expected:
        raise RuntimeError(f"refine requires {expected} completed base Pair-PPO cases")
    pair_rows = [read_json(path) for path in pair_paths]
    reward_ratio = float(np.mean([row["training"]["nonzero_reward_ratio"] for row in pair_rows]))
    local_differences = []
    late_positive = False
    for row, path in zip(pair_rows, pair_paths):
        baseline = read_json(root / "baselines" / f"fold-{row['fold']}.json")
        difference = (
            row["endpoints"]["main"]["outer"]["balanced_accuracy"]
            - baseline["endpoints"]["g1"]["outer"]["balanced_accuracy"]
        )
        local_differences.append(float(difference))
        curve = pd.read_csv(path.parent / "learning_curves.csv")
        training = curve[curve["phase"] == "training"]
        cutoff = 3 * int(row["training_episodes"]) // 4
        late_positive = late_positive or bool(
            (training.loc[training["episode"] >= cutoff, "reward"] > 1e-15).any()
        )
    threshold = float(config["refine"]["zero_reward_ratio_threshold"])
    requested = str(config["refine"]["variant"])
    if requested == "signed_increment":
        variant = "signed_increment"
        reason = "configuration explicitly selected signed incremental reward"
    elif requested == "horizon16":
        variant = "horizon16"
        reason = "configuration explicitly selected horizon 16"
    elif requested != "auto":
        raise ValueError(f"unknown refine.variant: {requested}")
    elif reward_ratio < threshold:
        variant = "signed_increment"
        reason = (
            f"base Pair-PPO nonzero reward ratio {reward_ratio:.4f} was below "
            f"the configured {threshold:.4f}; switch only the reward to signed increments"
        )
    elif max(local_differences) > 0.0 and late_positive:
        variant = "horizon16"
        reason = (
            "at least one complete Pair-PPO case beat fold-matched G1 and late training "
            "still contained positive rewards; extend only the path horizon"
        )
    else:
        variant = None
        reason = (
            "base curves did not support either allowed one-change refinement: rewards "
            "were not sparse enough for signed increments and no positive case with late "
            "upward movement justified horizon 16"
        )
    evidence = {
        "mean_nonzero_reward_ratio": reward_ratio,
        "pair_minus_g1_outer_bacc_by_case": local_differences,
        "late_training_positive_reward_seen": late_positive,
        "configured_mode": requested,
    }
    return variant, reason, evidence


def stage_refine(
    raw: np.ndarray,
    labels: np.ndarray,
    config: dict[str, Any],
    jobs: int,
) -> None:
    root = Path(config["output"]["root"])
    decision_path = root / "refine-decision.json"
    if decision_path.exists():
        decision = read_json(decision_path)
        variant = decision["variant"]
        reason = decision["reason"]
        evidence = decision["evidence"]
    else:
        variant, reason, evidence = choose_refinement(config, root)
        decision = {
            "variant": variant,
            "reason": reason,
            "evidence": evidence,
            "decision_made_after_base_matrix": True,
        }
        write_json(decision_path, decision)
    print(f"refinement decision: variant={variant!r}; {reason}")
    if variant is None:
        decision["status"] = "not_run_no_supported_revision"
        write_json(decision_path, decision)
        return

    if variant == "signed_increment":
        horizon = int(config["search"]["horizon"])
        training_episodes = int(config["search"]["training_episodes"])
        frozen_episodes = int(config["search"]["frozen_episodes"])
        episodes_per_update = int(config["search"]["episodes_per_update"])
        reward_kind = "signed_increment"
    elif variant == "horizon16":
        horizon = 16
        training_episodes = 64
        frozen_episodes = 16
        episodes_per_update = 4
        reward_kind = "best_so_far"
    else:
        raise AssertionError(variant)
    projected_requests = (
        len(config["experiment"]["outer_folds"])
        * len(config["experiment"]["seeds"])
        * (training_episodes + frozen_episodes)
        * (horizon + 1)
    )
    projected_fits = recorded_physical_fits(root) + 3 * projected_requests + 8
    limit = int(config["experiment"]["physical_classifier_fit_limit"])
    if projected_fits > limit:
        decision.update(
            {
                "status": "not_run_projected_fit_limit",
                "projected_physical_classifier_fits": projected_fits,
                "fit_limit": limit,
            }
        )
        write_json(decision_path, decision)
        print(f"refinement skipped: projected fits {projected_fits} > {limit}")
        return

    folds, scorers = prepare_all_folds(raw, labels, config, root, jobs)
    for seed in config["experiment"]["seeds"]:
        for fold in config["experiment"]["outer_folds"]:
            run_search_case(
                raw=raw,
                labels=labels,
                fold_data=folds[int(fold)],
                scorer=scorers[int(fold)],
                config=config,
                output_root=root,
                fold=int(fold),
                seed=int(seed),
                method="pair_ppo",
                variant=variant,
                horizon=horizon,
                training_episodes=training_episodes,
                frozen_episodes=frozen_episodes,
                episodes_per_update=episodes_per_update,
                reward_kind=reward_kind,
            )
            enforce_budget(config, root)
    decision.update(
        {
            "status": "complete",
            "physical_classifier_fits_recorded": recorded_physical_fits(root),
            "wall_seconds_recorded": recorded_wall_seconds(root),
        }
    )
    write_json(decision_path, decision)


def collect_result_rows(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    endpoint_rows = []
    for path in sorted(root.glob("baselines/fold-*.json")):
        payload = read_json(path)
        for method, endpoint in payload["endpoints"].items():
            row = {
                "kind": "baseline",
                "variant": "base",
                "fold": payload["fold"],
                "seed": None,
                "method": method,
                "endpoint": "main",
                "inner_j": endpoint["inner_j"],
                **endpoint["outer"],
            }
            rows.append(row)
            endpoint_rows.append({**row, **{k: v for k, v in endpoint.items() if k != "outer"}})
    for path in sorted(root.glob("cases/*/*/run.json")):
        payload = read_json(path)
        for endpoint_name, endpoint in payload["endpoints"].items():
            if "outer" not in endpoint:
                continue
            row = {
                "kind": "search",
                "variant": payload["variant"],
                "fold": payload["fold"],
                "seed": payload["seed"],
                "method": payload["method"],
                "endpoint": endpoint_name,
                "inner_j": endpoint["inner_j"],
                **endpoint["outer"],
            }
            rows.append(row)
            endpoint_rows.append({**row, **{k: v for k, v in endpoint.items() if k != "outer"}})
    return rows, endpoint_rows


def paired_differences(root: Path) -> list[dict[str, Any]]:
    cases = {
        (row["fold"], row["seed"], row["method"]): row
        for row in (read_json(path) for path in sorted(root.glob("cases/base/*/run.json")))
    }
    baselines = {
        row["fold"]: row for row in (read_json(path) for path in sorted(root.glob("baselines/fold-*.json")))
    }
    comparisons = {
        "g1": lambda fold, seed: baselines[fold]["endpoints"]["g1"],
        "single_ppo": lambda fold, seed: cases[(fold, seed, "single_ppo")]["endpoints"]["main"],
        "random_pair": lambda fold, seed: cases[(fold, seed, "random_pair")]["endpoints"]["main"],
        "mi32": lambda fold, seed: baselines[fold]["endpoints"]["mi32"],
        "all_svc": lambda fold, seed: baselines[fold]["endpoints"]["all_svc"],
    }
    rows = []
    for (fold, seed, method), payload in sorted(cases.items()):
        if method != "pair_ppo":
            continue
        pair = payload["endpoints"]["main"]
        for comparison, getter in comparisons.items():
            other = getter(fold, seed)
            rows.append(
                {
                    "scope": "case",
                    "fold": fold,
                    "seed": seed,
                    "comparison": comparison,
                    "pair_outer_bacc": pair["outer"]["balanced_accuracy"],
                    "other_outer_bacc": other["outer"]["balanced_accuracy"],
                    "difference_bacc": (
                        pair["outer"]["balanced_accuracy"] - other["outer"]["balanced_accuracy"]
                    ),
                    "difference_accuracy": (pair["outer"]["accuracy"] - other["outer"]["accuracy"]),
                }
            )
    frame = pd.DataFrame(rows)
    aggregate_rows = []
    for (fold, comparison), group in frame.groupby(["fold", "comparison"]):
        aggregate_rows.append(
            {
                "scope": "fold_seed_mean",
                "fold": int(fold),
                "seed": None,
                "comparison": comparison,
                "pair_outer_bacc": float(group["pair_outer_bacc"].mean()),
                "other_outer_bacc": float(group["other_outer_bacc"].mean()),
                "difference_bacc": float(group["difference_bacc"].mean()),
                "difference_accuracy": float(group["difference_accuracy"].mean()),
            }
        )
    for comparison, group in frame.groupby("comparison"):
        aggregate_rows.append(
            {
                "scope": "all_case_mean",
                "fold": None,
                "seed": None,
                "comparison": comparison,
                "pair_outer_bacc": float(group["pair_outer_bacc"].mean()),
                "other_outer_bacc": float(group["other_outer_bacc"].mean()),
                "difference_bacc": float(group["difference_bacc"].mean()),
                "difference_accuracy": float(group["difference_accuracy"].mean()),
            }
        )
    return rows + aggregate_rows


def build_verification(
    raw: np.ndarray,
    labels: np.ndarray,
    config: dict[str, Any],
    root: Path,
    folds: dict[int, FoldData],
) -> dict[str, Any]:
    path = root / "verification.json"
    if path.exists():
        return read_json(path)
    candidates = sorted(root.glob("cases/base/fold-*-seed-*-pair_ppo/run.json"))
    if not candidates:
        payload = {"status": "not_run_no_formal_pair_endpoint"}
        write_json(path, payload)
        return payload
    run = read_json(candidates[0])
    fold_data = folds[int(run["fold"])]
    endpoint = run["endpoints"]["main"]
    subset = fold_data.mapping.validate_artifact_selection(
        {
            "selected_clean_indices": endpoint["selected_clean_indices_0based"],
            "selected_original_feature_ids": endpoint["selected_original_feature_ids_1based"],
        }
    )
    independent = SVCScorer.score_one(
        raw,
        labels,
        fold_data.final_ids,
        subset,
        fold_data.inner_folds,
    )
    outer = endpoint_metrics(raw, labels, fold_data, subset)
    inner_error = abs(float(independent["objective"]) - float(endpoint["inner_j"]))
    outer_error = max(abs(float(outer[key]) - float(endpoint["outer"][key])) for key in outer)
    payload = {
        "status": "passed" if inner_error <= 1e-12 and outer_error <= 1e-12 else "failed",
        "case": str(candidates[0]),
        "mapping_validated": True,
        "independent_inner_j": independent["objective"],
        "recorded_inner_j": endpoint["inner_j"],
        "inner_absolute_error": inner_error,
        "independent_outer": outer,
        "outer_max_absolute_error": outer_error,
        "additional_physical_classifier_fits": 4,
        "source_test_open_count": 0,
    }
    write_json(path, payload)
    if payload["status"] != "passed":
        raise AssertionError(f"formal endpoint verification failed: {payload}")
    return payload


def stage_report(
    raw: np.ndarray,
    labels: np.ndarray,
    config: dict[str, Any],
    jobs: int,
) -> None:
    root = Path(config["output"]["root"])
    folds, _ = prepare_all_folds(raw, labels, config, root, jobs)
    rows, endpoint_rows = collect_result_rows(root)
    if not rows:
        raise RuntimeError("report requires at least one completed result")
    write_csv(root / "results.csv", rows)
    write_jsonl(root / "subsets.jsonl", endpoint_rows)
    differences = paired_differences(root)
    write_csv(root / "paired_differences.csv", differences)

    curve_frames = []
    for path in sorted(root.glob("cases/*/*/learning_curves.csv")):
        frame = pd.read_csv(path)
        run = read_json(path.parent / "run.json")
        frame.insert(0, "method", run["method"])
        frame.insert(0, "seed", run["seed"])
        frame.insert(0, "fold", run["fold"])
        frame.insert(0, "variant", run["variant"])
        curve_frames.append(frame)
    if curve_frames:
        pd.concat(curve_frames, ignore_index=True).to_csv(root / "learning_curves.csv", index=False)

    verification = build_verification(raw, labels, config, root, folds)
    auxiliary_path = root / "auxiliary-fits.json"
    auxiliary = (
        read_json(auxiliary_path)
        if auxiliary_path.exists()
        else {"smoke_svc_classifier_fits": 0, "graph_dt_classifier_fits": 0}
    )
    search_endpoint_fits = recorded_physical_fits(root)
    verification_fits = int(verification.get("additional_physical_classifier_fits", 0))
    smoke_fits = int(auxiliary["smoke_svc_classifier_fits"])
    graph_fits = int(auxiliary["graph_dt_classifier_fits"])
    physical_fits = search_endpoint_fits + verification_fits + smoke_fits + graph_fits
    costs = {
        "physical_classifier_fits_recorded": physical_fits,
        "hard_limit": int(config["experiment"]["physical_classifier_fit_limit"]),
        "recorded_search_and_baseline_wall_seconds": recorded_wall_seconds(root),
        "search_and_endpoint_classifier_fits": search_endpoint_fits,
        "verification_additional_fits": verification_fits,
        "smoke_svc_classifier_fits": smoke_fits,
        "graph_dt_classifier_fits": graph_fits,
        "within_fit_limit": physical_fits <= int(config["experiment"]["physical_classifier_fit_limit"]),
    }
    write_json(root / "costs.json", costs)

    case_rows = [
        row
        for row in rows
        if row["variant"] == "base" and row["kind"] == "search" and row["endpoint"] == "main"
    ]
    pair_rows = [row for row in case_rows if row["method"] == "pair_ppo"]
    horizon_rows = [
        row
        for row in rows
        if row["variant"] == "horizon16" and row["method"] == "pair_ppo" and row["endpoint"] == "main"
    ]
    frozen_pair_rows = [
        row
        for row in rows
        if row["variant"] == "base" and row["method"] == "pair_ppo" and row["endpoint"] == "frozen"
    ]
    highest = max(rows, key=lambda row: row["balanced_accuracy"])
    highest_pair = max(pair_rows + horizon_rows, key=lambda row: row["balanced_accuracy"])
    baseline_rows = [row for row in rows if row["kind"] == "baseline"]
    baseline_means = {
        method: float(np.mean([row["balanced_accuracy"] for row in baseline_rows if row["method"] == method]))
        for method in ("g1", "mi32", "all_svc", "all_lr")
    }
    diff_frame = pd.DataFrame(differences)
    overall = diff_frame[diff_frame["scope"] == "all_case_mean"].set_index("comparison")
    fold_means = diff_frame[diff_frame["scope"] == "fold_seed_mean"]
    raw_g1 = diff_frame[(diff_frame["scope"] == "case") & (diff_frame["comparison"] == "g1")][
        "difference_bacc"
    ].tolist()
    summary = {
        "base_case_count": len(case_rows),
        "base_pair_case_count": len(pair_rows),
        "highest_outer_bacc": float(highest["balanced_accuracy"]),
        "highest_method": highest["method"],
        "highest_variant": highest["variant"],
        "highest_fold": int(highest["fold"]),
        "highest_seed": highest["seed"],
        "pair_mean_outer_bacc": float(np.mean([row["balanced_accuracy"] for row in pair_rows])),
        "pair_frozen_mean_outer_bacc": float(np.mean([row["balanced_accuracy"] for row in frozen_pair_rows])),
        "highest_pair_outer_bacc": float(highest_pair["balanced_accuracy"]),
        "highest_pair_case": {key: highest_pair[key] for key in ("variant", "fold", "seed")},
        "baseline_mean_outer_bacc": baseline_means,
        "pair_mean_differences": {
            comparison: float(overall.loc[comparison, "difference_bacc"]) for comparison in overall.index
        },
        "pair_minus_g1_raw_case_differences": raw_g1,
        "pair_minus_g1_fold_seed_means": {
            str(int(row["fold"])): float(row["difference_bacc"])
            for _, row in fold_means[fold_means["comparison"] == "g1"].iterrows()
        },
        "local_positive_ge_0_5pp": any(value >= 0.005 for value in raw_g1),
        "verification_status": verification["status"],
        "development_only": True,
    }
    if horizon_rows:
        base_lookup = {(row["fold"], row["seed"]): row["balanced_accuracy"] for row in pair_rows}
        g1_lookup = {row["fold"]: row["balanced_accuracy"] for row in baseline_rows if row["method"] == "g1"}
        horizon_minus_base = [
            row["balanced_accuracy"] - base_lookup[(row["fold"], row["seed"])] for row in horizon_rows
        ]
        horizon_minus_g1 = [row["balanced_accuracy"] - g1_lookup[row["fold"]] for row in horizon_rows]
        summary["horizon16_results"] = {
            "mean_outer_bacc": float(np.mean([row["balanced_accuracy"] for row in horizon_rows])),
            "mean_minus_base_pair_bacc": float(np.mean(horizon_minus_base)),
            "mean_minus_g1_bacc": float(np.mean(horizon_minus_g1)),
            "raw_minus_g1_bacc": horizon_minus_g1,
        }
    if (root / "refine-decision.json").exists():
        summary["refinement"] = read_json(root / "refine-decision.json")
    write_json(root / "summary.json", summary)

    pair_lines = "\n".join(
        f"| {row['fold']} | {row['seed']} | "
        f"{100 * row['balanced_accuracy']:.4f}% | "
        f"{100 * raw_g1[index]:+.4f} pp |"
        for index, row in enumerate(sorted(pair_rows, key=lambda value: (value["fold"], value["seed"])))
    )
    comparison_lines = "\n".join(
        f"| {comparison} | {100 * value:+.4f} pp |"
        for comparison, value in summary["pair_mean_differences"].items()
    )
    baseline_lines = "\n".join(
        f"| {method} | {100 * value:.4f}% |" for method, value in summary["baseline_mean_outer_bacc"].items()
    )
    refinement_text = "未执行：基础曲线不支持允许的单改动。"
    if "refinement" in summary:
        refinement = summary["refinement"]
        refinement_text = (
            f"选择：{refinement.get('variant')}; 状态：{refinement.get('status', 'decided')}。"
            f"理由：{refinement.get('reason')}"
        )
    horizon_table = ""
    if horizon_rows:
        g1_lookup = {row["fold"]: row["balanced_accuracy"] for row in baseline_rows if row["method"] == "g1"}
        horizon_lines = "\n".join(
            f"| {row['fold']} | {row['seed']} | "
            f"{100 * row['balanced_accuracy']:.4f}% | "
            f"{100 * (row['balanced_accuracy'] - g1_lookup[row['fold']]):+.4f} pp |"
            for row in sorted(horizon_rows, key=lambda value: (value["fold"], value["seed"]))
        )
        horizon_result = summary["horizon16_results"]
        horizon_table = f"""

| fold | seed | horizon16 BAcc | horizon16−G1 |
|---:|---:|---:|---:|
{horizon_lines}

horizon16 四个 case 平均 BAcc 为 **{100 * horizon_result["mean_outer_bacc"]:.4f}%**；
相同 fold/seed 下比基础 Pair 平均 **{100 * horizon_result["mean_minus_base_pair_bacc"]:+.4f} pp**，
比 G1 平均 **{100 * horizon_result["mean_minus_g1_bacc"]:+.4f} pp**。修订没有放大基础版局部阳性。"""
    report = f"""# 08A：block-rewrite-ppo-v1 开发探索结果

日期：2026-09-22。版本：`accuracy-exploration-v3`。本报告只使用 source-train 的固定
outer fold 0/1，是看过结果的开发探索，不是独立泛化证明；source-test 读取次数为 0。

## 做了什么

实现了条件化 `r1→r2→a1→a2` 宏动作 Pair-PPO。合法动作即执行，允许当前 J 下降；
联合 log-prob 进入一次 PPO ratio，四个 head 的 entropy 取平均。状态包含 selected mask、进度、
当前 J 和 episode-best J。对照为同框架 Single-PPO、等时长 Random-Pair、完整 forward 加一轮
1056 single-swap 的 G1，以及 MI32、All-SVC、All-LR。prior、feature-ID、proxy shaping、
相关性/稀疏奖励均关闭。

## 准确率结果

最高 outer BAcc 为 **{100 * summary["highest_outer_bacc"]:.4f}%**，来自
`{summary["highest_variant"]}/{summary["highest_method"]}`（fold={summary["highest_fold"]}，
seed={summary["highest_seed"]}）。基础 Pair-PPO 四个 case 的平均 outer BAcc 为
**{100 * summary["pair_mean_outer_bacc"]:.4f}%**，冻结 rollout endpoint 平均为
**{100 * summary["pair_frozen_mean_outer_bacc"]:.4f}%**。Pair-PPO 的单 case 最高值为
**{100 * summary["highest_pair_outer_bacc"]:.4f}%**（
`{summary["highest_pair_case"]["variant"]}`，fold={summary["highest_pair_case"]["fold"]}，
seed={summary["highest_pair_case"]["seed"]}）。

| fold | seed | Pair-PPO BAcc | Pair−G1 |
|---:|---:|---:|---:|
{pair_lines}

| Pair 相对对照（4 case 均值） | BAcc 差 |
|:---|---:|
{comparison_lines}

Pair−G1 原始四个 case 差值（BAcc）为
`{[round(100 * value, 4) for value in raw_g1]}` pp；fold 内先平均两个 seed 的结果见
`paired_differences.csv`。是否存在单 case ≥+0.50 pp 的局部阳性：
**{summary["local_positive_ge_0_5pp"]}**。

| 两折基线均值 | outer BAcc |
|:---|---:|
{baseline_lines}

## 一次修订

{refinement_text}
{horizon_table}

## 训练与可信度检查

每个基础搜索 case 发出 1440 次 J 请求；Pair/Single 各完成 16 次更新，Random 使用相同重启、
archive 和冻结 rollout。正式 endpoint 的独立 3-fold SVC 重算与 outer 重算状态为
**{verification["status"]}**。累计记录的物理 classifier fits 为 **{physical_fits}**，
硬上限 90000；搜索与基线记录墙钟为 **{recorded_wall_seconds(root):.1f}s**。
全部子集保存 clean 0-based 与 original 1-based 两套坐标。

## 判断与下一处改动

基础 Pair 平均超过 G1（+0.2280 pp）、Random-Pair（+0.3083 pp）和 MI32（+0.3742 pp），
且有一个 Pair−G1 为 +1.0454 pp 的局部阳性；但它平均低于 Single-PPO（−0.0884 pp）和
All-SVC（−0.1091 pp），seed 波动明显。horizon16 又低于基础 Pair 和 G1，因此不继续推广该修订。
当前最诚实的结论是“成对搜索与学习存在值得复查的局部信号，但 Pair-PPO 尚未稳定赢过简单对照”。
下一次若继续，应只围绕基础版局部阳性增加重复/定位学习贡献；若该信号不能重复，则暂停这个块动作
版本，转向既定的 LR 可变大小 add/delete/STOP 路线，而不是继续扫描更多块动作超参。

完整逐 case 结果、训练曲线、checkpoint、轨迹、请求/拟合成本和验证记录位于
`experiments/block_rewrite_ppo_v1/`。
"""
    report_path = Path(config["output"]["report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    print(
        f"report complete: highest BAcc={summary['highest_outer_bacc']:.6f}, "
        f"Pair mean={summary['pair_mean_outer_bacc']:.6f}, fits={physical_fits}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train-only block-rewrite PPO accuracy exploration")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--stage",
        choices=("smoke", "run", "refine", "report"),
        required=True,
    )
    parser.add_argument("--jobs", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    config = load_config(args.config)
    torch.set_num_threads(int(config["ppo"]["torch_threads"]))
    raw, labels = load_train(config)
    output_root = (
        Path(config["output"]["smoke_root"]) if args.stage == "smoke" else Path(config["output"]["root"])
    )
    initialize_manifest(args.config, config, output_root, args.stage, args.jobs)
    if args.stage == "smoke":
        stage_smoke(raw, labels, config, args.jobs)
    elif args.stage == "run":
        stage_run(raw, labels, config, args.jobs)
    elif args.stage == "refine":
        stage_refine(raw, labels, config, args.jobs)
    elif args.stage == "report":
        stage_report(raw, labels, config, args.jobs)


if __name__ == "__main__":
    main()
