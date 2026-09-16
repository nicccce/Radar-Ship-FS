#!/usr/bin/env python3
"""Task 4A: enumerate old support and replay proposal mechanisms without RL training."""

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

import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed, parallel_config

from radar_ship_fs.ppo.config import ExperimentConfig
from radar_ship_fs.ppo.ppo_env import FeatureSelectionEnv
from radar_ship_fs.ppo.ppo_graph import FeatureGraph
from run_search_diagnosis import fit_search, load_raw, read, read_lines
from run_unified_baselines import _write_csv, _write_json, build_seed_context

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 environment
    import tomli as tomllib


ROOT = Path("experiments/action_support_v1")
TASK3 = Path("experiments/search_diagnosis_v1")
CONFIG = Path("configs/v16n/action_support_v1.toml")
PROTOCOL = Path("documents/research-plan/action-support-protocol.md")
SEEDS = (42, 43, 44, 45, 46)
KS = (8, 16, 32)
PHASES = ("forward", "local")
METHODS = ("old_mask", "quality_plus_global", "uniform_legal")
EPS = 1e-12


def sha(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def graph_from_context(context: dict) -> FeatureGraph:
    quality = np.asarray(context["pool_quality"], dtype=np.float32)
    d = len(quality)
    identity = np.eye(d, dtype=np.float32)
    return FeatureGraph(
        identity,
        identity,
        identity,
        identity,
        np.column_stack([quality, quality]),
        quality,
        np.asarray(context["mi_values"], dtype=np.float32),
        np.arange(d),
        0.8,
        0,
        0,
    )


def make_env(
    graph: FeatureGraph,
    subset: tuple[int, ...],
    *,
    method: str,
    seed: int,
) -> FeatureSelectionEnv:
    if method == "old_mask":
        quality_pool, exploration_pool = 4, 0
    elif method == "quality_plus_global":
        quality_pool, exploration_pool = 4, 4
    elif method == "uniform_legal":
        quality_pool, exploration_pool = graph.n_features, 0
    else:
        raise ValueError(f"unknown method: {method}")
    config = ExperimentConfig(
        seed=seed,
        feature_budget=len(subset),
        min_features=min(4, len(subset)),
        feature_id_node_feature=False,
        feature_id_reward_weight=0.0,
        sparsity_bonus=0.0,
        swap_candidate_pool=quality_pool,
        swap_exploration_pool=exploration_pool,
        max_swaps=2,
    )
    mask = np.zeros(graph.n_features, dtype=bool)
    mask[list(subset)] = True
    return FeatureSelectionEnv(graph, None, config, baseline_objective=0.0, initial_mask=mask)


def next_states_old(graph: FeatureGraph, subset: tuple[int, ...]) -> set[tuple[int, ...]]:
    env = make_env(graph, subset, method="old_mask", seed=0)
    removes = np.flatnonzero(env.observation().valid_actions[: graph.n_features])
    reached: set[tuple[int, ...]] = set()
    for remove in removes:
        branch = make_env(graph, subset, method="old_mask", seed=0)
        branch._take_feature_action(int(remove))
        adds = np.flatnonzero(branch.observation().valid_actions[: graph.n_features])
        for add in adds:
            reached.add(tuple(sorted(set(subset) - {int(remove)} | {int(add)})))
    assert len(reached) == 16
    return reached


def old_reachable(
    graph: FeatureGraph, subset: tuple[int, ...], max_swaps: int = 2
) -> dict[tuple[int, ...], int]:
    distance = {subset: 0}
    frontier = {subset}
    for depth in range(1, max_swaps + 1):
        following: set[tuple[int, ...]] = set()
        for state in frontier:
            following.update(next_states_old(graph, state))
        following -= set(distance)
        for state in following:
            distance[state] = depth
        frontier = following
    return distance


def full_single_group(groups: list[dict], anchor: int) -> dict:
    return next(group for group in groups if group["kind"] == "single" and group["anchor_id"] == anchor)


@dataclass
class ScoreStore:
    context: object
    original_ids: np.ndarray
    jobs: int
    records: dict[tuple[int, ...], dict]
    imported: set[tuple[int, ...]]
    new_fit_seconds: float = 0.0
    new_subsets: int = 0

    @classmethod
    def from_task3(cls, context, original_ids, candidates, jobs):
        records = {}
        for candidate in candidates:
            subset = tuple(candidate["clean_indices_0based"])
            records[subset] = {
                "clean_indices_0based": list(subset),
                "original_feature_ids_1based": candidate["original_feature_ids_1based"],
                "search_accuracy": candidate["search_accuracy"],
                "fold_scores": candidate["fold_scores"],
                "score_source": "search_diagnosis_v1",
                "physical_fit_count": 0,
                "fit_seconds": 0.0,
            }
        return cls(context, original_ids, jobs, records, set(records))

    def score_many(self, subsets: set[tuple[int, ...]]) -> None:
        missing = sorted(set(subsets) - set(self.records))
        results = Parallel(n_jobs=self.jobs)(delayed(fit_search)(self.context, subset) for subset in missing)
        for subset, (score, folds, elapsed) in zip(missing, results):
            self.records[subset] = {
                "clean_indices_0based": list(subset),
                "original_feature_ids_1based": self.original_ids[list(subset)].tolist(),
                "search_accuracy": float(score),
                "fold_scores": [float(value) for value in folds],
                "score_source": "action_support_v1_refit",
                "physical_fit_count": 5,
                "fit_seconds": float(elapsed),
            }
            self.new_fit_seconds += float(elapsed)
            self.new_subsets += 1

    def score(self, subset: tuple[int, ...]) -> float:
        return float(self.records[subset]["search_accuracy"])


def replay_seed_parts(replay_seed: int, seed: int, k: int, phase: str, method: str) -> tuple[int, ...]:
    return (
        replay_seed,
        seed,
        k,
        PHASES.index(phase),
        METHODS.index(method),
    )


def generate_replay(
    graph: FeatureGraph,
    start: tuple[int, ...],
    *,
    method: str,
    replay_seed: int,
    seed: int,
    k: int,
    phase: str,
    budget: int,
    max_requests: int,
) -> tuple[list[tuple[int, ...]], list[dict]]:
    depth_rng = np.random.default_rng(np.random.SeedSequence([replay_seed, seed, k, PHASES.index(phase)]))
    action_rng = np.random.default_rng(
        np.random.SeedSequence(replay_seed_parts(replay_seed, seed, k, phase, method))
    )
    env_seed = int(action_rng.integers(0, 2**31))
    env = make_env(graph, start, method=method, seed=env_seed)
    unique: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    events: list[dict] = []
    for request in range(1, max_requests + 1):
        depth = int(depth_rng.integers(1, 3))
        env.reset()
        actions: list[int] = []
        for _ in range(depth):
            valid_remove = np.flatnonzero(env.observation().valid_actions[: graph.n_features])
            remove = int(action_rng.choice(valid_remove))
            env._take_feature_action(remove)
            valid_add = np.flatnonzero(env.observation().valid_actions[: graph.n_features])
            add = int(action_rng.choice(valid_add))
            env._take_feature_action(add)
            actions.extend([remove, add])
        subset = tuple(np.flatnonzero(env.selected).tolist())
        cache_hit = subset == start or subset in seen
        if not cache_hit:
            seen.add(subset)
            unique.append(subset)
        events.append(
            {
                "request": request,
                "sampled_swap_depth": depth,
                "actions_clean_0based": actions,
                "clean_indices_0based": list(subset),
                "cache_hit": cache_hit,
                "unique_budget_after_request": len(unique),
            }
        )
        if len(unique) == budget:
            return unique, events
    raise RuntimeError(
        f"{method} seed={seed} K={k} {phase} replay={replay_seed} "
        f"did not reach {budget} unique subsets in {max_requests} requests"
    )


def support_probability_for_swap(
    graph: FeatureGraph,
    start: tuple[int, ...],
    target: tuple[int, ...],
) -> tuple[float, float, float]:
    removed = list(set(start) - set(target))
    added = list(set(target) - set(start))
    if len(removed) != 1 or len(added) != 1:
        raise ValueError("single exchange required")
    env = make_env(graph, start, method="quality_plus_global", seed=0)
    selected = np.zeros(graph.n_features, dtype=bool)
    selected[list(start)] = True
    remove_probability = env.swap_action_support_probability(selected, removed[0], largest=False)
    after_remove = selected.copy()
    after_remove[removed[0]] = False
    eligible_add = ~after_remove
    eligible_add[removed[0]] = False
    add_probability = env.swap_action_support_probability(eligible_add, added[0], largest=True)
    return remove_probability, add_probability, remove_probability * add_probability


def load_case(seed: int, jobs: int):
    case = TASK3 / "search" / f"nested_dev-seed-{seed}"
    context_payload = read(case / "context.json")
    complete = read(case / "complete.json")
    candidates = read_lines(case / "candidates.jsonl")
    groups = read(case / "groups.json")["groups"]
    raw, y = load_raw()
    fit_rows = np.asarray(context_payload["fit_original_rows"], dtype=int)
    original_ids = np.asarray(context_payload["final_feature_ids"], dtype=int)
    context = build_seed_context(
        raw[fit_rows][:, original_ids - 1],
        y[fit_rows],
        seed=seed,
        validation_fraction=0.25,
        n_splits=5,
    )
    assert context.cv_tree_random_state == context_payload["cv_tree_random_state"]
    return context_payload, complete, candidates, groups, context, original_ids, ScoreStore.from_task3(
        context, original_ids, candidates, jobs
    )


def prepare_manifest() -> dict:
    sources = [
        Path(__file__),
        CONFIG,
        PROTOCOL,
        Path("src/radar_ship_fs/ppo/ppo_env.py"),
        Path("src/radar_ship_fs/ppo/config.py"),
        Path("src/radar_ship_fs/experiment/config.py"),
        Path("src/radar_ship_fs/ppo/run_session.py"),
        Path("src/run_search_diagnosis.py"),
        Path("src/audit_action_support.py"),
        Path("tests/test_action_support.py"),
    ]
    hashes = {str(path): sha(path) for path in sources}
    task3_hashes = {
        str(path): sha(path)
        for path in [TASK3 / "manifest.json", TASK3 / "delivery-manifest.json", TASK3 / "audit.json"]
    }
    task3_delivery = read(TASK3 / "delivery-manifest.json")
    assert all(
        sha(path) == expected for path, expected in task3_delivery["artifact_sha256"].items()
    ), "Task 3 delivery artifact changed"
    if (ROOT / "manifest.json").exists():
        manifest = read(ROOT / "manifest.json")
        assert manifest["code_protocol_config_hashes"] == hashes
        assert manifest["task3_input_hashes"] == task3_hashes
        return manifest
    ROOT.mkdir(parents=True, exist_ok=False)
    config = tomllib.loads(CONFIG.read_text())
    manifest = {
        "version": config["version"],
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code_protocol_config_hashes": hashes,
        "task3_input_hashes": task3_hashes,
        "train_sha256": config["source_train_sha256"],
        "source_test_open_count": 0,
        "rl_training_count": 0,
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_status_before_run": subprocess.check_output(["git", "status", "--short"], text=True),
        "frozen_config": config,
        "task3_delivery_artifacts_verified": True,
        "historical_matched_policy_available": False,
    }
    _write_json(ROOT / "manifest.json", manifest)
    return manifest


def smoke() -> None:
    payload, complete, candidates, _, _, _, _ = load_case(42, jobs=1)
    graph = graph_from_context(payload)
    start = tuple(candidates[complete["starts"]["8"]]["clean_indices_0based"])
    assert len(old_reachable(graph, start)) == 117
    for method in METHODS:
        unique, _ = generate_replay(
            graph,
            start,
            method=method,
            replay_seed=41001,
            seed=42,
            k=8,
            phase="forward",
            budget=4,
            max_requests=100,
        )
        assert len(unique) == 4
    target = next(iter(next_states_old(graph, start)))
    assert support_probability_for_swap(graph, start, target)[2] > 0.0
    print("action-support smoke passed: old reachability=117, all replay methods budget=4")


def run(jobs: int) -> None:
    manifest = prepare_manifest()
    if (ROOT / "complete.json").exists():
        print("action-support-v1 already complete")
        return
    if any(path.name != "manifest.json" for path in ROOT.iterdir()):
        raise RuntimeError("partial action-support output refuses overwrite")

    config = manifest["frozen_config"]
    budget = int(config["replay"]["incremental_unique_subset_budget"])
    replay_seeds = [int(value) for value in config["replay"]["replay_seeds"]]
    max_requests = int(config["replay"]["maximum_requests_per_run"])
    reachable_rows: list[dict] = []
    gap_rows: list[dict] = []
    replay_rows: list[dict] = []
    curve_rows: list[dict] = []
    coverage_rows: list[dict] = []
    cost_rows: list[dict] = []
    case_dir = ROOT / "cases"
    case_dir.mkdir()
    low_mi = pd.read_csv(TASK3 / "ablation" / "low_mi_contributions.csv")
    strong_low_mi = low_mi[
        (low_mi.net_search_gain > EPS)
        & (low_mi.net_lr_gain > EPS)
        & (low_mi.conditional_search_gain > EPS)
        & (low_mi.conditional_lr_gain > EPS)
    ]
    assert len(strong_low_mi) == 23

    started_all = time.perf_counter()
    for seed in SEEDS:
        started = time.perf_counter()
        payload, complete, candidates, groups, _, original_ids, store = load_case(seed, jobs)
        graph = graph_from_context(payload)
        needed: set[tuple[int, ...]] = set()
        origins: list[dict] = []
        generated: list[dict] = []
        for k in KS:
            for phase, complete_key in (("forward", "starts"), ("local", "terminals")):
                anchor_id = int(complete[complete_key][str(k)])
                start = tuple(candidates[anchor_id]["clean_indices_0based"])
                direct = next_states_old(graph, start)
                reachable = old_reachable(graph, start)
                assert len(reachable) == 117
                group = full_single_group(groups, anchor_id)
                full_single = {start} | {
                    tuple(candidates[int(member["candidate_id"])]["clean_indices_0based"])
                    for member in group["members"]
                }
                assert len(full_single) == 1 + k * (graph.n_features - k)
                needed.update(reachable)
                needed.update(full_single)
                origin = {
                    "seed": seed,
                    "k": k,
                    "phase": phase,
                    "anchor_id_task3": anchor_id,
                    "start": start,
                    "direct": direct,
                    "reachable": reachable,
                    "full_single": full_single,
                    "task3_rule_endpoint_id": int(
                        complete["terminals" if phase == "forward" else "double_ends"][str(k)]
                    ),
                }
                origins.append(origin)
                for method in METHODS:
                    for replay_seed in replay_seeds:
                        unique, events = generate_replay(
                            graph,
                            start,
                            method=method,
                            replay_seed=replay_seed,
                            seed=seed,
                            k=k,
                            phase=phase,
                            budget=budget,
                            max_requests=max_requests,
                        )
                        needed.update(unique)
                        generated.append(
                            {
                                "seed": seed,
                                "k": k,
                                "phase": phase,
                                "method": method,
                                "replay_seed": replay_seed,
                                "start": start,
                                "unique": unique,
                                "events": events,
                            }
                        )

        store.score_many(needed)
        referenced = sorted(needed)
        candidate_id = {subset: index for index, subset in enumerate(referenced)}
        bank_rows = []
        for subset in referenced:
            record = dict(store.records[subset])
            record["candidate_id"] = candidate_id[subset]
            assert record["original_feature_ids_1based"] == original_ids[list(subset)].tolist()
            record["coordinate_crosscheck_passed"] = True
            bank_rows.append(record)
        seed_dir = case_dir / f"seed-{seed}"
        seed_dir.mkdir()
        write_jsonl(seed_dir / "candidate_scores.jsonl", bank_rows)

        for origin in origins:
            start = origin["start"]
            direct_with_start = set(origin["direct"]) | {start}
            full_best = max(origin["full_single"], key=lambda subset: store.score(subset))
            direct_best = max(direct_with_start, key=lambda subset: store.score(subset))
            ceiling_best = max(origin["reachable"], key=lambda subset: store.score(subset))
            task3_endpoint = tuple(candidates[origin["task3_rule_endpoint_id"]]["clean_indices_0based"])
            gap_rows.append(
                {
                    "seed": seed,
                    "k": origin["k"],
                    "phase": origin["phase"],
                    "anchor_id_task3": origin["anchor_id_task3"],
                    "start_candidate_id": candidate_id[start],
                    "start_score": store.score(start),
                    "complete_1swap_count_including_start": len(origin["full_single"]),
                    "old_direct_count_including_start": len(direct_with_start),
                    "old_reachable_2swap_count_including_start": len(origin["reachable"]),
                    "complete_1swap_best_score": store.score(full_best),
                    "old_direct_best_score": store.score(direct_best),
                    "old_2swap_ceiling_score": store.score(ceiling_best),
                    "action_support_gap_1swap": store.score(full_best) - store.score(direct_best),
                    "old_2swap_ceiling_minus_complete_1swap_best": store.score(ceiling_best)
                    - store.score(full_best),
                    "historical_matched_policy_best_score": None,
                    "historical_search_policy_gap": None,
                    "historical_gap_status": "NA_no_policy_run_on_these_30_nested_dev_origins",
                    "task3_different_support_rule_best_score": store.score(task3_endpoint),
                    "task3_rule_not_used_as_policy_gap": True,
                    "complete_best_clean_0based": json.dumps(list(full_best)),
                    "complete_best_original_1based": json.dumps(original_ids[list(full_best)].tolist()),
                    "old_direct_best_clean_0based": json.dumps(list(direct_best)),
                    "old_direct_best_original_1based": json.dumps(original_ids[list(direct_best)].tolist()),
                    "old_ceiling_best_clean_0based": json.dumps(list(ceiling_best)),
                    "old_ceiling_best_original_1based": json.dumps(original_ids[list(ceiling_best)].tolist()),
                }
            )
            for subset, distance in sorted(origin["reachable"].items()):
                reachable_rows.append(
                    {
                        "seed": seed,
                        "k": origin["k"],
                        "phase": origin["phase"],
                        "candidate_id": candidate_id[subset],
                        "minimum_swaps": distance,
                        "search_accuracy": store.score(subset),
                        "clean_indices_0based": json.dumps(list(subset)),
                        "original_feature_ids_1based": json.dumps(original_ids[list(subset)].tolist()),
                        "coordinate_crosscheck_passed": True,
                    }
                )

        ceiling_lookup = {
            (row["k"], row["phase"]): row["old_2swap_ceiling_score"]
            for row in gap_rows
            if row["seed"] == seed
        }
        event_rows: list[dict] = []
        for item in generated:
            best_score = store.score(item["start"])
            best_subset = item["start"]
            for logical_budget, subset in enumerate(item["unique"], start=1):
                score = store.score(subset)
                if score > best_score + EPS:
                    best_score, best_subset = score, subset
                curve_rows.append(
                    {
                        "seed": seed,
                        "k": item["k"],
                        "phase": item["phase"],
                        "method": item["method"],
                        "replay_seed": item["replay_seed"],
                        "unique_subset_budget": logical_budget,
                        "candidate_score": score,
                        "best_so_far_score": best_score,
                        "best_so_far_gain": best_score - store.score(item["start"]),
                        "best_candidate_id": candidate_id[best_subset],
                    }
                )
            requests = len(item["events"])
            cache_hits = sum(event["cache_hit"] for event in item["events"])
            ceiling = ceiling_lookup[(item["k"], item["phase"])]
            replay_rows.append(
                {
                    "seed": seed,
                    "k": item["k"],
                    "phase": item["phase"],
                    "method": item["method"],
                    "replay_seed": item["replay_seed"],
                    "incremental_unique_subsets": len(item["unique"]),
                    "logical_classifier_fits": 5 * len(item["unique"]),
                    "requests": requests,
                    "cache_hits": cache_hits,
                    "final_best_score": best_score,
                    "final_best_gain": best_score - store.score(item["start"]),
                    "old_2swap_ceiling_score": ceiling if item["method"] == "old_mask" else None,
                    "search_policy_gap_at_64": ceiling - best_score
                    if item["method"] == "old_mask"
                    else None,
                }
            )
            event_rows.extend(
                {
                    **event,
                    "k": item["k"],
                    "phase": item["phase"],
                    "method": item["method"],
                    "replay_seed": item["replay_seed"],
                    "candidate_id": candidate_id[tuple(event["clean_indices_0based"])],
                    "original_feature_ids_1based": original_ids[
                        event["clean_indices_0based"]
                    ].tolist(),
                }
                for event in item["events"]
            )
        write_jsonl(seed_dir / "replay_requests.jsonl", event_rows)

        for k in KS:
            if k == 16:
                start_id = int(complete["starts"][str(k)])
                terminal_id = int(complete["terminals"][str(k)])
                current_id = start_id
                path_probability = 1.0
                edge_index = 0
                while current_id != terminal_id:
                    group = full_single_group(groups, current_id)
                    next_id = max(
                        [int(member["candidate_id"]) for member in group["members"]],
                        key=lambda index: candidates[index]["search_accuracy"],
                    )
                    assert (
                        candidates[next_id]["search_accuracy"]
                        > candidates[current_id]["search_accuracy"] + EPS
                    )
                    current = tuple(candidates[current_id]["clean_indices_0based"])
                    target = tuple(candidates[next_id]["clean_indices_0based"])
                    remove_p, add_p, pair_p = support_probability_for_swap(graph, current, target)
                    path_probability *= pair_p
                    edge_index += 1
                    coverage_rows.append(
                        {
                            "case_type": "k16_accepted_edge",
                            "seed": seed,
                            "k": k,
                            "phase": "forward",
                            "edge_index": edge_index,
                            "anchor_id_task3": current_id,
                            "candidate_id_task3": next_id,
                            "removed_original_id": original_ids[list(set(current) - set(target))][0],
                            "added_original_id": original_ids[list(set(target) - set(current))][0],
                            "remove_support_probability": remove_p,
                            "add_support_probability_given_remove": add_p,
                            "exchange_support_probability": pair_p,
                            "path_support_probability_through_edge": path_probability,
                            "passed": pair_p > 0.0,
                        }
                    )
                    current_id = next_id
                if terminal_id != start_id:
                    coverage_rows.append(
                        {
                            "case_type": "k16_improved_endpoint",
                            "seed": seed,
                            "k": k,
                            "phase": "forward",
                            "edge_index": edge_index,
                            "anchor_id_task3": start_id,
                            "candidate_id_task3": terminal_id,
                            "removed_original_id": None,
                            "added_original_id": None,
                            "remove_support_probability": None,
                            "add_support_probability_given_remove": None,
                            "exchange_support_probability": path_probability,
                            "path_support_probability_through_edge": path_probability,
                            "passed": path_probability > 0.0,
                        }
                    )

        for row in strong_low_mi[strong_low_mi.seed == seed].itertuples(index=False):
            start = tuple(candidates[int(row.anchor_id)]["clean_indices_0based"])
            target = tuple(candidates[int(row.candidate_id)]["clean_indices_0based"])
            remove_p, add_p, pair_p = support_probability_for_swap(graph, start, target)
            coverage_rows.append(
                {
                    "case_type": "low_mi_conditional_contribution",
                    "seed": seed,
                    "k": int(row.k),
                    "phase": row.phase,
                    "edge_index": 1,
                    "anchor_id_task3": int(row.anchor_id),
                    "candidate_id_task3": int(row.candidate_id),
                    "removed_original_id": int(row.removed_original_id),
                    "added_original_id": int(row.added_original_id),
                    "remove_support_probability": remove_p,
                    "add_support_probability_given_remove": add_p,
                    "exchange_support_probability": pair_p,
                    "path_support_probability_through_edge": pair_p,
                    "passed": pair_p > 0.0,
                }
            )

        cost_rows.append(
            {
                "seed": seed,
                "referenced_unique_subsets": len(referenced),
                "imported_task3_scores": sum(subset in store.imported for subset in referenced),
                "new_unique_scored_subsets": store.new_subsets,
                "physical_classifier_fits": 5 * store.new_subsets,
                "sum_physical_fit_seconds": store.new_fit_seconds,
                "wall_seconds": time.perf_counter() - started,
            }
        )
        print(
            f"seed={seed}: referenced={len(referenced)} new={store.new_subsets} "
            f"fits={5 * store.new_subsets}",
            flush=True,
        )

    analysis = ROOT / "analysis"
    analysis.mkdir()
    _write_csv(analysis / "old_reachable_subsets.csv", reachable_rows)
    _write_csv(analysis / "support_gaps.csv", gap_rows)
    _write_csv(analysis / "replay_runs.csv", replay_rows)
    _write_csv(analysis / "budget_curves.csv", curve_rows)
    _write_csv(analysis / "coverage_regression.csv", coverage_rows)
    _write_csv(analysis / "costs.csv", cost_rows)

    curves = pd.DataFrame(curve_rows)
    budget_summary = (
        curves.groupby(["k", "phase", "method", "unique_subset_budget"], as_index=False)
        .agg(
            n_runs=("best_so_far_score", "size"),
            best_so_far_mean=("best_so_far_score", "mean"),
            best_so_far_sd=("best_so_far_score", "std"),
            gain_mean=("best_so_far_gain", "mean"),
            gain_sd=("best_so_far_gain", "std"),
        )
        .sort_values(["k", "phase", "method", "unique_subset_budget"])
    )
    _write_csv(analysis / "budget_summary.csv", budget_summary.to_dict("records"))
    runs = pd.DataFrame(replay_rows)
    replay_summary = (
        runs.groupby(["k", "phase", "method"], as_index=False)
        .agg(
            n_runs=("final_best_score", "size"),
            final_best_mean=("final_best_score", "mean"),
            final_best_sd=("final_best_score", "std"),
            final_gain_mean=("final_best_gain", "mean"),
            final_gain_sd=("final_best_gain", "std"),
            requests_mean=("requests", "mean"),
            cache_hits_mean=("cache_hits", "mean"),
        )
        .sort_values(["k", "phase", "method"])
    )
    _write_csv(analysis / "replay_summary.csv", replay_summary.to_dict("records"))
    pivot = runs.pivot(
        index=["seed", "k", "phase", "replay_seed"], columns="method", values="final_best_score"
    ).reset_index()
    pivot["quality_plus_global_minus_old"] = pivot["quality_plus_global"] - pivot["old_mask"]
    pivot["quality_plus_global_minus_uniform"] = (
        pivot["quality_plus_global"] - pivot["uniform_legal"]
    )
    _write_csv(analysis / "paired_final_differences.csv", pivot.to_dict("records"))

    total_wall = time.perf_counter() - started_all
    _write_json(
        ROOT / "complete.json",
        {
            "version": "action-support-v1",
            "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "origins": len(gap_rows),
            "old_reachable_rows": len(reachable_rows),
            "replay_runs": len(replay_rows),
            "budget_curve_rows": len(curve_rows),
            "coverage_regression_rows": len(coverage_rows),
            "k16_improved_endpoints": sum(
                row["case_type"] == "k16_improved_endpoint" for row in coverage_rows
            ),
            "low_mi_regression_cases": sum(
                row["case_type"] == "low_mi_conditional_contribution" for row in coverage_rows
            ),
            "all_coverage_regressions_passed": all(row["passed"] for row in coverage_rows),
            "source_test_open_count": 0,
            "rl_training_count": 0,
            "physical_new_unique_subsets": sum(row["new_unique_scored_subsets"] for row in cost_rows),
            "physical_classifier_fits": sum(row["physical_classifier_fits"] for row in cost_rows),
            "wall_seconds": total_wall,
            "historical_search_policy_gap_status": "NA_no_matched_historical_policy",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "run"), required=True)
    parser.add_argument("--jobs", type=int, default=24)
    args = parser.parse_args()
    if args.stage == "smoke":
        smoke()
    else:
        with parallel_config(backend="loky", inner_max_num_threads=1):
            run(args.jobs)


if __name__ == "__main__":
    main()
