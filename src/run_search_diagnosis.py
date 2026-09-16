#!/usr/bin/env python3
"""Frozen, train-only local-search diagnosis; never reads source-test or trains RL."""

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

import numpy as np
import sklearn
from joblib import Parallel, delayed, parallel_config
from scipy.stats import kendalltau, spearmanr
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from data.loader import _find_unique_columns
from radar_ship_fs.ppo.config import ExperimentConfig
from radar_ship_fs.ppo.ppo_env import FeatureSelectionEnv
from radar_ship_fs.ppo.ppo_graph import build_feature_graph
from run_unified_baselines import _fit_fold_scores, _write_csv, _write_json, build_seed_context

ROOT = Path("experiments/search_diagnosis_v1")
TRAIN = Path("../dataset/sim_ship_cr_v16n_2x_noise.train.svm")
BASE = Path("experiments/unified_strong_baselines_v1/selection")
SEEDS = (42, 43, 44, 45, 46)
KS = (8, 16, 32)
EPS = 1e-12


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_lines(path, rows):
    with Path(path).open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def load_raw():
    assert sha(TRAIN) == "2191fdc297aa49ce6a700df956ec44f68dc13e4c737937b31167543a203d22bb"
    X, y = load_svmlight_file(TRAIN, n_features=75)
    return X.toarray().astype(np.float32), y.astype(np.int64)


def clean(X):
    constant = np.isclose(X.min(axis=0), X.max(axis=0))
    nonconstant = np.flatnonzero(~constant)
    unique, duplicates = _find_unique_columns(X[:, nonconstant])
    ids = nonconstant[unique] + 1
    return ids, {
        "constant_feature_ids": (np.flatnonzero(constant) + 1).tolist(),
        "duplicate_feature_mapping": {
            str(nonconstant[a] + 1): int(nonconstant[b] + 1) for a, b in duplicates.items()
        },
    }


def fit_search(context, subset):
    t = time.perf_counter()
    score, folds = _fit_fold_scores(context.X, context.y, context.folds, subset, context.cv_tree_random_state)
    return score, list(folds), time.perf_counter() - t


class Ledger:
    def __init__(self, context, ids, graph, jobs):
        self.context, self.ids, self.graph, self.jobs = context, ids, graph, jobs
        self.cache, self.events = {}, []
        self.fits = 0

    def score(self, subsets, *, source, anchor=None):
        keys = [tuple(sorted(s)) for s in subsets]
        for s in keys:
            assert len(s) == len(set(s)) and len(s) > 0 and 0 <= min(s) <= max(s) < len(self.ids)
        missing = list(dict.fromkeys(s for s in keys if s not in self.cache))
        results = Parallel(n_jobs=self.jobs)(delayed(fit_search)(self.context, s) for s in missing)
        fresh = set(missing)
        for s, (score, folds, elapsed) in zip(missing, results):
            mask = np.zeros(len(self.ids), dtype=bool)
            mask[list(s)] = True
            redundancy = self.graph.redundancy(mask)
            self.cache[s] = {
                "id": len(self.cache),
                "clean_indices_0based": list(s),
                "original_feature_ids_1based": self.ids[list(s)].tolist(),
                "coordinate_crosscheck_passed": True,
                "search_accuracy": score,
                "fold_scores": folds,
                "redundancy": redundancy,
                "search_J": score - 0.02 * redundancy,
                "search_fit_count": 5,
                "search_fit_seconds": elapsed,
            }
            self.fits += 5
        for s in keys:
            new = s in fresh
            fresh.discard(s)
            self.events.append(
                {
                    "candidate_id": self.cache[s]["id"],
                    "source": source,
                    "anchor_id": self.cache[anchor]["id"] if anchor is not None else None,
                    "gain_accuracy": self.cache[s]["search_accuracy"] - self.cache[anchor]["search_accuracy"]
                    if anchor is not None
                    else None,
                    "gain_J": self.cache[s]["search_J"] - self.cache[anchor]["search_J"]
                    if anchor is not None
                    else None,
                    "actual_new_fit_count": 5 if new else 0,
                    "cache_hit": not new,
                }
            )
        return [self.cache[s]["search_accuracy"] for s in keys]


def swaps(s, d):
    return [tuple(sorted(set(s) - {r} | {a})) for r in s for a in range(d) if a not in s]


def best(subsets, scores):
    return subsets[int(np.argmax(scores))]


def env_for(graph, s):
    cfg = ExperimentConfig(
        feature_budget=len(s),
        min_features=min(4, len(s)),
        feature_id_node_feature=False,
        feature_id_reward_weight=0,
        sparsity_bonus=0,
        swap_candidate_pool=4,
        max_swaps=2,
    )
    mask = np.zeros(graph.n_features, dtype=bool)
    mask[list(s)] = True
    return FeatureSelectionEnv(graph, None, cfg, baseline_objective=0, initial_mask=mask)


def covered(graph, s, target):
    removed, added = sorted(set(s) - set(target)), sorted(set(target) - set(s))
    if len(removed) != 1:
        raise ValueError("single swap required")
    env = env_for(graph, s)
    rem_ok = bool(env.observation().valid_actions[removed[0]])
    env._take_feature_action(removed[0])
    add_ok = bool(env.observation().valid_actions[added[0]])
    return rem_ok, add_ok


def double_covered(graph, s, target):
    import itertools

    for rs in itertools.permutations(sorted(set(s) - set(target))):
        for adds in itertools.permutations(sorted(set(target) - set(s))):
            env = env_for(graph, s)
            ok = True
            for r, a in zip(rs, adds):
                for action in (r, a):
                    if not env.observation().valid_actions[action]:
                        ok = False
                        break
                    env._take_feature_action(action)
                if not ok:
                    break
            if ok:
                return True
    return False


def add_neighborhood(ledger, groups, s, source, k):
    neighbors = swaps(s, len(ledger.ids))
    scores = ledger.score(neighbors, source=source, anchor=s)
    rows = []
    for target in neighbors:
        rem, add = covered(ledger.graph, s, target)
        rows.append(
            {
                "candidate_id": ledger.cache[target]["id"],
                "remove_pool": rem,
                "add_pool": add,
                "ppo_covered": rem and add,
            }
        )
    groups.append(
        {"name": source, "k": k, "anchor_id": ledger.cache[s]["id"], "kind": "single", "members": rows}
    )
    return neighbors, scores


def add_double(ledger, groups, s, neighbors, scores, k, seed):
    assert max(scores) <= ledger.cache[s]["search_accuracy"] + EPS
    bridges = sorted(range(len(neighbors)), key=lambda i: -scores[i])[:4]
    rng = np.random.default_rng(seed * 100 + k)
    candidates, members = [], []
    for b in bridges:
        bridge = neighbors[b]
        possibilities = [
            tuple(sorted(set(bridge) - {r} | {a}))
            for r in s
            if r in bridge
            for a in range(len(ledger.ids))
            if a not in s and a not in bridge
        ]
        for j in rng.choice(len(possibilities), size=min(128, len(possibilities)), replace=False):
            target = possibilities[int(j)]
            assert len(set(s) - set(target)) == len(set(target) - set(s)) == 2
            candidates.append(target)
            members.append(
                {
                    "bridge_id": ledger.cache[bridge]["id"],
                    "bridge_gain": scores[b] - ledger.cache[s]["search_accuracy"],
                    "ppo_covered": double_covered(ledger.graph, s, target),
                }
            )
    values = ledger.score(candidates, source=f"k{k}/bounded_double", anchor=s)
    for target, member in zip(candidates, members):
        member["candidate_id"] = ledger.cache[target]["id"]
    groups.append(
        {
            "name": f"k{k}/bounded_double",
            "k": k,
            "kind": "double",
            "anchor_id": ledger.cache[s]["id"],
            "members": members,
        }
    )
    chosen = best(candidates, values)
    return chosen if max(values) > ledger.cache[s]["search_accuracy"] + EPS else s


def selection_case(mode, seed, jobs):
    out = ROOT / "search" / f"{mode}-seed-{seed}"
    if (out / "complete.json").exists():
        return
    if out.exists():
        raise RuntimeError(f"Partial output refuses overwrite: {out}")
    out.mkdir(parents=True)
    start = time.perf_counter()
    raw, y = load_raw()
    if mode == "original":
        fit_rows, val_rows = np.arange(len(y)), np.array([], dtype=int)
    else:
        fit_rows, val_rows = train_test_split(
            np.arange(len(y)), test_size=0.25, stratify=y, random_state=10000 + seed
        )
    ids, cleaning = clean(raw[fit_rows])
    assert len(ids) == 65
    context = build_seed_context(
        raw[fit_rows][:, ids - 1], y[fit_rows], seed=seed, validation_fraction=0.25, n_splits=5
    )
    graph = build_feature_graph(
        context.X,
        context.y,
        np.zeros(len(ids), dtype=int),
        threshold=0.8,
        seed=seed,
        tree_seed=seed,
        include_feature_id_node_feature=False,
    )
    mi_order = np.argsort(-graph.mutual_information, kind="stable")
    mi_rank = np.empty(len(ids), dtype=int)
    mi_rank[mi_order] = np.arange(1, len(ids) + 1)
    payload = {
        "mode": mode,
        "seed": seed,
        "fit_original_rows": fit_rows.tolist(),
        "validation_original_rows": val_rows.tolist(),
        "final_feature_ids": ids.tolist(),
        "cleaning": cleaning,
        "cleaning_fit_scope": "search_train_only",
        "split_random_state": context.split_random_state,
        "outer_split_random_state": 10000 + seed if mode == "nested_dev" else None,
        "cv_tree_random_state": context.cv_tree_random_state,
        "mi_random_state": seed,
        "inherited_context_mi_random_state": context.mi_random_state,
        "development_original_rows": fit_rows[context.development_original_rows].tolist(),
        "fold_original_rows": [
            {
                "fit": fit_rows[context.development_original_rows[a]].tolist(),
                "held_out": fit_rows[context.development_original_rows[b]].tolist(),
            }
            for a, b in context.folds
        ],
        "mi_values": graph.mutual_information.tolist(),
        "mi_rank": mi_rank.tolist(),
        "pool_quality": graph.static_node_features[:, :2].mean(axis=1).tolist(),
    }
    _write_json(out / "context.json", payload)
    ledger = Ledger(context, ids, graph, jobs)
    groups, starts, terminals, double_ends, original_checks = [], {}, {}, {}, []
    if mode == "nested_dev":
        current = ()
        for step in range(1, 33):
            candidates = [tuple(sorted((*current, a))) for a in range(len(ids)) if a not in current]
            scores = ledger.score(candidates, source=f"forward/step{step}", anchor=current or None)
            current = best(candidates, scores)
            if step in KS:
                starts[step] = current
        print(f"{mode} seed={seed}: forward complete", flush=True)
    else:
        old_context = read(BASE / f"seed-{seed}" / "context.json")
        assert old_context["cv_tree_random_state"] == context.cv_tree_random_state
        for k in KS:
            path = BASE / f"seed-{seed}" / f"forward_greedy_single_swap__fixed__k{k}" / "result.json"
            old = read(path)
            starts[k] = tuple(old["clean_indices_0based"])
            value = ledger.score([starts[k]], source=f"k{k}/imported_start")[0]
            assert abs(value - old["inner_cv_accuracy"]) < EPS
            assert ids[list(starts[k])].tolist() == old["original_feature_ids_1based"]
            original_checks.append(
                {"path": str(path), "sha256": sha(path), "score_error": abs(value - old["inner_cv_accuracy"])}
            )
    for k in KS:
        current = starts[k]
        for rnd in range(32):
            neighbors, scores = add_neighborhood(ledger, groups, current, f"k{k}/single_round{rnd}", k)
            chosen = best(neighbors, scores)
            if max(scores) <= ledger.cache[current]["search_accuracy"] + EPS:
                break
            if mode == "original":
                raise AssertionError("Frozen baseline no longer locally optimal")
            current = chosen
        else:
            raise RuntimeError("32-round cap hit; cannot launch terminal double search")
        terminals[k] = current
        double_ends[k] = add_double(ledger, groups, current, neighbors, scores, k, seed)
        print(
            f"{mode} seed={seed} K={k}: single rounds={rnd + 1}, "
            f"fits={ledger.fits}, wall={time.perf_counter() - start:.1f}s",
            flush=True,
        )
    write_lines(out / "candidates.jsonl", list(ledger.cache.values()))
    write_lines(out / "requests.jsonl", ledger.events)
    _write_json(out / "groups.json", {"groups": groups})
    summary = {
        "mode": mode,
        "seed": seed,
        "starts": {k: ledger.cache[s]["id"] for k, s in starts.items()},
        "terminals": {k: ledger.cache[s]["id"] for k, s in terminals.items()},
        "double_ends": {k: ledger.cache[s]["id"] for k, s in double_ends.items()},
        "original_checks": original_checks,
        "unique_candidates": len(ledger.cache),
        "requests": len(ledger.events),
        "cache_hits": sum(e["cache_hit"] for e in ledger.events),
        "search_fit_count": ledger.fits,
        "mi_fit_count": 1,
        "graph_dt_fit_count": 1,
        "search_sum_fit_seconds": sum(x["search_fit_seconds"] for x in ledger.cache.values()),
        "wall_seconds": time.perf_counter() - start,
    }
    _write_json(out / "complete.json", summary)


def fit_validation(X, y, V, vy, subset, seed):
    t = time.perf_counter()
    lr = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0, solver="liblinear", max_iter=5000, class_weight="balanced", random_state=seed
        ),
    )
    lr.fit(X[:, subset], y)
    lp = lr.predict(V[:, subset])
    dt = DecisionTreeClassifier(random_state=seed).fit(X[:, subset], y)
    dp = dt.predict(V[:, subset])
    return {
        "lr_bacc": float(balanced_accuracy_score(vy, lp)),
        "lr_accuracy": float(accuracy_score(vy, lp)),
        "dt_bacc": float(balanced_accuracy_score(vy, dp)),
        "dt_accuracy": float(accuracy_score(vy, dp)),
        "validation_fit_count": 2,
        "validation_fit_seconds": time.perf_counter() - t,
    }


def validation_case(seed, jobs):
    case = ROOT / "search" / f"nested_dev-seed-{seed}"
    out = ROOT / "validation" / f"seed-{seed}"
    if (out / "complete.json").exists():
        return
    if out.exists():
        raise RuntimeError(f"Partial validation refuses overwrite: {out}")
    out.mkdir(parents=True)
    context, complete = read(case / "context.json"), read(case / "complete.json")
    candidates = read_lines(case / "candidates.jsonl")
    groups = read(case / "groups.json")["groups"]
    requested = {g["anchor_id"] for g in groups}
    requested.update(m["candidate_id"] for g in groups for m in g["members"])
    for name in ("starts", "terminals", "double_ends"):
        requested.update(complete[name].values())
    ids = sorted(requested)
    raw, y = load_raw()
    cols = np.array(context["final_feature_ids"]) - 1
    fit, val = context["development_original_rows"], context["validation_original_rows"]
    assert not set(fit) & set(val)
    X, V = raw[fit][:, cols], raw[val][:, cols]
    results = Parallel(n_jobs=jobs)(
        delayed(fit_validation)(X, y[fit], V, y[val], candidates[i]["clean_indices_0based"], seed)
        for i in ids
    )
    rows = [dict(candidate_id=i, **r) for i, r in zip(ids, results)]
    write_lines(out / "scores.jsonl", rows)
    _write_json(
        out / "complete.json",
        {
            "seed": seed,
            "validated_candidates": len(rows),
            "validation_fit_count": 2 * len(rows),
            "sum_fit_seconds": sum(r["validation_fit_seconds"] for r in rows),
        },
    )
    print(f"validation seed={seed}: {len(rows)} candidates, {2 * len(rows)} fits", flush=True)


def sign(x):
    return np.where(x > EPS, 1, np.where(x < -EPS, -1, 0))


def analyze():
    ranking, decisions, coverage = [], [], []
    for seed in SEEDS:
        case = ROOT / "search" / f"nested_dev-seed-{seed}"
        context, complete = read(case / "context.json"), read(case / "complete.json")
        candidates = read_lines(case / "candidates.jsonl")
        validation = {
            r["candidate_id"]: r for r in read_lines(ROOT / "validation" / f"seed-{seed}" / "scores.jsonl")
        }
        groups = read(case / "groups.json")["groups"]
        for g in groups:
            anchor = g["anchor_id"]
            members = {r["candidate_id"]: r for r in g["members"]}
            ids = list(members)
            for search in ("search_accuracy", "search_J"):
                s = np.array([candidates[i][search] for i in ids])
                ds = s - candidates[anchor][search]
                top_n = max(1, int(np.ceil(0.1 * len(ids))))
                st = set(np.argsort(-s, kind="stable")[:top_n])
                for metric in ("lr_bacc", "dt_accuracy", "dt_bacc"):
                    v = np.array([validation[i][metric] for i in ids])
                    dv = v - validation[anchor][metric]
                    good = [i for i in np.argsort(-v, kind="stable") if dv[i] > EPS][:top_n]
                    ranking.append(
                        {
                            "seed": seed,
                            "k": g["k"],
                            "group": g["name"],
                            "kind": g["kind"],
                            "search": search,
                            "validation": metric,
                            "n": len(ids),
                            "spearman": float(spearmanr(s, v).statistic),
                            "kendall_tau_b": float(kendalltau(s, v).statistic),
                            "sign_agreement": float(np.mean(sign(ds) == sign(dv))),
                            "search_positive": int(np.sum(ds > EPS)),
                            "both_positive": int(np.sum((ds > EPS) & (dv > EPS))),
                            "positive_precision": float(np.mean(dv[ds > EPS] > EPS))
                            if np.any(ds > EPS)
                            else None,
                            "validation_good_count": len(good),
                            "good_retained_count": len(st & set(good)),
                            "good_recall": len(st & set(good)) / len(good) if good else None,
                        }
                    )
            for metric in ("lr_bacc", "dt_accuracy"):
                for i in ids:
                    c, a, v, av = candidates[i], candidates[anchor], validation[i], validation[anchor]
                    added = sorted(set(c["clean_indices_0based"]) - set(a["clean_indices_0based"]))
                    removed = sorted(set(a["clean_indices_0based"]) - set(c["clean_indices_0based"]))
                    coverage.append(
                        {
                            "seed": seed,
                            "k": g["k"],
                            "group": g["name"],
                            "kind": g["kind"],
                            "candidate_id": i,
                            "anchor_id": anchor,
                            "metric": metric,
                            "search_gain": c["search_accuracy"] - a["search_accuracy"],
                            "J_gain": c["search_J"] - a["search_J"],
                            "validation_gain": v[metric] - av[metric],
                            "ppo_covered": members[i]["ppo_covered"],
                            "remove_pool": members[i].get("remove_pool"),
                            "add_pool": members[i].get("add_pool"),
                            "added_original_ids": json.dumps(
                                [context["final_feature_ids"][j] for j in added]
                            ),
                            "removed_original_ids": json.dumps(
                                [context["final_feature_ids"][j] for j in removed]
                            ),
                            "added_mi_ranks": json.dumps([context["mi_rank"][j] for j in added]),
                            "any_added_low_mi": any(context["mi_rank"][j] > 33 for j in added),
                        }
                    )
        for k in KS:
            s, t, d = (complete[name][str(k)] for name in ("starts", "terminals", "double_ends"))
            rules = [("forward_to_local", s, t), ("local_to_double", t, d)]
            for tag, a in [("forward", s), ("local", t)]:
                group = next(g for g in groups if g["kind"] == "single" and g["anchor_id"] == a)
                ids = list(dict.fromkeys(m["candidate_id"] for m in group["members"]))
                for search in ("search_accuracy", "search_J"):
                    chosen = max(ids, key=lambda i: candidates[i][search])
                    if candidates[chosen][search] <= candidates[a][search] + EPS:
                        chosen = a
                    rules.append((f"{tag}_best_{search}", a, chosen))
            for rule, a, b in rules:
                decisions.append(
                    {
                        "seed": seed,
                        "k": k,
                        "rule": rule,
                        "anchor_id": a,
                        "chosen_id": b,
                        "changed": a != b,
                        **{
                            key + "_gain": candidates[b][key] - candidates[a][key]
                            for key in ("search_accuracy", "search_J")
                        },
                        **{
                            key + "_gain": validation[b][key] - validation[a][key]
                            for key in ("lr_bacc", "dt_accuracy", "dt_bacc")
                        },
                    }
                )
    out = ROOT / "analysis"
    out.mkdir(exist_ok=True)
    _write_csv(out / "ranking.csv", ranking)
    _write_csv(out / "decisions.csv", decisions)
    _write_csv(out / "coverage.csv", coverage)
    _write_json(
        out / "complete.json",
        {"ranking_rows": len(ranking), "decision_rows": len(decisions), "coverage_rows": len(coverage)},
    )


def prepare():
    sources = [
        Path(__file__),
        Path("documents/research-plan/search-diagnosis-protocol.md"),
        Path("src/run_unified_baselines.py"),
        Path("src/radar_ship_fs/ppo/ppo_env.py"),
        Path("src/radar_ship_fs/ppo/ppo_graph.py"),
        Path("src/data/loader.py"),
        Path("configs/v16n/research_baseline.toml"),
    ]
    hashes = {str(p): sha(p) for p in sources}
    if (ROOT / "manifest.json").exists():
        assert read(ROOT / "manifest.json")["code_protocol_hashes"] == hashes, "Frozen source changed"
        return
    ROOT.mkdir(parents=True, exist_ok=False)
    _write_json(
        ROOT / "manifest.json",
        {
            "version": "search-diagnosis-v1",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "code_protocol_hashes": hashes,
            "train_sha256": sha(TRAIN),
            "source_test_open_count": 0,
            "rl_training_count": 0,
            "python": sys.version,
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "hardware": platform.platform(),
            "cpu_count": os.cpu_count(),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "git_status": subprocess.check_output(["git", "status", "--short"], text=True),
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["search", "validate", "analyze"], required=True)
    parser.add_argument("--jobs", type=int, default=32)
    args = parser.parse_args()
    prepare()
    with parallel_config(backend="loky", inner_max_num_threads=1):
        if args.stage == "search":
            for mode in ("original", "nested_dev"):
                for seed in SEEDS:
                    selection_case(mode, seed, args.jobs)
            files = sorted((ROOT / "search").rglob("*"))
            _write_json(
                ROOT / "search-complete.json",
                {
                    "files": {str(p): sha(p) for p in files if p.is_file()},
                    "all_candidates_frozen_before_validation": True,
                },
            )
        elif args.stage == "validate":
            frozen = read(ROOT / "search-complete.json")
            assert all(sha(p) == h for p, h in frozen["files"].items())
            for seed in SEEDS:
                validation_case(seed, args.jobs)
        else:
            analyze()


if __name__ == "__main__":
    main()
