#!/usr/bin/env python3
"""Prespecified deletion controls separate removal gains from low-MI addition gains."""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from sklearn.tree import DecisionTreeClassifier

from run_search_diagnosis import ROOT, fit_validation, load_raw, read, read_lines, sha, write_lines
from run_unified_baselines import _write_csv, _write_json


def score_control(raw, labels, context, subset):
    started = time.perf_counter()
    cols = np.array(context["final_feature_ids"])[list(subset)] - 1
    scores = []
    for fold in context["fold_original_rows"]:
        tr, va = fold["fit"], fold["held_out"]
        clf = DecisionTreeClassifier(random_state=context["cv_tree_random_state"])
        clf.fit(raw[tr][:, cols], labels[tr])
        scores.append(float(np.mean(clf.predict(raw[va][:, cols]) == labels[va])))
    elapsed = time.perf_counter() - started
    tr, va = context["development_original_rows"], context["validation_original_rows"]
    v = fit_validation(
        raw[tr][:, cols], labels[tr], raw[va][:, cols], labels[va], list(range(len(cols))), context["seed"]
    )
    return {
        "clean_indices_0based": list(subset),
        "original_feature_ids_1based": (cols + 1).tolist(),
        "search_accuracy": float(np.mean(scores)),
        "fold_scores": scores,
        "search_fit_count": 5,
        "search_fit_seconds": elapsed,
        **v,
    }


def main():
    out = ROOT / "ablation"
    if (out / "complete.json").exists():
        return
    out.mkdir(exist_ok=True)
    raw, labels = load_raw()
    results, rows = [], []
    for seed in (42, 43, 44, 45, 46):
        case = ROOT / "search" / f"nested_dev-seed-{seed}"
        context, complete = read(case / "context.json"), read(case / "complete.json")
        candidates, groups = read_lines(case / "candidates.jsonl"), read(case / "groups.json")["groups"]
        val = {
            r["candidate_id"]: r for r in read_lines(ROOT / "validation" / f"seed-{seed}" / "scores.jsonl")
        }
        subsets = sorted(
            {
                tuple(j for j in candidates[i]["clean_indices_0based"] if j != r)
                for name in ("starts", "terminals")
                for i in complete[name].values()
                for r in candidates[i]["clean_indices_0based"]
            }
        )
        with parallel_config(backend="loky", inner_max_num_threads=1):
            computed = Parallel(n_jobs=24)(delayed(score_control)(raw, labels, context, s) for s in subsets)
        controls = dict(zip(subsets, computed))
        results.extend(dict(seed=seed, **c) for c in computed)
        for k in (8, 16, 32):
            for phase, key in [("forward", "starts"), ("local", "terminals")]:
                anchor = complete[key][str(k)]
                s = set(candidates[anchor]["clean_indices_0based"])
                group = next(g for g in groups if g["kind"] == "single" and g["anchor_id"] == anchor)
                for m in group["members"]:
                    i = m["candidate_id"]
                    target = set(candidates[i]["clean_indices_0based"])
                    added = next(iter(target - s))
                    if context["mi_rank"][added] <= 33:
                        continue
                    control = controls[tuple(sorted(s & target))]
                    rows.append(
                        {
                            "seed": seed,
                            "k": k,
                            "phase": phase,
                            "candidate_id": i,
                            "anchor_id": anchor,
                            "ppo_covered": m["ppo_covered"],
                            "added_original_id": context["final_feature_ids"][added],
                            "removed_original_id": context["final_feature_ids"][next(iter(s - target))],
                            "added_mi_rank": context["mi_rank"][added],
                            "net_search_gain": candidates[i]["search_accuracy"]
                            - candidates[anchor]["search_accuracy"],
                            "net_lr_gain": val[i]["lr_bacc"] - val[anchor]["lr_bacc"],
                            "conditional_search_gain": candidates[i]["search_accuracy"]
                            - control["search_accuracy"],
                            "conditional_lr_gain": val[i]["lr_bacc"] - control["lr_bacc"],
                        }
                    )
        print(f"ablation seed={seed}: {len(computed)} controls", flush=True)
    write_lines(out / "controls.jsonl", results)
    _write_csv(out / "low_mi_contributions.csv", rows)
    _write_json(
        out / "complete.json",
        {
            "unique_controls": len(results),
            "search_fits": 5 * len(results),
            "validation_fits": 2 * len(results),
            "code_hash": sha(Path(__file__)),
            "protocol_hash": sha("documents/research-plan/search-diagnosis-ablation-addendum.md"),
            "used_for_search": False,
        },
    )
    df = pd.DataFrame(rows)
    effective = df[(df.net_search_gain > 1e-12) & (df.net_lr_gain > 1e-12)]
    strong = effective[(effective.conditional_search_gain > 1e-12) & (effective.conditional_lr_gain > 1e-12)]
    print("Low-MI conditional contribution confirmed within development:")
    print(strong.groupby(["k", "phase"]).agg(count=("candidate_id", "count"), covered=("ppo_covered", "sum")))
    print(strong.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
