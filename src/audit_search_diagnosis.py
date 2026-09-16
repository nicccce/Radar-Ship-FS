#!/usr/bin/env python3
"""Independent structural audit and sampled estimator recomputation."""

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from run_search_diagnosis import ROOT, SEEDS, TRAIN, read, read_lines, sha
from run_unified_baselines import _write_json


def manual_bacc(y, pred):
    return np.mean([np.mean(pred[y == label] == label) for label in np.unique(y)])


def main():
    frozen = read(ROOT / "search-complete.json")
    assert all(sha(path) == digest for path, digest in frozen["files"].items())
    sparse, labels = load_svmlight_file(TRAIN, n_features=75)
    raw, labels = sparse.toarray().astype(np.float32), labels.astype(np.int64)
    audit = []
    search_errors, validation_errors = [], []
    for case in sorted((ROOT / "search").iterdir()):
        context = read(case / "context.json")
        complete = read(case / "complete.json")
        candidates = read_lines(case / "candidates.jsonl")
        events = read_lines(case / "requests.jsonl")
        groups = read(case / "groups.json")["groups"]
        ids = np.array(context["final_feature_ids"])
        fit, val = set(context["fit_original_rows"]), set(context["validation_original_rows"])
        assert not fit & val and fit | val == set(range(len(labels)))
        assert set(context["development_original_rows"]) == fit
        seen = []
        for fold in context["fold_original_rows"]:
            tr, va = set(fold["fit"]), set(fold["held_out"])
            assert not tr & va and tr | va == fit and not (tr | va) & val
            seen.extend(fold["held_out"])
        assert sorted(seen) == sorted(fit)
        assert len(candidates) == complete["unique_candidates"]
        assert sum(e["actual_new_fit_count"] for e in events) == 5 * len(candidates)
        assert sum(e["cache_hit"] for e in events) == complete["cache_hits"]
        assert complete["search_fit_count"] == 5 * len(candidates)
        assert len({tuple(c["clean_indices_0based"]) for c in candidates}) == len(candidates)
        for c in candidates:
            s = c["clean_indices_0based"]
            assert len(s) == len(set(s))
            assert ids[s].tolist() == c["original_feature_ids_1based"]
            assert abs(np.mean(c["fold_scores"]) - c["search_accuracy"]) < 1e-12
            assert abs(c["search_J"] - (c["search_accuracy"] - 0.02 * c["redundancy"])) < 1e-12
        for event in events:
            a = event["anchor_id"]
            if a is not None:
                for key, metric in [("gain_accuracy", "search_accuracy"), ("gain_J", "search_J")]:
                    assert (
                        abs(event[key] - (candidates[event["candidate_id"]][metric] - candidates[a][metric]))
                        < 1e-12
                    )
        for group in groups:
            anchor = set(candidates[group["anchor_id"]]["clean_indices_0based"])
            actual = [tuple(candidates[m["candidate_id"]]["clean_indices_0based"]) for m in group["members"]]
            if group["kind"] == "single":
                expected = {
                    tuple(sorted(anchor - {r} | {a}))
                    for r, a in itertools.product(anchor, set(range(len(ids))) - anchor)
                }
                assert set(actual) == expected and len(actual) == len(expected)
                assert sum(m["ppo_covered"] for m in group["members"]) == 16
            else:
                assert len(actual) <= 512
                for member, target in zip(group["members"], actual):
                    bridge = set(candidates[member["bridge_id"]]["clean_indices_0based"])
                    assert len(anchor - set(target)) == len(set(target) - anchor) == 2
                    assert len(anchor - bridge) == len(bridge - set(target)) == 1
                    assert member["bridge_gain"] <= 1e-12
        sampled = (
            set(complete["starts"].values())
            | set(complete["terminals"].values())
            | set(complete["double_ends"].values())
        )
        for i in sorted(sampled):
            c = candidates[i]
            cols = np.array(c["original_feature_ids_1based"]) - 1
            scores = []
            for fold in context["fold_original_rows"]:
                tr, va = fold["fit"], fold["held_out"]
                clf = DecisionTreeClassifier(random_state=context["cv_tree_random_state"])
                clf.fit(raw[tr][:, cols], labels[tr])
                scores.append(np.mean(clf.predict(raw[va][:, cols]) == labels[va]))
            search_errors.append(abs(np.mean(scores) - c["search_accuracy"]))
        validated = 0
        if context["mode"] == "nested_dev":
            vrows = read_lines(ROOT / "validation" / f"seed-{context['seed']}" / "scores.jsonl")
            vmap = {r["candidate_id"]: r for r in vrows}
            expected = {g["anchor_id"] for g in groups}
            expected.update(m["candidate_id"] for g in groups for m in g["members"])
            assert set(vmap) == expected | sampled
            assert sum(v["validation_fit_count"] for v in vrows) == 2 * len(vrows)
            tr, va = context["development_original_rows"], context["validation_original_rows"]
            for i in sorted(sampled):
                cols = np.array(candidates[i]["original_feature_ids_1based"]) - 1
                scaler = StandardScaler().fit(raw[tr][:, cols])
                lr = LogisticRegression(
                    C=1,
                    solver="liblinear",
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=context["seed"],
                )
                lr.fit(scaler.transform(raw[tr][:, cols]), labels[tr])
                pred = lr.predict(scaler.transform(raw[va][:, cols]))
                dt = DecisionTreeClassifier(random_state=context["seed"]).fit(raw[tr][:, cols], labels[tr])
                dp = dt.predict(raw[va][:, cols])
                validation_errors.extend(
                    [
                        abs(manual_bacc(labels[va], pred) - vmap[i]["lr_bacc"]),
                        abs(np.mean(dp == labels[va]) - vmap[i]["dt_accuracy"]),
                        abs(manual_bacc(labels[va], dp) - vmap[i]["dt_bacc"]),
                    ]
                )
            validated = len(vrows)
        audit.append(
            {
                "case": case.name,
                "all_groups_exhaustive_or_bounded": True,
                "coordinates_rows_and_costs_passed": True,
                "candidate_count": len(candidates),
                "sampled_search_recomputations": len(sampled),
                "validated_candidates": validated,
            }
        )
    assert max(search_errors) < 1e-12 and max(validation_errors) < 1e-12
    # Independent arithmetic check of every reported decision.
    decisions = pd.read_csv(ROOT / "analysis" / "decisions.csv")
    for seed in SEEDS:
        case = ROOT / "search" / f"nested_dev-seed-{seed}"
        candidates = read_lines(case / "candidates.jsonl")
        val = {
            r["candidate_id"]: r for r in read_lines(ROOT / "validation" / f"seed-{seed}" / "scores.jsonl")
        }
        for row in decisions[decisions.seed == seed].itertuples():
            a, b = row.anchor_id, row.chosen_id
            assert abs(row.lr_bacc_gain - (val[b]["lr_bacc"] - val[a]["lr_bacc"])) < 1e-12
            assert (
                abs(
                    row.search_accuracy_gain
                    - (candidates[b]["search_accuracy"] - candidates[a]["search_accuracy"])
                )
                < 1e-12
            )
    _write_json(
        ROOT / "audit.json",
        {
            "status": "passed_with_stated_statistical_limits",
            "cases": audit,
            "search_recompute_max_error": max(search_errors),
            "validation_recompute_max_error": max(validation_errors),
            "all_decisions_recomputed": len(decisions),
            "audit_script_sha256": sha(Path(__file__)),
            "test_data_used": False,
            "scope": "All structural/cost/coordinate invariants; sampled model refits; "
            "no claim of independent dataset replicates or global two-swap optimality.",
        },
    )
    print(json.dumps(read(ROOT / "audit.json"), indent=2))


if __name__ == "__main__":
    main()
