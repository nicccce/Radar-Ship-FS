#!/usr/bin/env python3
"""Descriptive, paired development summaries; does not feed candidate generation."""

from math import comb

import pandas as pd

from run_search_diagnosis import ROOT, read, read_lines
from run_unified_baselines import _write_csv, _write_json


def main():
    rank = pd.read_csv(ROOT / "analysis" / "ranking.csv")
    cover = pd.read_csv(ROOT / "analysis" / "coverage.csv")
    dec = pd.read_csv(ROOT / "analysis" / "decisions.csv")
    terminal_groups = set()
    costs, originals, endpoints = [], [], []
    for case in sorted((ROOT / "search").iterdir()):
        c, context = read(case / "complete.json"), read(case / "context.json")
        candidates = read_lines(case / "candidates.jsonl")
        groups = read(case / "groups.json")["groups"]
        costs.append(
            {
                "case": case.name,
                **{
                    key: c[key]
                    for key in (
                        "unique_candidates",
                        "requests",
                        "cache_hits",
                        "search_fit_count",
                        "wall_seconds",
                        "search_sum_fit_seconds",
                    )
                },
            }
        )
        for k in (8, 16, 32):
            a, b, d = (c[key][str(k)] for key in ("starts", "terminals", "double_ends"))
            terminal = next(g for g in groups if g["kind"] == "single" and g["anchor_id"] == b)
            doubles = next(g for g in groups if g["kind"] == "double" and g["k"] == k)
            if context["mode"] == "nested_dev":
                terminal_groups.add((context["seed"], terminal["name"]))
                val = {
                    r["candidate_id"]: r
                    for r in read_lines(ROOT / "validation" / f"seed-{context['seed']}" / "scores.jsonl")
                }
                for name, i in [("forward", a), ("local", b), ("double", d)]:
                    endpoints.append(
                        {
                            "seed": context["seed"],
                            "k": k,
                            "stage": name,
                            "search_accuracy": candidates[i]["search_accuracy"],
                            **val[i],
                        }
                    )
            else:
                originals.append(
                    {
                        "seed": context["seed"],
                        "k": k,
                        "anchor_accuracy": candidates[a]["search_accuracy"],
                        "max_single_gain": max(
                            candidates[m["candidate_id"]]["search_accuracy"]
                            - candidates[a]["search_accuracy"]
                            for m in terminal["members"]
                        ),
                        "double_gain": candidates[d]["search_accuracy"] - candidates[b]["search_accuracy"],
                        "double_unique": len({m["candidate_id"] for m in doubles["members"]}),
                        "double_universe": comb(k, 2) * comb(65 - k, 2),
                    }
                )

    def phase(row):
        if row["kind"] == "double":
            return "double"
        if (row["seed"], row["group"]) in terminal_groups:
            return "local"
        if row["group"].endswith("single_round0"):
            return "forward"
        return "intermediate"

    # When forward already is terminal, include its neighborhood in both descriptive anchors.
    for frame in (rank, cover):
        frame["phase"] = frame.apply(phase, axis=1)
    rank_forward = rank[rank["group"].str.endswith("single_round0")].copy()
    rank_forward["phase"] = "forward"
    rank = pd.concat([rank[rank.phase != "forward"], rank_forward], ignore_index=True)
    cover_forward = cover[cover["group"].str.endswith("single_round0")].copy()
    cover_forward["phase"] = "forward"
    cover = pd.concat([cover[cover.phase != "forward"], cover_forward], ignore_index=True)
    dsum = []
    for (k, rule), g in dec.groupby(["k", "rule"]):
        row = {"k": int(k), "rule": rule, "n": len(g), "changed": int(g.changed.sum())}
        for metric in (
            "search_accuracy_gain",
            "search_J_gain",
            "lr_bacc_gain",
            "dt_accuracy_gain",
            "dt_bacc_gain",
        ):
            row[metric + "_mean"] = g[metric].mean()
            row[metric + "_sd"] = g[metric].std(ddof=1)
            row[metric + "_positive"] = int((g[metric] > 1e-12).sum())
            row[metric + "_zero"] = int((g[metric].abs() <= 1e-12).sum())
            row[metric + "_negative"] = int((g[metric] < -1e-12).sum())
        row["stable_lr"] = bool(row["lr_bacc_gain_mean"] >= 0.002 and row["lr_bacc_gain_positive"] >= 4)
        dsum.append(row)
    rsum = []
    for key, g in rank.groupby(["k", "phase", "search", "validation"]):
        k, p, s, v = key
        row = {"k": int(k), "phase": p, "search": s, "validation": v, "n_groups": len(g)}
        for metric in ("spearman", "kendall_tau_b", "sign_agreement", "good_recall"):
            row[metric + "_mean"] = g[metric].mean()
        for metric in (
            "n",
            "search_positive",
            "both_positive",
            "validation_good_count",
            "good_retained_count",
        ):
            row[metric + "_sum"] = int(g[metric].sum())
        row["positive_precision_pooled"] = (
            g.both_positive.sum() / g.search_positive.sum() if g.search_positive.sum() else None
        )
        rsum.append(row)
    csum = []
    for (k, p, metric), g in cover.groupby(["k", "phase", "metric"]):
        effective = g[(g.search_gain > 1e-12) & (g.validation_gain > 1e-12)]
        low = effective[effective.any_added_low_mi]
        csum.append(
            {
                "k": int(k),
                "phase": p,
                "metric": metric,
                "candidate_exchanges": len(g),
                "effective": len(effective),
                "covered_effective": int(effective.ppo_covered.sum()),
                "low_mi_effective": len(low),
                "covered_low_mi_effective": int(low.ppo_covered.sum()),
                "excluded_remove": int((~effective.remove_pool.fillna(True).astype(bool)).sum()),
                "excluded_add": int((~effective.add_pool.fillna(True).astype(bool)).sum()),
            }
        )
    _write_csv(ROOT / "analysis" / "decision_summary.csv", dsum)
    _write_csv(ROOT / "analysis" / "ranking_summary.csv", rsum)
    _write_csv(ROOT / "analysis" / "coverage_summary.csv", csum)
    _write_csv(ROOT / "analysis" / "costs.csv", costs)
    _write_csv(ROOT / "analysis" / "original_summary.csv", originals)
    _write_csv(ROOT / "analysis" / "endpoints.csv", endpoints)
    examples = cover[
        (cover.phase == "forward")
        & (cover.metric == "lr_bacc")
        & (cover.search_gain > 1e-12)
        & (cover.validation_gain > 1e-12)
        & cover.any_added_low_mi
        & ~cover.ppo_covered
    ]
    _write_json(
        ROOT / "analysis" / "summary.json",
        {
            "decisions": dsum,
            "ranking": rsum,
            "coverage": csum,
            "low_mi_examples": examples.head(12).to_dict("records"),
        },
    )
    print("Decision rules (mean gains and positive splits):")
    print(
        pd.DataFrame(dsum)[
            [
                "k",
                "rule",
                "search_accuracy_gain_mean",
                "lr_bacc_gain_mean",
                "lr_bacc_gain_positive",
                "dt_accuracy_gain_mean",
                "stable_lr",
            ]
        ].to_string(index=False)
    )
    print("Forward neighborhood ranking:")
    print(
        pd.DataFrame(rsum).query("phase == 'forward' and search == 'search_accuracy'").to_string(index=False)
    )
    print("Forward and double coverage:")
    print(pd.DataFrame(csum).query("phase != 'intermediate'").to_string(index=False))
    print("Original double:")
    print(
        pd.DataFrame(originals)
        .groupby("k")
        .agg({"double_gain": ["mean", "max"], "max_single_gain": "max"})
        .to_string()
    )


if __name__ == "__main__":
    main()
