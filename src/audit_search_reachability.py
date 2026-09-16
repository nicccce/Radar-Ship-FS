#!/usr/bin/env python3
"""Enumerate every legal PPO path of at most two swaps, without training a policy."""

import numpy as np
import pandas as pd

from radar_ship_fs.ppo.ppo_graph import FeatureGraph
from run_search_diagnosis import ROOT, env_for, read, read_lines
from run_unified_baselines import _write_csv


def graph_from_context(context):
    q = np.asarray(context["pool_quality"], dtype=np.float32)
    d = len(q)
    identity = np.eye(d, dtype=np.float32)
    # Only the mean of first two static columns enters the action mask.
    return FeatureGraph(
        identity, identity, identity, identity, np.column_stack([q, q]), q, q, np.arange(d), 0.8, 0, 0
    )


def next_states(graph, subset):
    env = env_for(graph, subset)
    removes = np.flatnonzero(env.observation().valid_actions[: graph.n_features])
    reached = set()
    for remove in removes:
        env = env_for(graph, subset)
        env._take_feature_action(int(remove))
        adds = np.flatnonzero(env.observation().valid_actions[: graph.n_features])
        for add in adds:
            reached.add(tuple(sorted(set(subset) - {int(remove)} | {int(add)})))
    assert len(reached) == 16
    return reached


def main():
    rows = []
    endpoints = []
    for seed in (42, 43, 44, 45, 46):
        case = ROOT / "search" / f"nested_dev-seed-{seed}"
        context, complete = read(case / "context.json"), read(case / "complete.json")
        graph = graph_from_context(context)
        candidates = read_lines(case / "candidates.jsonl")
        validation = {
            r["candidate_id"]: r for r in read_lines(ROOT / "validation" / f"seed-{seed}" / "scores.jsonl")
        }
        groups = read(case / "groups.json")["groups"]
        for k in (8, 16, 32):
            for phase, key in [("forward", "starts"), ("local", "terminals")]:
                anchor = complete[key][str(k)]
                s = tuple(candidates[anchor]["clean_indices_0based"])
                once = next_states(graph, s)
                twice = once | {s}
                for t in once:
                    twice.update(next_states(graph, t))
                if phase == "forward":
                    end = complete["terminals"][str(k)]
                    target = tuple(candidates[end]["clean_indices_0based"])
                    rounds = sum(g["kind"] == "single" and g["k"] == k for g in groups)
                    endpoints.append(
                        {
                            "seed": seed,
                            "k": k,
                            "accepted_swaps": rounds - 1,
                            "endpoint_distance": len(set(s) - set(target)),
                            "changed": end != anchor,
                            "within_two_reachable": target in twice,
                            "lr_gain": validation[end]["lr_bacc"] - validation[anchor]["lr_bacc"],
                            "dt_gain": validation[end]["dt_accuracy"] - validation[anchor]["dt_accuracy"],
                            "J_gain": candidates[end]["search_J"] - candidates[anchor]["search_J"],
                        }
                    )
                group = next(g for g in groups if g["kind"] == "single" and g["anchor_id"] == anchor)
                for member in group["members"]:
                    i = member["candidate_id"]
                    target = tuple(candidates[i]["clean_indices_0based"])
                    assert (target in once) == member["ppo_covered"]
                    gain = candidates[i]["search_accuracy"] - candidates[anchor]["search_accuracy"]
                    val_gain = validation[i]["lr_bacc"] - validation[anchor]["lr_bacc"]
                    added = set(target) - set(s)
                    rows.append(
                        {
                            "seed": seed,
                            "k": k,
                            "phase": phase,
                            "candidate_id": i,
                            "anchor_id": anchor,
                            "search_gain": gain,
                            "lr_bacc_gain": val_gain,
                            "effective": gain > 1e-12 and val_gain > 1e-12,
                            "low_mi": any(context["mi_rank"][j] > 33 for j in added),
                            "one_swap_reachable": target in once,
                            "within_two_swaps_reachable": target in twice,
                            "all_one_swap_endpoints": len(once),
                            "all_within_two_swap_endpoints": len(twice),
                        }
                    )
    _write_csv(ROOT / "analysis" / "reachability.csv", rows)
    _write_csv(ROOT / "analysis" / "endpoint_reachability.csv", endpoints)
    df = pd.DataFrame(rows)
    print(
        df[df.effective]
        .groupby(["k", "phase"])
        .agg(
            effective=("candidate_id", "count"),
            one_swap=("one_swap_reachable", "sum"),
            within_two=("within_two_swaps_reachable", "sum"),
            low_mi=("low_mi", "sum"),
        )
        .to_string()
    )
    print("Low MI effective:")
    print(
        df[df.effective & df.low_mi]
        .groupby(["k", "phase"])
        .agg(
            effective=("candidate_id", "count"),
            one_swap=("one_swap_reachable", "sum"),
            within_two=("within_two_swaps_reachable", "sum"),
        )
        .to_string()
    )


if __name__ == "__main__":
    main()
