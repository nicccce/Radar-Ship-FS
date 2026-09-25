#!/usr/bin/env python3
"""09C-1 locked, train-only fold 2-4 accuracy validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

import run_block_rewrite as base
from run_stg_residual_rl import ranked_mask, train_stg

CONFIG = Path("configs/v16n/locked_accuracy_validation_v1.toml")
CODE_PATHS = (
    Path(__file__),
    Path("src/run_block_rewrite.py"),
    Path("src/run_stg_residual_rl.py"),
    Path("src/run_k32_robustness.py"),
    Path("src/radar_ship_fs/feature_mapping.py"),
    Path("src/radar_ship_fs/ppo/ppo_graph.py"),
)


def row_hash(rows):
    return hashlib.sha256(np.asarray(rows, dtype="<i8").tobytes()).hexdigest()


def protocol_hashes():
    return {str(path): base.sha256(path) for path in (CONFIG, *CODE_PATHS)}


def paths(config):
    root = Path(config["output"]["root"])
    return root, root / "manifest.json"


def check_partition(data, partition, labels, fold):
    train = data.train_rows
    held = data.validation_rows
    n = len(labels)
    if len(train) != len(set(train)) or len(held) != len(set(held)):
        raise AssertionError("duplicate outer row")
    if not set(train).isdisjoint(held) or set(train) | set(held) != set(range(n)):
        raise AssertionError("outer row overlap or omission")
    assigned = np.asarray(partition["validation_fold_by_original_row"], dtype=int)
    if len(assigned) != n or set(held) != set(np.flatnonzero(assigned == fold)):
        raise AssertionError("outer rows differ from frozen partition")
    if set(train) != set(np.flatnonzero(assigned != fold)):
        raise AssertionError("outer training rows differ from frozen partition")
    for split in data.inner_folds:
        fit, check = set(split["fit"]), set(split["held_out"])
        if not fit or not check or fit & check or fit | check != set(train):
            raise AssertionError("invalid inner split")
    if len(data.final_ids) != 65 or data.mapping.clean_feature_count != 65:
        raise AssertionError("wrong clean feature count")
    clean = tuple(range(65))
    original = data.mapping.clean_indices_to_original_ids_1based(clean)
    if data.mapping.original_ids_1based_to_clean_indices(original) != clean:
        raise AssertionError("feature coordinate round trip failed")
    if len(data.mi32) != 32 or len(set(data.mi32)) != 32:
        raise AssertionError("invalid MI32")
    return {
        "fold": fold,
        "train_rows": len(train),
        "validation_rows": len(held),
        "train_row_sha256": row_hash(train),
        "validation_row_sha256": row_hash(held),
        "clean_feature_ids_1based": list(original),
        "mi32": base.subset_payload(data.mi32, data),
        "inner_split_row_sha256": [
            {part: row_hash(split[part]) for part in ("fit", "held_out")} for split in data.inner_folds
        ],
    }


def load_folds(config, raw, labels, root):
    partition = base.read_json(config["dataset"]["outer_partition"])
    folds = {}
    checks = []
    for fold in config["experiment"]["outer_folds"]:
        data = base.prepare_fold(raw, labels, config, fold, root)
        checks.append(check_partition(data, partition, labels, fold))
        folds[fold] = data
    return folds, checks


def initialize(config, raw, labels, root, manifest_path):
    if manifest_path.exists():
        raise RuntimeError("manifest already exists; use smoke/stage/evaluate")
    if sorted(config["experiment"]["outer_folds"]) != [2, 3, 4]:
        raise AssertionError("fold protocol changed")
    if config["experiment"]["seeds"] != [2026092501, 2026092502]:
        raise AssertionError("seed protocol changed")
    if config["dataset"]["source_test_open_count"] != 0:
        raise AssertionError("source-test must remain unread")
    if abs(config["scorer"]["old_gamma"] - 1 / 65) > 1e-15:
        raise AssertionError("old gamma changed")
    if abs(config["scorer"]["new_gamma"] - 1 / 32) > 1e-15:
        raise AssertionError("new gamma changed")
    folds, checks = load_folds(config, raw, labels, root)
    base.write_json(
        manifest_path,
        {
            "protocol_version": config["protocol_version"],
            "source_train_sha256": base.sha256(config["dataset"]["source_train"]),
            "outer_partition_sha256": base.sha256(config["dataset"]["outer_partition"]),
            "config_and_code_sha256": protocol_hashes(),
            "frozen_before_outer_endpoint": True,
            "outer_calls_at_freeze": 0,
            "source_test_open_count": 0,
            "folds": checks,
        },
    )
    return folds


def verify_manifest(config, manifest_path, checks):
    manifest = base.read_json(manifest_path)
    if manifest["config_and_code_sha256"] != protocol_hashes():
        raise RuntimeError("frozen config or code hash changed")
    if manifest["source_train_sha256"] != base.sha256(config["dataset"]["source_train"]):
        raise RuntimeError("source-train changed")
    if manifest["outer_partition_sha256"] != base.sha256(config["dataset"]["outer_partition"]):
        raise RuntimeError("outer partition changed")
    if manifest["folds"] != checks:
        raise RuntimeError("fold rows or clean mapping changed")
    return manifest


def grid(config):
    params = config["scorer"]
    gammas = list(dict.fromkeys([factor / 65 for factor in params["gamma_factors"]] + [1 / 65]))
    result = [(float(C), float(gamma)) for C in params["C_grid"] for gamma in gammas]
    if len(result) != 12:
        raise AssertionError("All65 grid must have 12 unique points")
    return result


def stage_fold(config, raw, labels, data, root):
    fold = data.fold
    target = root / "stages" / f"fold-{fold}.json"
    if target.exists():
        print(f"stage already exists: fold {fold}", flush=True)
        return
    started = time.perf_counter()
    masks = {
        "mi32": base.subset_payload(data.mi32, data),
        "all65": base.subset_payload(tuple(range(65)), data),
    }
    gates = {}
    for seed in config["experiment"]["seeds"]:
        rank, gate = train_stg(raw, labels, data, config, seed)
        mask = ranked_mask(rank, 32)
        if len(mask) != 32 or len(set(mask)) != 32:
            raise AssertionError("invalid STG32")
        payload = base.subset_payload(mask, data)
        if (
            data.mapping.original_ids_1based_to_clean_indices(payload["selected_original_feature_ids_1based"])
            != mask
        ):
            raise AssertionError("STG dual-coordinate mismatch")
        gates[str(seed)] = gate
        masks[f"stg32_seed_{seed}"] = payload
    choices = []
    all65 = tuple(range(65))
    for C, gamma in grid(config):
        inner = base.SVCScorer.score_one(
            raw, labels, data.final_ids, all65, data.inner_folds, C=C, gamma=gamma
        )
        choices.append({"C": C, "gamma": gamma, "inner": inner})
    winner = min(
        choices,
        key=lambda row: (
            -row["inner"]["objective"],
            -row["inner"]["accuracy"],
            masks["all65"]["selected_original_feature_ids_1based"],
            row["C"],
            row["gamma"],
        ),
    )
    base.write_json(
        target,
        {
            "fold": fold,
            "masks": masks,
            "gates": gates,
            "grid_choices": choices,
            "all65_grid_selected": winner,
            "grid_physical_svc_fits": 36,
            "grid_fit_seconds": sum(row["inner"]["fit_seconds"] for row in choices),
            "stg_fit_count": 2,
            "stg_fit_seconds": sum(gate["wall_seconds"] for gate in gates.values()),
            "stage_wall_seconds": time.perf_counter() - started,
            "outer_calls_during_training": 0,
            "source_test_open_count": 0,
        },
    )
    print(f"staged fold {fold}: two real STG fits, 36 inner SVC fits, zero outer calls", flush=True)


def evaluate_fold(config, raw, labels, data, root):
    fold = data.fold
    stage_path = root / "stages" / f"fold-{fold}.json"
    target = root / "cases" / f"fold-{fold}.json"
    if target.exists():
        print(f"endpoint already exists: fold {fold}", flush=True)
        return
    stage = base.read_json(stage_path)
    if stage["outer_calls_during_training"] != 0 or stage["source_test_open_count"] != 0:
        raise AssertionError("training-stage data access audit failed")
    if stage["grid_physical_svc_fits"] != 36 or len(stage["grid_choices"]) != 12:
        raise AssertionError("incomplete All65 grid")
    scorer = config["scorer"]
    old = (scorer["old_C"], scorer["old_gamma"])
    new = (scorer["new_C"], scorer["new_gamma"])
    masks = stage["masks"]
    endpoint_cache = {}
    fit_seconds = 0.0

    def endpoint(name, params):
        nonlocal fit_seconds
        mask = tuple(masks[name]["selected_clean_indices_0based"])
        key = (mask, *params)
        if key not in endpoint_cache:
            begun = time.perf_counter()
            endpoint_cache[key] = base.endpoint_metrics(raw, labels, data, mask, C=params[0], gamma=params[1])
            fit_seconds += time.perf_counter() - begun
        return {"C": params[0], "gamma": params[1], "metrics": endpoint_cache[key]}

    comparisons = {}
    for name in masks:
        comparisons[name] = {"old": endpoint(name, old), "new": endpoint(name, new)}
    grid_point = stage["all65_grid_selected"]
    all65_grid = endpoint("all65", (grid_point["C"], grid_point["gamma"]))
    case = {
        "fold": fold,
        "stage_sha256": base.sha256(stage_path),
        "masks": masks,
        "comparisons": comparisons,
        "primary": {
            "stg32_old": {
                str(seed): comparisons[f"stg32_seed_{seed}"]["old"] for seed in config["experiment"]["seeds"]
            },
            "mi32_new": comparisons["mi32"]["new"],
            "all65_old": comparisons["all65"]["old"],
            "all65_grid": all65_grid,
        },
        "outer_physical_svc_fits": len(endpoint_cache),
        "outer_fit_seconds": fit_seconds,
        "outer_calls_during_training": 0,
        "source_test_open_count": 0,
    }
    base.write_json(target, case)
    print(f"evaluated fold {fold}: {len(endpoint_cache)} outer SVC fits", flush=True)


def summarize(config, root):
    folds = config["experiment"]["outer_folds"]
    cases = [base.read_json(root / "cases" / f"fold-{fold}.json") for fold in folds]
    rows = []
    for case in cases:
        p = case["primary"]
        stg = np.mean(
            [
                p["stg32_old"][str(seed)]["metrics"]["balanced_accuracy"]
                for seed in config["experiment"]["seeds"]
            ]
        )
        values = {
            "stg32_old": stg,
            "mi32_new": p["mi32_new"]["metrics"]["balanced_accuracy"],
            "all65_old": p["all65_old"]["metrics"]["balanced_accuracy"],
            "all65_grid": p["all65_grid"]["metrics"]["balanced_accuracy"],
        }
        rows.append({"fold": case["fold"], **values})
    decisions = {}
    for candidate in ("stg32_old", "mi32_new"):
        gains_grid = [100 * (row[candidate] - row["all65_grid"]) for row in rows]
        gains_old = [100 * (row[candidate] - row["all65_old"]) for row in rows]
        decisions[candidate] = {
            "gain_vs_all65_grid_pp_by_fold": gains_grid,
            "mean_gain_vs_all65_grid_pp": float(np.mean(gains_grid)),
            "positive_folds_vs_all65_grid": int(sum(value > 0 for value in gains_grid)),
            "gain_vs_all65_old_pp_by_fold": gains_old,
            "mean_gain_vs_all65_old_pp": float(np.mean(gains_old)),
            "passed_first_gate": (
                float(np.mean(gains_grid)) >= config["decision"]["min_mean_gain_pp"]
                and int(sum(value > 0 for value in gains_grid)) >= config["decision"]["min_positive_folds"]
            ),
        }
    summary = {
        "fold_bacc": rows,
        "mean_bacc": {
            name: float(np.mean([row[name] for row in rows])) for name in rows[0] if name != "fold"
        },
        "decisions": decisions,
        "advance_to_09c_2": any(item["passed_first_gate"] for item in decisions.values()),
        "physical_stg_fits": 6,
        "physical_inner_svc_fits": sum(
            base.read_json(root / "stages" / f"fold-{f}.json")["grid_physical_svc_fits"] for f in folds
        ),
        "physical_outer_svc_fits": sum(case["outer_physical_svc_fits"] for case in cases),
        "source_test_open_count": 0,
    }
    base.write_json(root / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("init", "smoke", "stage", "evaluate", "report"))
    args = parser.parse_args()
    config = base.load_config(CONFIG)
    root, manifest_path = paths(config)
    if args.command == "report":
        summarize(config, root)
        return
    raw, labels = base.load_train(config)
    if args.command == "init":
        initialize(config, raw, labels, root, manifest_path)
        return
    folds, checks = load_folds(config, raw, labels, root)
    verify_manifest(config, manifest_path, checks)
    if args.command == "smoke":
        stage_fold(config, raw, labels, folds[2], root)
        stage = base.read_json(root / "stages" / "fold-2.json")
        if stage["outer_calls_during_training"] or stage["source_test_open_count"]:
            raise AssertionError("smoke data access audit failed")
        print(
            "fold 2 smoke passed: dual coordinates, K32, disjoint rows, "
            "zero outer calls and source-test reads",
            flush=True,
        )
    elif args.command == "stage":
        if not (root / "stages" / "fold-2.json").exists():
            raise RuntimeError("fold 2 smoke must run first")
        for fold in config["experiment"]["outer_folds"]:
            stage_fold(config, raw, labels, folds[fold], root)
    elif args.command == "evaluate":
        if any(not (root / "stages" / f"fold-{fold}.json").exists() for fold in folds):
            raise RuntimeError("all folds must be staged before any outer endpoint")
        for fold in config["experiment"]["outer_folds"]:
            evaluate_fold(config, raw, labels, folds[fold], root)


if __name__ == "__main__":
    main()
