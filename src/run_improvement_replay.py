#!/usr/bin/env python3
"""08B train-only development experiment; reuses 08A collection and SVC scoring."""

from __future__ import annotations

import argparse
import copy
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import run_block_rewrite as base
from radar_ship_fs.ppo.improvement_replay import ImprovementUpdateHook

METHODS = ("plain_single_ppo", "replay_ppo", "replay_only")
OLD_ROOT = Path("experiments/block_rewrite_ppo_v1")


def prepare(raw, labels, config, root, jobs):
    graph_cost_path = root / "graph-costs.json"
    graph_cost = base.read_json(graph_cost_path) if graph_cost_path.exists() else {"classifier_fits": 0}
    # build_feature_graph fits one dependency decision tree per outer fold.
    graph_cost["classifier_fits"] += len(config["experiment"]["outer_folds"])
    base.write_json(graph_cost_path, graph_cost)
    folds, scorers = base.prepare_all_folds(raw, labels, config, root, jobs)
    references = {}
    for fold in folds:
        context_path = OLD_ROOT / "contexts" / f"outer-fold-{fold}.json"
        actual = base.read_json(root / "contexts" / context_path.name)
        expected = base.read_json(context_path)
        assert actual == expected, f"08A context mismatch fold {fold}"
        path = OLD_ROOT / "baselines" / f"fold-{fold}.json"
        references[fold] = dict(
            source=str(path),
            sha256=base.sha256(path),
            context_source=str(context_path),
            context_sha256=base.sha256(context_path),
            endpoints=base.read_json(path)["endpoints"],
        )
        # Only deterministic scores are shared; no archive, trajectory or checkpoint is imported.
        mi = references[fold]["endpoints"]["mi32"]
        assert mi["selected_clean_indices_0based"] == list(folds[fold].mi32)
        scorers[fold].preload({folds[fold].mi32: mi["inner_j"]})
    base.write_json(root / "static_references.json", references)
    return folds, scorers, references


def cost_summary(config):
    roots = [Path(config["output"][key]) for key in ("root", "smoke_root")]
    fits, requests, seconds = 0, 0, 0.0
    completed, partial = 0, []
    graph_fits = 0
    for root in roots:
        graph_path = root / "graph-costs.json"
        if graph_path.exists():
            graph_fits += base.read_json(graph_path)["classifier_fits"]
        for path in root.glob("cases/*/*/checkpoint.pt"):
            result_path = path.parent / "run.json"
            if result_path.exists():
                run = base.read_json(result_path)
                c = run["costs"]
                fits += c["classifier_fit_count"] + c["outer_endpoint_classifier_fits"]
                requests += c["candidate_requests"]
                seconds += c["wall_seconds_this_invocation"]
                completed += 1
            else:
                state = torch.load(path, map_location="cpu")
                fits += state["case_cost"]["classifier_fit_count"]
                requests += state["case_cost"]["candidate_requests"]
                partial.append(str(path))
        verification = root / "verification.json"
        if verification.exists():
            fits += base.read_json(verification).get("additional_physical_classifier_fits", 0)
    return dict(
        classifier_fits=fits + graph_fits,
        svc_classifier_fits=fits,
        graph_dependency_tree_fits=graph_fits,
        candidate_requests=requests,
        case_wall_seconds=seconds,
        completed_including_smoke=completed,
        partial=partial,
    )


def check_budget(config, reserve=0):
    costs = cost_summary(config)
    if costs["classifier_fits"] + reserve > config["experiment"]["physical_classifier_fit_limit"]:
        raise RuntimeError("classifier fit budget exhausted")
    if costs["case_wall_seconds"] > config["experiment"]["wall_time_limit_hours"] * 3600:
        raise RuntimeError("compute time budget exhausted")


def run_case(raw, labels, config, root, folds, scorers, refs, fold, seed, method, variant="base"):
    steps = config["refine"]["steps"] if variant == "steps64" else config["replay"]["steps"]
    hook = ImprovementUpdateHook(
        method,
        refs[fold]["endpoints"]["mi32"]["inner_j"],
        seed,
        fold,
        steps,
        config["replay"]["capacity"],
        config["replay"]["batch_size"],
    )
    return base.run_search_case(
        raw=raw,
        labels=labels,
        fold_data=folds[fold],
        scorer=scorers[fold],
        config=config,
        output_root=root,
        fold=fold,
        seed=seed,
        method=method,
        variant=variant,
        model_method="single_ppo",
        torch_seed_override=seed + 100 * fold + 23,
        update_hook=hook,
        evaluate_outer=variant != "smoke",
    )


def smoke(raw, labels, config, root, folds, scorers, refs):
    short = copy.deepcopy(config)
    short["search"].update(training_episodes=2, frozen_episodes=0, episodes_per_update=2)
    result = run_case(
        raw,
        labels,
        short,
        root,
        folds,
        scorers,
        refs,
        0,
        config["experiment"]["seeds"][0],
        "replay_ppo",
        "smoke",
    )
    state = torch.load(result["checkpoint"], map_location="cpu")
    agent = base._build_agent(folds[0].graph, "single_ppo", config, torch.device("cpu"))
    agent.model.load_state_dict(state["model_state"])
    agent.optimizer.load_state_dict(state["optimizer_state"])
    hook = ImprovementUpdateHook(
        "replay_ppo", refs[0]["endpoints"]["mi32"]["inner_j"], config["experiment"]["seeds"][0], 0
    )
    hook.load_state_dict(state["update_hook_state"])
    assert result["costs"]["classifier_fit_count"] <= 100
    assert result["replay"]["ppo_update_count"] == 1
    base.write_json(
        root / "smoke.json",
        dict(
            status="passed",
            checkpoint_loaded=True,
            real_replay_gradient_steps=result["replay"]["replay_gradient_steps"],
            empty_pool_fallback="tests/test_improvement_replay.py uses synthetic pure-function data",
            costs=result["costs"],
        ),
    )


def verify(raw, labels, root, folds):
    path = root / "verification.json"
    if path.exists():
        return
    candidates = sorted(root.glob("cases/base/*replay_ppo/run.json"))
    if not candidates:
        return
    run = base.read_json(candidates[0])
    data = folds[run["fold"]]
    endpoint = run["endpoints"]["main"]
    subset = data.mapping.validate_artifact_selection(
        dict(
            selected_clean_indices=endpoint["selected_clean_indices_0based"],
            selected_original_feature_ids=endpoint["selected_original_feature_ids_1based"],
        )
    )
    inner = base.SVCScorer.score_one(raw, labels, data.final_ids, subset, data.inner_folds)
    outer = base.endpoint_metrics(raw, labels, data, subset)
    error = abs(inner["objective"] - endpoint["inner_j"])
    outer_error = max(abs(outer[k] - endpoint["outer"][k]) for k in outer)
    payload = dict(
        status="passed" if max(error, outer_error) <= 1e-12 else "failed",
        case=str(candidates[0]),
        mapping_validated=True,
        inner_absolute_error=error,
        outer_max_absolute_error=outer_error,
        independent_inner=inner,
        independent_outer=outer,
        additional_physical_classifier_fits=4,
    )
    base.write_json(path, payload)
    assert payload["status"] == "passed"


def results(root):
    rows = []
    runs = [base.read_json(p) for p in sorted(root.glob("cases/*/*/run.json"))]
    for run in runs:
        for endpoint, metrics in run["endpoints"].items():
            rows.append(
                dict(
                    variant=run["variant"],
                    fold=run["fold"],
                    seed=run["seed"],
                    method=run["method"],
                    endpoint=endpoint,
                    inner_j=metrics["inner_j"],
                    **metrics["outer"],
                )
            )
    return runs, pd.DataFrame(rows)


def report(config, root):
    runs, frame = results(root)
    if frame.empty:
        return
    refs = base.read_json(root / "static_references.json")
    frame.to_csv(root / "results.csv", index=False)
    main = frame[frame.endpoint == "main"]
    frozen = frame[frame.endpoint == "frozen"]
    metric_columns = ["inner_j", "balanced_accuracy", "accuracy", "recall_negative", "recall_positive"]
    summary = main.groupby(["variant", "method", "fold"])[metric_columns].mean().reset_index()
    summary.to_csv(root / "fold_summary.csv", index=False)
    means = summary.groupby(["variant", "method"])[metric_columns].mean()
    means.to_csv(root / "method_summary.csv")
    pairs = []
    for _, row in main[main.method == "replay_ppo"].iterrows():
        for reference in ("plain_single_ppo", "replay_only", "all_svc", "g1", "mi32", "all_lr"):
            if reference in METHODS:
                other = main[
                    (main.variant == "base")
                    & (main.fold == row.fold)
                    & (main.seed == row.seed)
                    & (main.method == reference)
                ]
                if other.empty:
                    continue
                score = float(other.iloc[0].balanced_accuracy)
            else:
                score = refs[str(row.fold)]["endpoints"][reference]["outer"]["balanced_accuracy"]
            pairs.append(
                dict(
                    variant=row.variant,
                    fold=row.fold,
                    seed=row.seed,
                    reference=reference,
                    difference_pp=100 * (row.balanced_accuracy - score),
                )
            )
    base.write_csv(root / "paired_differences.csv", pairs)
    paired = pd.DataFrame(pairs)
    if not paired.empty:
        paired.groupby(["variant", "reference", "fold"]).difference_pp.mean().to_csv(
            root / "paired_fold_means.csv"
        )
    learning = []
    for run in runs:
        r = run["replay"]
        learning.append(
            dict(
                variant=run["variant"],
                fold=run["fold"],
                seed=run["seed"],
                method=run["method"],
                pool_size=r["pool_size"],
                pool_declines=r["pool_immediate_declines"],
                replay_steps=r["replay_gradient_steps"],
                ppo_updates=r["ppo_update_count"],
                probe_probability_before=r["probe_before"]["probability"],
                probe_probability_after=r["probe_after"]["probability"],
                probe_entropy_before=r["probe_before"]["entropy"],
                probe_entropy_after=r["probe_after"]["entropy"],
                pool_nll=r["final_pool"].get("nll"),
            )
        )
    base.write_csv(root / "learning_summary.csv", learning)
    costs = cost_summary(config)
    base.write_json(root / "costs.json", costs)
    expected = [
        (v, f, s, m)
        for v, methods in [("base", METHODS)]
        for s in config["experiment"]["seeds"]
        for f in config["experiment"]["outer_folds"]
        for m in methods
    ]
    if config["refine"]["enabled"]:
        expected.extend(
            ("steps64", f, s, "replay_ppo")
            for s in config["experiment"]["seeds"]
            for f in config["experiment"]["outer_folds"]
        )
    done = {(r["variant"], r["fold"], r["seed"], r["method"]) for r in runs}
    missing = [list(key) for key in expected if key not in done]
    base.write_json(root / "completion.json", dict(completed=len(runs), missing=missing))
    lines = [
        "# 08B：改进轨迹回放辅助 PPO 实验结果",
        "",
        "本轮为已看过的 source-train 开发 fold0/1；不是新 test。历史 source-test 未打开。",
        "",
        f"已完成 {len(runs)} cases；未完成 {missing}。主 endpoint 按所有已评分子集最高 inner J 选择，"
        "冻结 endpoint 仅限冻结32条轨迹；outer 不参与选择。",
        "",
        "| 版本/方法 | 主 BAcc | 冻结 BAcc | Accuracy | Recall− | Recall+ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for (variant, method), row in means.iterrows():
        fr = frozen[(frozen.variant == variant) & (frozen.method == method)]
        fb = fr.groupby("fold").balanced_accuracy.mean().mean()
        lines.append(
            f"| {variant}/{method} | {100 * row.balanced_accuracy:.4f}% | {100 * fb:.4f}% | "
            f"{100 * row.accuracy:.4f}% | {100 * row.recall_negative:.4f}% | "
            f"{100 * row.recall_positive:.4f}% |"
        )
    lines += [
        "",
        "先在每fold内平均3 seeds，再平均两个fold。静态参照来自08A相同context/model，"
        "文件路径及hash见 static_references.json；未重新拟合基线。",
        "",
    ]
    for reference in ("all_svc", "g1", "mi32", "all_lr"):
        val = np.mean([r["endpoints"][reference]["outer"]["balanced_accuracy"] for r in refs.values()])
        lines.append(f"- {reference}: {100 * val:.4f}%")
    lines += ["", "| 版本 | Replay-PPO 相对 | 平均差 pp |", "|---|---|---:|"]
    if not paired.empty:
        for (v, ref), vals in paired.groupby(["variant", "reference"]):
            value = vals.groupby("fold").difference_pp.mean().mean()
            lines.append(f"| {v} | {ref} | {value:+.4f} |")
    lines += [
        "",
        "| 版本 | fold | seed | Plain | Replay-PPO | Replay-only | Replay−Plain pp |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for (v, f, s), group in main.groupby(["variant", "fold", "seed"]):
        scores = dict(zip(group.method, group.balanced_accuracy))

        def fmt(m):
            return f"{100 * scores[m]:.4f}%" if m in scores else "—"

        delta = (
            100 * (scores["replay_ppo"] - scores["plain_single_ppo"])
            if all(m in scores for m in ("replay_ppo", "plain_single_ppo"))
            else float("nan")
        )
        lines.append(
            f"| {v} | {f} | {s} | {fmt('plain_single_ppo')} | {fmt('replay_ppo')} | "
            f"{fmt('replay_only')} | {delta:+.4f} |"
        )
    lines += [
        "",
        "学习行为、结论及下一处修改见本报告末尾的实测解读。",
        "",
        f"本轮成本：{costs['classifier_fits']} classifier fits（含smoke、端点评价、独立复核及图依赖树拟合）；"
        f"{costs['candidate_requests']} J请求；case累计墙钟 {costs['case_wall_seconds']:.1f}s。",
        "",
        "代码复用08A评分、收集与重启；三臂都是Single/K32/H8，torch_seed=seed+100*fold+23。"
        "回放仅从本case训练轨迹纳入，PPO之后单独优化；冻结不纳入。"
        "Plain也统计候选回放机会但执行0个回放步。原始记录、曲线、子集、checkpoint、RNG、"
        "probe、成本及配置位于 experiments/improvement_replay_ppo_v1。",
    ]
    interpretation = root / "interpretation.md"
    if interpretation.exists():
        lines += ["", interpretation.read_text()]
    Path(config["output"]["report"]).write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v16n/improvement_replay_ppo_v1.toml"))
    parser.add_argument("--stage", choices=["smoke", "run", "refine", "report"], required=True)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    config = base.load_config(args.config)
    torch.set_num_threads(config["ppo"]["torch_threads"])
    root = Path(config["output"]["smoke_root" if args.stage == "smoke" else "root"])
    root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    raw, labels = base.load_train(config)
    base.write_json(root / f"config-{args.stage}.json", config)
    base.write_json(
        root / f"invocation-{args.stage}.json",
        dict(
            started=started,
            jobs=args.jobs,
            source_test_open_count=0,
            torch=str(torch.__version__),
            cuda=torch.cuda.is_available(),
            thread_limits={
                k: os.environ.get(k)
                for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
            },
            config_sha256=base.sha256(args.config),
            source_hashes={
                str(p): base.sha256(p)
                for p in [
                    Path(__file__),
                    Path(base.__file__),
                    Path("src/radar_ship_fs/ppo/improvement_replay.py"),
                ]
            },
        ),
    )
    folds, scorers, refs = prepare(raw, labels, config, root, args.jobs)
    if args.stage == "smoke":
        smoke(raw, labels, config, root, folds, scorers, refs)
    elif args.stage in ("run", "refine"):
        if args.stage == "refine" and not config["refine"]["enabled"]:
            raise ValueError("refinement must be justified and enabled in config first")
        variant = "steps64" if args.stage == "refine" else "base"
        methods = ("replay_ppo",) if args.stage == "refine" else METHODS
        for seed in config["experiment"]["seeds"]:
            for fold in config["experiment"]["outer_folds"]:
                for method in methods:
                    check_budget(config, reserve=4322)
                    run = run_case(
                        raw, labels, config, root, folds, scorers, refs, fold, seed, method, variant
                    )
                    print(
                        f"ENDPOINT {variant} f{fold} s{seed} {method}: "
                        f"{run['endpoints']['main']['outer']['balanced_accuracy']:.8f}",
                        flush=True,
                    )
                    report(config, root)
    else:
        check_budget(config, reserve=4)
        verify(raw, labels, root, folds)
        report(config, root)
    base.write_json(root / f"timing-{args.stage}.json", dict(wall_seconds=time.time() - started))


if __name__ == "__main__":
    main()
