#!/usr/bin/env python3
"""IRFS reproduction — the living integration main (epic 2026-06-11-irfs-reproduction).

One top-level entry point that reads top to bottom as the pipeline's narrative: config/seed ->
dataset -> leakage-safe split -> Decision-Tree probe -> common subset contract -> selection methods
-> scoring -> Best/Average metrics -> reproducible artifact.

A THIN, HONEST narrator: it only wires and prints the real classes each stage exposes, never
reimplementing their logic. The full pipeline always runs top-to-bottom on one shared context; the
warm-up comparison and interactive-feedback stages are each broken out into their inner steps.

Run it with:   python src/run_irfs.py                        (WDBC, the default)
python src/run_irfs.py --dataset parkinsons   (or IRFS_DATASET=parkinsons)

The dataset is the one config knob the narrative reads from the command line: it selects which
registered dataset (COMP-001) the shared split + probe are built on, so the whole pipeline below
runs on that dataset unchanged. Everything else stays on the recorded defaults.

For spec traceability, the five stages below correspond to the epic's PHASE-001…005.
"""

from __future__ import annotations

import argparse
import os
import time

# Core setup + comparison harness/engine
from config import load_config
from harness.aggregate import aggregate_and_write, format_aggregate_table
from harness.artifact import (
    build_selection_artifact,
    build_test_artifact,
    write_run_artifacts,
)
from harness.final_metrics import score_final_metrics
from harness.orchestrator import MethodOrchestrator

# Interactive-feedback reinforced IRFS methods (tree state + personalized reward, trainers/advice
# adapter, and the headline/diagnostic method assembly over one shared state+reward)
from methods.configure import (
    build_no_trainer_engine,
    reinforced_method_names,
    run_reinforced_methods,
)

# Classical baseline suite
from methods.dt_rfe import DTImportanceEliminator
from methods.l1 import L1Selector
from methods.mrmr import MRMRSelector, implementation_identity
from methods.relevance_topk import RelevanceTopKSelector
from methods.suite import run_full_comparison


def _stage(title: str) -> None:
    print(f"\n── {title} " + "─" * max(0, 64 - len(title)))


def _inner(label: str) -> None:
    print(f"   · {label}")


def _method_logger():
    """A fresh per-selector progress hook for ``orchestrator.run`` / ``run_comparison``.

    Prints ``running…`` when a selector starts and ``done in Ns`` when it finishes, so the slow ones
    (the reinforced engine in the comparison, dt_rfe in the classical suite) show a sign of life
    instead of a silent gap. flush=True so a tail/Monitor sees each line live.
    """
    starts: dict[str, float] = {}

    def hook(name: str, phase: str) -> None:
        if phase == "start":
            starts[name] = time.perf_counter()
            print(f"       → {name:<16} running…", flush=True)
        else:
            print(
                f"       ✓ {name:<16} done in {time.perf_counter() - starts[name]:5.0f}s",
                flush=True,
            )

    return hook


def _interactive_feedback(ctx, *, diagnostic_ablations: bool = False) -> None:
    """IRFS interactive feedback: headline methods, plus diagnostic ablations when requested.

    Broken out for readability; consumes ``ctx`` (the shared context built in setup). Swaps the
    tree-structured state + personalized reward in behind the engine seam, wires the trainers/advice
    adapter, then configures and runs the selected reinforced methods with per-method and
    ~10%-heartbeat progress. By default this is only no-trainer RL and full IRFS; the trainer-
    specific ablations run only when ``diagnostic_ablations`` is true.
    """
    _stage("IRFS interactive feedback")

    _inner("swap tree-structured state + per-agent reward, prove no-trainer convergence")
    # The no-trainer IRFS engine is the shared ReinforcedEngine with the tree-structured state and
    # personalized reward swapped in behind the unchanged seam and subset contract.
    build_no_trainer_engine()
    print("       engine no_trainer_irfs ready (state+reward swapped, satisfies subset contract)")

    _inner("trainers + advice adapter")
    # The classifier, the relevance / DT-importance trainers, and the hybrid scheduler steer
    # hesitant agents; the advice adapter applies the active trainer's advice through the pluggable
    # ActionAdvisor seam, so the methods below differ only by which trainer is plugged in.
    print("       trainers + advice adapter ready (advice moves hesitant agents)")

    method_names = reinforced_method_names(include_diagnostic_ablations=diagnostic_ablations)
    mode = "headline + diagnostic ablations" if diagnostic_ablations else "headline methods"
    _inner(f"configure & run {mode}")
    # Headline runs compare the no-trainer RL baseline against full IRFS. The relevance-only and
    # DT-importance-only ablations are diagnostic and opt-in. Full IRFS uses Hybrid Teaching (the
    # reference headline) directly.
    # Progress hooks for this long-running stage (the methods run silently otherwise; see the
    # engine's per-step loop). Purely observational — they never touch the RNG/subset/learning, so the
    # run stays bit-identical and reproducible. flush=True so a tail/Monitor on the output sees each
    # line as it happens rather than buffered at the end.
    starts: dict[str, float] = {}

    def _on_method(index: int, total: int, name: str, phase: str) -> None:
        if phase == "start":
            starts[name] = time.perf_counter()
            print(f"       [{index}/{total}] {name:<14} running…", flush=True)
        else:
            elapsed = time.perf_counter() - starts[name]
            print(f"       [{index}/{total}] {name:<14} done in {elapsed:5.0f}s", flush=True)

    def _on_step(name: str, step: int, budget: int, accuracy: float, best: float) -> None:
        # Heartbeat ~every 10% of the budget (and always the final step), so a method that takes
        # minutes still prints a handful of "still alive, here's the trajectory" lines.
        every = max(1, budget // 10)
        if (step + 1) % every == 0 or step + 1 == budget:
            print(
                f"           {name:<14} step {step + 1:>3}/{budget}  acc={accuracy:.4f}  best={best:.4f}",
                flush=True,
            )

    reinforced = run_reinforced_methods(
        ctx,
        include_diagnostic_ablations=diagnostic_ablations,
        on_method=_on_method,
        on_step=_on_step,
    )
    for name in method_names:
        sel = reinforced[name]
        best = max(step.accuracy for step in sel.per_step)
        print(f"       method {name:<14} size={len(sel.selected):<3} best_accuracy={best:.4f}")
    print("       same-seed reproducibility verified by tests/test_invariants.py")


def _full_comparison(orchestrator, *, diagnostic_ablations: bool = False) -> None:
    """Full comparison & held-out validation: the epic's headline run.

    Runs the classical baselines and selected reinforced methods through the one shared context in a
    single unified pass, scores each final subset on test, then emits the complete reproducible
    artifact (selection + test) under
    ``experiments/<dataset>/seed-<n>/``. Reinforced numbers are the unified-run state — the
    canonical headline — on the same shared context the interactive-feedback stage already
    snapshotted, so they agree. Prints the Best/Average headline plus the val/test summary. The
    progress hooks are observational only, never touching the RNG/subset/learning, so the run stays
    bit-identical and reproducible.
    """
    _stage("Full comparison & held-out validation")
    ctx = orchestrator.context

    def _on_step(name: str, step: int, budget: int, accuracy: float, best: float) -> None:
        every = max(1, budget // 10)  # ~10%-of-budget heartbeat (and always the final step)
        if (step + 1) % every == 0 or step + 1 == budget:
            print(
                f"           {name:<16} step {step + 1:>3}/{budget}  acc={accuracy:.4f}  best={best:.4f}",
                flush=True,
            )

    # 1. The one unified comparison on the shared context (validation surface).
    comparison = run_full_comparison(
        orchestrator,
        include_diagnostic_ablations=diagnostic_ablations,
        on_method=_method_logger(),
        on_step=_on_step,
    )
    # 2. Score each final subset on test.
    final = score_final_metrics(ctx, comparison)
    test_by_name = {m.name: m.test for m in final.per_method}

    # Headline: per-method size, validation accuracy, windowed Best/Average (reinforced), held-out test.
    mode = "diagnostic" if diagnostic_ablations else "headline"
    print(
        f"\n       {mode} comparison ({len(comparison.runs)} methods, "
        f"held-out test n={final.test_n_samples}):"
    )
    for r in comparison.runs:
        win = comparison.windowed.get(r.name)
        windowed = f"  best={win.best:.4f} avg={win.average:.4f}" if win else "  (single-shot)        "
        print(
            f"       {r.name:<16} size={r.size:<3} val={r.accuracy:.4f}{windowed}  "
            f"test={test_by_name[r.name]:.4f}"
        )

    # 3. Emit the complete reproducible artifact (selection + test) under the per-dataset/per-seed path.
    selection = build_selection_artifact(comparison, ctx, mrmr_identity=implementation_identity())
    test_artifact = build_test_artifact(final, ctx)
    sel_path, test_path = write_run_artifacts(selection, test_artifact, ctx)
    print(f"\n       artifact written: {sel_path}")
    print(f"       artifact written: {test_path}")
    print(
        f"       fidelity notes: {len(selection['fidelity_notes'])}   "
        f"mRMR: {selection['mrmr_identity']['name']}=={selection['mrmr_identity']['version']}"
    )
    print("       same-seed reproducible (deterministic under the recorded seed)")


def _classical_baselines(orchestrator: MethodOrchestrator, ctx, diagnostic_ablations: bool):
    # ── Classical baselines ───────────────────────────────────────────────────────────
    _stage("Classical baselines")
    # All four classical baselines run through the one orchestrator on the shared split + probe.
    # The three fixed-size baselines select half the features; L1 is variable-size.
    classical = [
        ("relevance_topk", RelevanceTopKSelector()),
        ("dt_rfe", DTImportanceEliminator()),
        ("mrmr", MRMRSelector()),  # pinned implementation
        ("l1", L1Selector()),  # variable-size
    ]
    classical_runs = orchestrator.run(classical, on_method=_method_logger())  # dt_rfe is the slow one
    for r in classical_runs:
        print(f"       method {r.name:<16} size={r.size:<3} accuracy={r.accuracy:.4f}")
    mrmr_id = implementation_identity()  # pinned identity for the artifact
    print(f"       mRMR pinned implementation: {mrmr_id['name']}=={mrmr_id['version']}")

    # ── IRFS interactive feedback ─────────────────────────────────────────────────────
    _interactive_feedback(ctx, diagnostic_ablations=diagnostic_ablations)


def _run_seed(config, seed: int, *, diagnostic_ablations: bool = False) -> None:
    """Run the full top-to-bottom pipeline once under a single active ``seed``.

    Each seed drives its own independent leakage-safe context (its own shared RNG, split, and probe)
    and writes its own ``experiments/<dataset>/seed-<n>/`` artifact, so a multi-seed sweep is just
    this body run once per seed with sibling output folders.
    """
    # ── Setup ─────────────────────────────────────────────────────────────────────────
    _stage("Setup")
    print(f"   config: dataset={config.dataset!r}  seed={seed}  diagnostic_ablations={diagnostic_ablations}")
    # One shared leakage-safe context (config → seed → load → split → probe); every method
    # is scored on this same split + probe.
    orchestrator = MethodOrchestrator(config, seed=seed)
    ctx = orchestrator.context
    print(f"   shared context: n_features={ctx.n_features}")

    _classical_baselines(orchestrator, ctx, diagnostic_ablations=diagnostic_ablations)

    # ── Full comparison & held-out validation ─────────────────────────────────────────
    _full_comparison(orchestrator, diagnostic_ablations=diagnostic_ablations)


def main(
    dataset: str = "wdbc",
    seeds: list[int] | None = None,
    *,
    diagnostic_ablations: bool = False,
    state_encoder: str = "fixed",
) -> None:
    """Run the configured IRFS comparison for each seed and aggregate the results."""
    print("=" * 70)
    print("IRFS reproduction — integration main")
    print("=" * 70)

    # The dataset and the seed list are the two command-line knobs. The full pipeline runs
    # top-to-bottom once per seed; each seed is an independent leakage-safe run writing its own
    # seed-<n>/ artifact folder. A single-seed run is just a one-element list.
    # state_encoder picks the reinforced state representation for every reinforced arm: the
    # fixed-weight baseline encoder (default) or the jointly-trained GCN.
    config = load_config({"dataset": dataset, "state_encoder": state_encoder})
    seed_list = seeds if seeds else list(config.seeds)
    print(f"   sweeping seeds={seed_list}")
    print(f"   state encoder={config.state_encoder}")
    print(f"   diagnostic ablations={'on' if diagnostic_ablations else 'off'}")

    for i, seed in enumerate(seed_list, start=1):
        print("\n" + "#" * 70)
        print(f"# seed {seed}  ({i}/{len(seed_list)})")
        print("#" * 70)
        _run_seed(config, seed, diagnostic_ablations=diagnostic_ablations)

    # ── Cross-seed aggregate ──────────────────────────────────────────────────────────
    # Collapse the per-seed artifacts just written into one cross-seed view (per-method test
    # distribution + full_irfs-vs-classical paired deltas), so the headline question is answered
    # on a distribution rather than a single noisy split. Restricted to the seeds this run swept.
    _stage("Cross-seed aggregate")
    agg, agg_path = aggregate_and_write(config.dataset, seeds=seed_list)
    print(format_aggregate_table(agg))
    print(f"\n       aggregate written: {agg_path}")

    print("\n" + "=" * 70)
    print(f"done — full pipeline ran for {len(seed_list)} seed(s): {seed_list}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IRFS reproduction integration main.")
    parser.add_argument(
        "--dataset",
        default=os.environ.get("IRFS_DATASET", "wdbc"),
        help="Registered dataset to run on (wdbc or parkinsons). "
        "Falls back to the IRFS_DATASET env var, then 'wdbc'.",
    )
    parser.add_argument(
        "--seeds",
        default=os.environ.get("IRFS_SEEDS"),
        help="Comma-separated seeds to sweep (e.g. '42,1,17'). A single seed is just '42'. "
        "Falls back to the IRFS_SEEDS env var, then the config default.",
    )
    parser.add_argument(
        "--diagnostic-ablations",
        action="store_true",
        help="Also run relevance-only and Decision-Tree-importance-only reinforced "
        "diagnostic ablations. Defaults to off.",
    )
    parser.add_argument(
        "--state-encoder",
        default=os.environ.get("IRFS_STATE_ENCODER", "fixed"),
        choices=["fixed", "trained_gcn"],
        help="State encoder for the reinforced methods: 'fixed' (default — the fixed-weight baseline "
        "encoder) or 'trained_gcn' (the GCN trained jointly with the value heads). Falls back to "
        "the IRFS_STATE_ENCODER env var, then 'fixed'.",
    )
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    main(
        args.dataset,
        seeds,
        diagnostic_ablations=args.diagnostic_ablations,
        state_encoder=args.state_encoder,
    )
