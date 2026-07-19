"""Cross-seed aggregator for a multi-seed sweep.

A single seed's run is noisy: ``full_irfs`` can trail the strongest classical baseline
(``dt_rfe``) by a single test sample on one split and lead it on the next. The multi-seed
sweep (``src/run_irfs.py --seeds ...``) writes one independent run per seed under
``experiments/<dataset>/seed-<n>/``; this module collapses those sibling folders into a
single cross-seed view so the headline question — *does the reinforced method match or beat
the best classical baseline once the lucky/unlucky split is averaged out?* — can be answered
on a distribution rather than one draw.

It reads only the already-written run artifacts (``test.json`` for the held-out metrics,
``selection.json`` for the windowed Best/Average of the reinforced methods); it never
re-runs the pipeline, so aggregation is cheap and can be repeated over existing folders.

Three sections come out of :func:`aggregate`:

* **per-method summary** — mean/std/min/max test (and validation) accuracy, mean subset
  size, and mean windowed Best/Average (reinforced methods only), one row per method;
* **paired comparison** — ``full_irfs`` paired *by seed* against each baseline (the classical
  four plus the MARLFS reinforced baseline; both methods ran on the identical split, so the
  per-seed delta is a true paired difference), reported in accuracy and in test samples, with
  win/tie/loss counts — so the paper's "IRFS improves over MARLFS" claim is answered directly;
* **integrity** — which seeds were found vs requested, whether ``test_n_samples`` and the
  method set are constant across seeds, and any warnings — so a partial sweep cannot
  silently masquerade as a complete one.

Run standalone over existing folders with::

    python -m harness.aggregate --dataset wdbc
    python -m harness.aggregate --dataset wdbc --seeds 42,1,17
"""

from __future__ import annotations

import glob
import json
import os
import statistics
from typing import Any, Optional

from harness.artifact import write_artifact
from methods.configure import REINFORCED_METHOD_NAMES

# The reinforced method whose competitiveness the study turns on (the paired comparison's left side).
HEADLINE_METHOD = "full_irfs"

# Reinforced methods the headline is ALSO paired against, beyond the classical baselines. MARLFS is a
# reinforced method (so it is excluded from ``classical``), but the paper's headline claim is precisely
# "IRFS improves over MARLFS" (§4.4), so the study wants the paired full_irfs−marlfs delta reported
# alongside the classical baselines. A name here that is absent from a run (or is the headline itself) is
# skipped.
REINFORCED_BASELINES = ("marlfs",)


# --- Reading the per-seed artifacts -----------------------------------------------------------


class SeedRun:
    """One seed's loaded run: its held-out per-method metrics plus the windowed Best/Average.

    ``methods`` maps method name -> the ``test.json`` row (``size``/``validation``/``test``).
    ``windowed`` maps the reinforced method name -> ``{"best", "average"}`` from ``selection.json``
    (empty when that file is absent). ``test_n_samples`` is the size of the held-out partition the
    test accuracies were scored on (used to express deltas in samples).
    """

    def __init__(self, seed: int, test_n_samples: Optional[int], methods: dict, windowed: dict) -> None:
        self.seed = seed
        self.test_n_samples = test_n_samples
        self.methods = methods
        self.windowed = windowed


def _seed_dir(base_dir: str, dataset: str, seed: int) -> str:
    return os.path.join(base_dir, dataset, f"seed-{seed}")


def _discover_seeds(base_dir: str, dataset: str) -> list[int]:
    """Every seed with a ``seed-<n>/`` folder under ``<base_dir>/<dataset>/``, ascending."""
    pattern = os.path.join(base_dir, dataset, "seed-*")
    seeds: list[int] = []
    for path in glob.glob(pattern):
        tail = os.path.basename(path).removeprefix("seed-")
        if os.path.isdir(path) and tail.lstrip("-").isdigit():
            seeds.append(int(tail))
    return sorted(seeds)


def load_seed_runs(
    dataset: str,
    *,
    base_dir: str = "experiments",
    seeds: Optional[list[int]] = None,
) -> tuple[list[SeedRun], list[int]]:
    """Load each requested seed's run; return ``(loaded_runs, missing_seeds)``.

    ``seeds`` defaults to every seed folder found on disk for ``dataset``. A seed whose
    ``test.json`` is absent is reported in ``missing_seeds`` rather than loaded. ``selection.json``
    is optional: when present its windowed Best/Average is attached, otherwise ``windowed`` is empty.
    """
    requested = seeds if seeds is not None else _discover_seeds(base_dir, dataset)
    runs: list[SeedRun] = []
    missing: list[int] = []
    for seed in requested:
        test_path = os.path.join(_seed_dir(base_dir, dataset, seed), "test.json")
        if not os.path.isfile(test_path):
            missing.append(seed)
            continue
        with open(test_path, encoding="utf-8") as fh:
            test = json.load(fh)
        methods = {m["name"]: m for m in test.get("methods", [])}

        windowed: dict = {}
        sel_path = os.path.join(_seed_dir(base_dir, dataset, seed), "selection.json")
        if os.path.isfile(sel_path):
            with open(sel_path, encoding="utf-8") as fh:
                windowed = json.load(fh).get("comparison", {}).get("windowed", {}) or {}

        runs.append(SeedRun(seed, test.get("test_n_samples"), methods, windowed))
    return runs, missing


# --- Statistics helpers -----------------------------------------------------------------------


def _summary(values: list[float]) -> dict[str, Any]:
    """Mean/std/min/max plus the raw values.

    Sample std needs >=2 points; 0.0 for one seed.
    """
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) >= 2 else 0.0,
        "min": min(values),
        "max": max(values),
        "values": values,
    }


# --- Aggregation ------------------------------------------------------------------------------


def _method_names_in_order(runs: list[SeedRun]) -> list[str]:
    """Method names in their first-seen artifact order across the loaded runs (stable)."""
    ordered: list[str] = []
    for run in runs:
        for name in run.methods:
            if name not in ordered:
                ordered.append(name)
    return ordered


def _per_method_summary(runs: list[SeedRun], names: list[str]) -> list[dict[str, Any]]:
    """One summary row per method: test/validation/size distributions + windowed (reinforced)."""
    rows: list[dict[str, Any]] = []
    for name in names:
        present = [run for run in runs if name in run.methods]
        test = [float(run.methods[name]["test"]) for run in present]
        val = [float(run.methods[name]["validation"]) for run in present]
        size = [float(run.methods[name]["size"]) for run in present]
        row: dict[str, Any] = {
            "name": name,
            "n_seeds": len(present),
            "test": _summary(test),
            "validation": _summary(val),
            "size": _summary(size),
            "windowed": None,
        }
        if name in REINFORCED_METHOD_NAMES:
            best = [float(run.windowed[name]["best"]) for run in present if name in run.windowed]
            avg = [float(run.windowed[name]["average"]) for run in present if name in run.windowed]
            if best and avg:
                row["windowed"] = {
                    "best_mean": statistics.fmean(best),
                    "average_mean": statistics.fmean(avg),
                }
        rows.append(row)
    return rows


def _paired_vs_headline(runs: list[SeedRun], names: list[str]) -> list[dict[str, Any]]:
    """Pair the headline method against each baseline (classical + MARLFS), seed by seed.

    For each seed where both methods ran (same split), the delta is ``headline.test −
    baseline.test``, reported as accuracy and as test samples (delta × that seed's
    ``test_n_samples``). Win/tie/loss counts are from the headline's perspective. The baselines are
    the classical methods plus any :data:`REINFORCED_BASELINES` present (MARLFS), so the paper's
    "IRFS improves over MARLFS" delta is reported here too. Empty when the headline method is
    absent.
    """
    if HEADLINE_METHOD not in names:
        return []
    classical = [n for n in names if n not in REINFORCED_METHOD_NAMES]
    baselines = classical + [n for n in REINFORCED_BASELINES if n in names and n != HEADLINE_METHOD]
    pairs: list[dict[str, Any]] = []
    for baseline in baselines:
        delta_acc: list[float] = []
        delta_samples: list[float] = []
        win = tie = loss = 0
        for run in runs:
            if HEADLINE_METHOD not in run.methods or baseline not in run.methods:
                continue
            d = float(run.methods[HEADLINE_METHOD]["test"]) - float(run.methods[baseline]["test"])
            delta_acc.append(d)
            if run.test_n_samples:
                delta_samples.append(d * int(run.test_n_samples))
            if d > 0:
                win += 1
            elif d < 0:
                loss += 1
            else:
                tie += 1
        if not delta_acc:
            continue
        pairs.append(
            {
                "baseline": baseline,
                "n_seeds": len(delta_acc),
                "per_seed_delta_acc": delta_acc,
                "per_seed_delta_samples": delta_samples,
                "mean_delta_acc": statistics.fmean(delta_acc),
                "mean_delta_samples": statistics.fmean(delta_samples) if delta_samples else None,
                "win": win,
                "tie": tie,
                "loss": loss,
            }
        )
    return pairs


def aggregate(
    dataset: str,
    *,
    base_dir: str = "experiments",
    seeds: Optional[list[int]] = None,
) -> dict[str, Any]:
    """Collapse a dataset's per-seed runs into the cross-seed aggregate structure.

    ``seeds`` (defaults to every seed folder found) is also the *expected* set: a requested seed
    with no ``test.json`` lands in ``integrity.missing_seeds`` so a partial sweep is visible. Raises
    ``FileNotFoundError`` when not a single requested seed could be loaded.
    """
    runs, missing = load_seed_runs(dataset, base_dir=base_dir, seeds=seeds)
    if not runs:
        raise FileNotFoundError(
            f"No loadable seed runs for dataset {dataset!r} under {base_dir!r} "
            f"(requested {seeds if seeds is not None else 'all discovered'}; missing {missing})."
        )

    names = _method_names_in_order(runs)
    n_samples_values = sorted({run.test_n_samples for run in runs if run.test_n_samples is not None})
    methods_consistent = all(set(run.methods) == set(runs[0].methods) for run in runs)

    warnings: list[str] = []
    if missing:
        warnings.append(f"missing seed runs: {missing}")
    if len(n_samples_values) > 1:
        warnings.append(f"test_n_samples varies across seeds: {n_samples_values}")
    if not methods_consistent:
        warnings.append("the method set differs across seeds")

    return {
        "dataset": dataset,
        "seeds": [run.seed for run in runs],
        "n_seeds": len(runs),
        "test_n_samples": n_samples_values[0] if len(n_samples_values) == 1 else n_samples_values,
        "methods": _per_method_summary(runs, names),
        "paired_vs_full_irfs": _paired_vs_headline(runs, names),
        "integrity": {
            "missing_seeds": missing,
            "test_n_samples_constant": len(n_samples_values) <= 1,
            "methods_consistent": methods_consistent,
            "warnings": warnings,
        },
    }


# --- Presentation -----------------------------------------------------------------------------


def format_aggregate_table(agg: dict[str, Any]) -> str:
    """A human-readable two-table view: per-method summary and the paired comparison."""
    lines: list[str] = []
    lines.append(
        f"cross-seed aggregate — dataset={agg['dataset']}  "
        f"seeds={agg['seeds']}  test_n_samples={agg['test_n_samples']}"
    )

    lines.append("")
    lines.append(
        f"{'method':<16} {'n':>2}  {'test mean±std':>15}  {'[min, max]':>16}  "
        f"{'val':>6}  {'size':>5}  best/avg"
    )
    for m in agg["methods"]:
        t = m["test"]
        w = m["windowed"]
        wtxt = f"{w['best_mean']:.4f}/{w['average_mean']:.4f}" if w else "—"
        lines.append(
            f"{m['name']:<16} {m['n_seeds']:>2}  {t['mean']:>7.4f}±{t['std']:<6.4f}  "
            f"[{t['min']:.4f}, {t['max']:.4f}]  {m['validation']['mean']:>6.4f}  "
            f"{m['size']['mean']:>5.1f}  {wtxt}"
        )

    pairs = agg["paired_vs_full_irfs"]
    if pairs:
        lines.append("")
        lines.append(f"{HEADLINE_METHOD} vs baselines — classical + MARLFS (paired by seed):")
        lines.append(
            f"{'baseline':<16} {'n':>2}  {'mean Δacc':>10}  {'mean Δsamp':>11}  {'win/tie/loss':>12}"
        )
        for p in pairs:
            samp = f"{p['mean_delta_samples']:+.2f}" if p["mean_delta_samples"] is not None else "—"
            lines.append(
                f"{p['baseline']:<16} {p['n_seeds']:>2}  {p['mean_delta_acc']:>+10.4f}  {samp:>11}  "
                f"{p['win']}/{p['tie']}/{p['loss']:>}"
            )
        lines.append(f"(Δ = {HEADLINE_METHOD} − baseline; positive favors {HEADLINE_METHOD})")

    if agg["integrity"]["warnings"]:
        lines.append("")
        lines.append("integrity warnings:")
        for w in agg["integrity"]["warnings"]:
            lines.append(f"  ! {w}")
    return "\n".join(lines)


def write_aggregate(agg: dict[str, Any], *, base_dir: str = "experiments") -> str:
    """Write the aggregate to ``<base_dir>/<dataset>/aggregate.json`` and return the path."""
    path = os.path.join(base_dir, str(agg["dataset"]), "aggregate.json")
    write_artifact(agg, path)
    return path


def aggregate_and_write(
    dataset: str,
    *,
    base_dir: str = "experiments",
    seeds: Optional[list[int]] = None,
) -> tuple[dict[str, Any], str]:
    """Aggregate, persist ``aggregate.json``, and return ``(aggregate, path)`` — the wiring entry
    point."""
    agg = aggregate(dataset, base_dir=base_dir, seeds=seeds)
    path = write_aggregate(agg, base_dir=base_dir)
    return agg, path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Aggregate a multi-seed IRFS sweep across seeds.")
    parser.add_argument("--dataset", default=os.environ.get("IRFS_DATASET", "wdbc"))
    parser.add_argument("--base-dir", default="experiments")
    parser.add_argument(
        "--seeds", default=None, help="Comma-separated seeds to include (default: all on disk)."
    )
    args = parser.parse_args()
    seed_filter = [int(s) for s in args.seeds.split(",")] if args.seeds else None

    agg, path = aggregate_and_write(args.dataset, base_dir=args.base_dir, seeds=seed_filter)
    print(format_aggregate_table(agg))
    print(f"\naggregate written: {path}")
