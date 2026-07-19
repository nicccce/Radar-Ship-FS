"""Run-artifact emitter (COMP-002) — classical-only slice.

Serializes a single run into one inspectable, reproducible artifact: the effective configuration,
the recorded seed, the dataset identity, and each method's selected subset and size. This task emits
the classical-only artifact (the relevance top-k method); the schema is deliberately forward-
compatible so PHASE-003 (TASK-214) completes it by appending the reinforced method's entry (with its
per-step accuracy series) and a top-level Best/ Average ``comparison`` — without reshaping the
``dataset`` / ``seed`` / ``config`` / ``methods`` keys established here.

The whole effective configuration is dumped (``dataclasses.asdict``), so the three reference-
unspecified parameters REQ-012/AC-009 calls out (exploration step budget, the correlation-penalty
weight β, and the exploration parameter) surface in the configuration view automatically once the
engine introduces them — no schema change required.

Dataset identity is derived from the training partition and configuration only; the test partition
is never released here (it is reserved for final reported metrics).

Satisfies COMP-002 -> REQ-011 (partial — classical-only artifact; completed in PHASE-003).
"""

from __future__ import annotations

import dataclasses
import json
import os
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from harness.comparison import ComparisonResult
from harness.contract import SelectionContext
from harness.fidelity import fidelity_notes_as_dicts
from harness.final_metrics import FinalMetricsResult, final_metrics_to_dict
from harness.orchestrator import MethodRun


def _method_to_dict(run: MethodRun) -> dict[str, Any]:
    """Serialize one method's run record to a JSON-ready dict.

    ``per_step`` becomes a list of ``{"subset", "accuracy"}`` entries; it is empty for a single-shot
    classical method and carries the reinforced engine's series unchanged when populated (TASK-214
    reuses this helper as-is).
    """
    return {
        "name": run.name,
        "selected": [int(i) for i in run.selected],
        "size": int(run.size),
        "accuracy": float(run.accuracy),
        "per_step": [
            {"subset": [int(i) for i in step.subset], "accuracy": float(step.accuracy)}
            for step in run.per_step
        ],
    }


def _comparison_to_dict(comparison: ComparisonResult) -> dict[str, Any]:
    """Serialize the two-method Best/Average comparison to a JSON-ready dict.

    ``window`` becomes ``[start, end]`` (or ``None`` for the full series); ``windowed`` maps each
    *series-bearing* method's name to its ``{"best", "average"}`` pair. A single-shot classical
    method is absent here by construction (it produced no series) and is represented in ``methods``
    by its single ``accuracy`` — the equal-footing decision fixed in TASK-213, preserved verbatim in
    the artifact.
    """
    return {
        "window": list(comparison.window) if comparison.window is not None else None,
        "windowed": {
            name: {"best": float(metrics.best), "average": float(metrics.average)}
            for name, metrics in comparison.windowed.items()
        },
    }


def build_artifact(
    runs: Sequence[MethodRun],
    context: SelectionContext,
    comparison: Optional[ComparisonResult] = None,
) -> dict[str, Any]:
    """Assemble the JSON-serializable run artifact from method runs and the shared context.

    Captures the dataset identity (name and derived feature/class counts), the recorded seed, the
    full effective configuration, and each method's serialized run. Identity is derived from
    ``context`` (configuration + training partition) without releasing the test partition. The
    returned dict is plain JSON-serializable data.

    When ``comparison`` is provided (the completed PHASE-003 artifact, TASK-214), the two-method
    Best/Average summary is appended as the last top-level ``comparison`` key; when it is ``None``
    the artifact is byte-for-byte the classical-only / partial form (TASK-204). All other keys are
    unchanged either way, so the persisted schema the epic's PHASE-005 consumes only ever grows
    additively.
    """
    config = context.config
    artifact: dict[str, Any] = {
        "dataset": {
            "name": config.dataset,
            "n_features": int(context.n_features),
            "n_classes": int(np.unique(context.split.train.y).size),
        },
        "seed": int(context.rng.seed),
        "config": dataclasses.asdict(config),
        "methods": [_method_to_dict(run) for run in runs],
    }
    if comparison is not None:
        artifact["comparison"] = _comparison_to_dict(comparison)
    return artifact


def write_artifact(artifact: Mapping[str, Any], path: str | os.PathLike[str]) -> None:
    """Write ``artifact`` to ``path`` as indented JSON.

    The sole side-effecting entry point; nothing is persisted unless this is called with a caller-
    chosen path. Key order is preserved so the artifact reads in the schema's natural order
    (dataset, seed, config, methods).
    """
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(artifact, fh, indent=2, sort_keys=False)


# --- The complete PHASE-005 run artifact (COMP-023) — TASK-504 -----------------------------------
#
# The completed run emits TWO files under experiments/<dataset>/seed-<n>/: a selection artifact (the
# validation surface plus provenance) and a test artifact (the held-out surface). This is a recorded
# departure from the spec's single self-contained JSON — isolating test in its own file physically
# mirrors the gated one-time release (TASK-503), and the per-seed folder makes a future multi-seed run
# additive (fidelity note ``artifact-multi-file-layout``). The classical-only / two-method
# ``build_artifact`` above is left untouched, so the PHASE-002 ``run.json`` format stays a strict
# subset of the selection artifact (same dataset/seed/config/methods/comparison keys, same order).


def build_selection_artifact(
    comparison: ComparisonResult,
    context: SelectionContext,
    *,
    mrmr_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble the selection-surface artifact: the full comparison plus run provenance.

    Reuses :func:`build_artifact` for the established
    dataset/seed/config/methods(+per_step)/comparison block (so those keys are byte-identical to the
    PHASE-002 artifact), then appends the pinned mRMR implementation identity (injected by the
    caller — the harness imports no method module) and the recorded fidelity notes. Plain JSON-
    serializable data; the validation accuracy lives in ``methods``/``comparison`` here, while the
    held-out test accuracy lives in the test artifact.
    """
    artifact = build_artifact(comparison.runs, context, comparison=comparison)
    artifact["mrmr_identity"] = dict(mrmr_identity)
    artifact["fidelity_notes"] = fidelity_notes_as_dicts()
    return artifact


def build_test_artifact(final_metrics: FinalMetricsResult, context: SelectionContext) -> dict[str, Any]:
    """Assemble the held-out test artifact: each method's ``(validation, test)`` pair + the test
    size.

    Carries the dataset name and seed so the file is self-identifying, then the per-method dual metrics
    from :func:`~harness.final_metrics.final_metrics_to_dict` (TASK-503). Joined back to the
    selection artifact by method name. Plain JSON-serializable data.
    """
    config = context.config
    return {
        "dataset": config.dataset,
        "seed": int(context.rng.seed),
        **final_metrics_to_dict(final_metrics),
    }


def write_run_artifacts(
    selection: Mapping[str, Any],
    test: Mapping[str, Any],
    context: SelectionContext,
    *,
    base_dir: str | os.PathLike[str] = "experiments",
) -> tuple[str, str]:
    """Write the two run artifacts to ``base_dir/<dataset>/seed-<n>/`` and return their paths.

    The per-dataset, per-seed directory is derived from the effective configuration, so a
    Parkinson's run or a different seed never clobbers a WDBC run (and a future multi-seed run adds
    sibling folders, no migration). ``selection.json`` and ``test.json`` are written with the same
    indented, key-order- preserving dump style as :func:`write_artifact`. This is the sole side-
    effecting entry point.
    """
    config = context.config
    run_dir = os.path.join(os.fspath(base_dir), str(config.dataset), f"seed-{int(context.rng.seed)}")
    os.makedirs(run_dir, exist_ok=True)
    selection_path = os.path.join(run_dir, "selection.json")
    test_path = os.path.join(run_dir, "test.json")
    write_artifact(selection, selection_path)
    write_artifact(test, test_path)
    return selection_path, test_path
