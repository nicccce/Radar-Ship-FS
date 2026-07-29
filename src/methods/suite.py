"""The method selection suite for the unified PHASE-005 comparison (TASK-501).

Assembles the four classical baselines (PHASE-003) and the selected reinforced methods (PHASE-004) into
one ordered list of ``(name, Selector)`` pairs on the single shared context, and runs them through the
existing equal-footing comparison (:func:`harness.comparison.run_comparison`) in one pass.
This is the single entry point the held-out scoring (TASK-503), the artifact emitter (TASK-504), and
the ``run_irfs`` wiring (TASK-505) consume — so the method assembly lives in exactly one place rather
than being re-enumerated per caller.

The classical four are plain constructors with no RNG subtlety; the reinforced methods are built by
:func:`methods.configure.build_reinforced_selectors`, which pins each to a snapshot of the
INITIAL context RNG so they reproduce their per-phase numbers regardless of ordering. The suite must
therefore be built BEFORE the comparison runs (the snapshot is captured at construction; the
orchestrator advances the shared RNG once selectors start).

Validation surface only (REQ-010 / RISK-002): every method is scored on ``split.validation`` through
the shared probe by the orchestrator; final test scoring happens later. No method internal is touched
— each is consumed as-is through the common subset contract (COMPAT-001). This
module deliberately owns the concrete-method imports the import-light :mod:`harness.comparison`
must not carry.

Satisfies COMP-022 (orchestration portion) -> REQ-018; consumes COMP-004 windowed metrics via
:func:`run_comparison`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from harness.comparison import run_comparison
from methods.configure import build_reinforced_selectors
from methods.dt_rfe import DTImportanceEliminator
from methods.l1 import L1Selector
from methods.mrmr import MRMRSelector
from methods.relevance_topk import RelevanceTopKSelector

if TYPE_CHECKING:
    from harness.comparison import ComparisonResult
    from harness.contract import SelectionContext, Selector
    from harness.orchestrator import MethodOrchestrator


CLASSICAL_METHOD_NAMES: tuple[str, ...] = ("relevance_topk", "dt_rfe", "mrmr", "l1")


def build_classical_selectors() -> "list[tuple[str, Selector]]":
    """The four classical baselines as ``(name, Selector)`` pairs, in the canonical PHASE-003 order.

    Plain constructors with no RNG dependence at construction; ``mrmr`` is the pinned implementation
    and ``l1`` is variable-size. They produce no per-step series, so they carry no windowed entry.
    """
    return [
        ("relevance_topk", RelevanceTopKSelector()),
        ("dt_rfe", DTImportanceEliminator()),
        ("mrmr", MRMRSelector()),
        ("l1", L1Selector()),
    ]


def build_method_suite(
    context: "SelectionContext",
    *,
    include_diagnostic_ablations: bool = False,
    on_step=None,
) -> "list[tuple[str, Selector]]":
    """The ``(name, Selector)`` pairs: the four classical then the selected reinforced methods.

    Call this BEFORE running the comparison: the reinforced methods pin to a snapshot of ``context.rng``
    captured here, which must be the initial post-build state (the orchestrator advances the shared
    RNG once selectors start). ``on_step`` is the reinforced per-step hook
    ``on_step(name, step, budget, accuracy, best)``, forwarded verbatim; the classical baselines
    produce no per-step series and ignore it.
    """
    return build_classical_selectors() + build_reinforced_selectors(
        context,
        include_diagnostic_ablations=include_diagnostic_ablations,
        on_step=on_step,
    )


def run_full_comparison(
    orchestrator: "MethodOrchestrator",
    *,
    include_diagnostic_ablations: bool = False,
    on_method=None,
    on_step=None,
) -> "ComparisonResult":
    """Run the headline method suite on the orchestrator's shared context and summarize one
    comparison.

    Builds the method suite on ``orchestrator.context`` (capturing the initial reinforced RNG
    snapshot) and drives :func:`run_comparison` once, returning the :class:`ComparisonResult`
    carrying every included method's subset/size, the reinforced per-step series, and windowed
    Best/Average for included reinforced methods (``compare`` adds a windowed entry for each series-
    bearing run). Validation surface only — test is not used here.

    ``on_method`` (per-selector start/done) and ``on_step`` (reinforced per-step) are optional,
    observational progress hooks for the long-running selectors; default ``None`` -> silent and bit-
    identical.
    """
    suite = build_method_suite(
        orchestrator.context,
        include_diagnostic_ablations=include_diagnostic_ablations,
        on_step=on_step,
    )
    return run_comparison(orchestrator, suite, on_method=on_method)
