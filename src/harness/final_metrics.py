"""Held-out test release + dual val/test metrics (COMP-022, held-out portion) — TASK-503.

The single, auditable place the gated test partition is touched. After a run has produced every
method's final subset on the validation surface (TASK-501,
:func:`methods.suite.run_full_comparison`), this module releases the test partition **exactly once**
via ``Split.release_test_for_final_metrics`` and scores each method's already-selected subset on it
through the shared Decision-Tree probe — pairing the validation accuracy that drove selection with
the honest held-out test accuracy.

Leakage invariant (REQ-002 / DEC-005 / AC-002 / RISK-002): the release is strictly post-selection
and operates only over subsets that already exist; nothing here re-runs a method or scores during
selection, so the test partition cannot influence any subset. ``release_test_for_final_metrics`` has
no other production caller — this is its first and only one (greppable for the AC-002 inspection).
The probe is fit on ``split.train`` and merely *evaluated* on test, and it memoizes by ``(subset,
partition)``, so scoring the final subsets on test is cheap and changes nothing about selection.

This module produces the numbers only; embedding them in the persisted artifact is TASK-504, and the
val-vs-test protocol caveat is already recorded by TASK-502 (``validation-step-protocol``).

Satisfies COMP-022 -> REQ-018 (held-out scoring portion); preserves AC-002 (leakage invariant).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from harness.comparison import ComparisonResult
    from harness.contract import SelectionContext


class MethodFinalMetrics(NamedTuple):
    """One method's subset size and paired accuracies: the validation score that drove selection,
    and the held-out test score.

    ``size`` is the method's selected feature count (carried so the test surface is self-contained —
    it matches the same method's ``size`` in the selection artifact). ``validation`` is taken
    verbatim from the method's :class:`~harness.orchestrator.MethodRun` (the in-run validation
    accuracy, TASK-501) — it is *not* recomputed here. ``test`` is the same final subset scored on
    the released test partition through the shared probe.
    """

    name: str
    size: int
    validation: float
    test: float


class FinalMetricsResult(NamedTuple):
    """The held-out scoring outcome for one run: a ``(val, test)`` pair per method + the test size.

    ``per_method`` preserves the comparison's method order. ``test_n_samples`` is the released test
    partition's sample count, surfaced here because this is the one place test is legitimately
    touched (so TASK-504 need not release it a second time).
    """

    per_method: tuple[MethodFinalMetrics, ...]
    test_n_samples: int


def score_final_metrics(context: "SelectionContext", comparison: "ComparisonResult") -> FinalMetricsResult:
    """Release the test partition once and score every method's final subset on it.

    Strictly post-selection: ``comparison`` already carries each method's final subset and its
    validation accuracy (TASK-501). This releases the gated test partition a single time and, for each
    method, scores its existing subset on test through the shared probe — pairing that held-out accuracy
    with the validation accuracy. No method is re-run and no selection path reaches test (AC-002).
    """
    test_partition = context.split.release_test_for_final_metrics()  # the one and only release (AC-002)
    per_method = tuple(
        MethodFinalMetrics(
            name=run.name,
            size=int(run.size),
            validation=float(run.accuracy),
            test=float(context.probe.probe(run.selected, test_partition).accuracy),
        )
        for run in comparison.runs
    )
    return FinalMetricsResult(
        per_method=per_method,
        test_n_samples=int(test_partition.y.shape[0]),
    )


def final_metrics_to_dict(result: FinalMetricsResult) -> dict[str, Any]:
    """Serialize the held-out metrics to a JSON-ready dict (round-trip safe).

    ``methods`` maps each method name to its ``{"validation", "test"}`` pair (order preserved as a
    list); ``test_n_samples`` carries the held-out size. All values are plain JSON types — this is
    the block TASK-504 embeds for the test surface, requiring no reshaping.
    """
    return {
        "test_n_samples": int(result.test_n_samples),
        "methods": [
            {
                "name": m.name,
                "size": int(m.size),
                "validation": float(m.validation),
                "test": float(m.test),
            }
            for m in result.per_method
        ],
    }
