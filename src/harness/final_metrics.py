"""Final validation/test metrics.

After selection, score every selected subset on test through the shared Decision-Tree probe and
pair that result with the validation accuracy already produced during selection. This module
produces the numbers; artifact persistence is handled elsewhere.
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
    the test partition through the shared probe.
    """

    name: str
    size: int
    validation: float
    test: float


class FinalMetricsResult(NamedTuple):
    """The held-out scoring outcome for one run: a ``(val, test)`` pair per method + the test size.

    ``per_method`` preserves the comparison's method order. ``test_n_samples`` is the test
    partition's sample count.
    """

    per_method: tuple[MethodFinalMetrics, ...]
    test_n_samples: int


def score_final_metrics(context: "SelectionContext", comparison: "ComparisonResult") -> FinalMetricsResult:
    """Score every method's final subset on test.

    ``comparison`` already carries each final subset and its validation accuracy. No selection
    method is rerun here.
    """
    test_partition = context.split.test
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
