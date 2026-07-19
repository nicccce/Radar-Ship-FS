"""Two-method comparison summarizer (COMP-001, comparison portion) — TASK-213.

Places both selection methods on equal footing and reduces the run into one comparison result. Equal
footing is already guaranteed upstream: every method satisfies the one common subset contract and is
scored on the *same* shared split + probe by the orchestrator (``MethodRun.accuracy`` is the same-
probe validation score for every method). This component adds only the cross-method summary — it
drives the external windowed Best/Average metrics (``metrics.compute_windowed_metrics``) over the
per-step accuracy series, and the per-step series exists only for the reinforced engine.

So Best/Average are computed for **series-bearing methods only**; the classical baseline, which
returns no per-step series, is represented on equal footing by its single validation accuracy and
carries no windowed entry. A single-shot method's lone accuracy is not an exploration Best/Average,
and ``compute_windowed_metrics`` rejects an empty series — so no degenerate one-element series is
synthesized to force a uniform shape.

The summary is pure data over already-produced :class:`MethodRun` records, kept separate from the
orchestrator (which stays metric-free) and reusing the pure metrics function verbatim — mirroring
how :mod:`metrics` is its own standalone component. TASK-214 serializes a :class:`ComparisonResult`
into the completed run artifact; TASK-215 drives two same-seed comparisons for the reproducibility
proof.

Satisfies COMP-001 -> REQ-002 (full two-method form); AC-001 (both-methods).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Optional, Sequence

from metrics import compute_windowed_metrics

if TYPE_CHECKING:
    from harness.contract import Selector
    from harness.orchestrator import MethodOrchestrator, MethodRun


class WindowedMetrics(NamedTuple):
    """Best and Average accuracy over the windowed per-step series of one method."""

    best: float
    average: float


class ComparisonResult(NamedTuple):
    """One run's two-method comparison on equal footing.

    ``runs`` are the per-method outcomes (both scored on the one shared split + probe). ``windowed``
    maps a method name to its :class:`WindowedMetrics`, and contains an entry only for methods that
    produced a per-step series (the reinforced engine); a single-shot classical method is absent
    here and is represented by its ``MethodRun.accuracy``. ``window`` records the ``(start, end)``
    slice applied (or ``None`` for the full series) so the artifact can disclose it (TASK-214).
    """

    runs: tuple["MethodRun", ...]
    windowed: dict[str, WindowedMetrics]
    window: Optional[tuple[int, int]]


def compare(
    runs: Sequence["MethodRun"],
    window: Optional[tuple[int, int]] = None,
) -> ComparisonResult:
    """Summarize per-method runs into a :class:`ComparisonResult` on equal footing.

    For each run that produced a per-step series, the windowed Best/Average metrics are computed
    over its per-step accuracies via :func:`compute_windowed_metrics` (honoring ``window``); runs
    with no series (classical single-shot) get no windowed entry. The runs are preserved in their
    given order, which is the orchestrator's invocation order.
    """
    windowed: dict[str, WindowedMetrics] = {}
    for run in runs:
        if run.per_step:
            best, average = compute_windowed_metrics([step.accuracy for step in run.per_step], window)
            windowed[run.name] = WindowedMetrics(best=best, average=average)
    return ComparisonResult(runs=tuple(runs), windowed=windowed, window=window)


def run_comparison(
    orchestrator: "MethodOrchestrator",
    selectors: Sequence[tuple[str, "Selector"]],
    *,
    on_method=None,
) -> ComparisonResult:
    """Run every selector once on the shared context, then summarize the comparison.

    Invokes ``orchestrator.run(selectors)`` a single time so both method families are carried
    through the common contract on the one shared split + probe (equal footing), reads the
    configured metric window from the orchestrator's shared context, and returns the :func:`compare`
    summary over the resulting runs.

    ``on_method`` is forwarded verbatim to :meth:`MethodOrchestrator.run` as an optional progress
    hook (default ``None`` → silent); it is observational and leaves results unchanged.
    """
    runs = orchestrator.run(selectors, on_method=on_method)
    window = orchestrator.context.config.metric_window
    return compare(runs, window)
