"""Overall reward calculator (COMP-008) — accuracy minus a correlation penalty.

Computes the single scalar reward the per-agent personalizer (COMP-009 / TASK-404) shapes into a
per-agent signal:

    reward = accuracy − β · (average intra-subset feature correlation)

where ``accuracy`` is the shared Decision-Tree probe's score for the subset on the **validation**
partition (the per-step, non-test evaluation surface, DEC-005) and the correlation penalty is the
mean absolute pairwise correlation of the selected features. β is ``config.correlation_penalty_weight``.

**Reuse boundary (DEC-003).** The correlation penalty is read from the state block's correlation graph
(:func:`state.graph.average_pairwise_abs_correlation`) rather than recomputed here, so the reward's
penalty and the state's correlation structure can never diverge. Note this sources the correlation from
the **training** partition (COMP-005's mandate), whereas PHASE-002's provisional
``engine.reward_overall`` recomputed it on validation; both partitions are non-test, so the per-step
signal stays leakage-safe (REQ-013). The tree edges are irrelevant to the average, so the lighter
:func:`state.graph.build_correlation_graph` is used (it yields the same correlation matrix as the
augmented build, with no extra probe fit).

With fewer than two selected features there are no pairs, so the penalty is ``0.0`` and the reward is the
bare accuracy. No randomness is consumed — the reward is a pure, deterministic function of
``(selected, context)`` (CON-003).

Satisfies COMP-008 -> REQ-008, REQ-013.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from state.graph import average_pairwise_abs_correlation, build_correlation_graph

if TYPE_CHECKING:
    from harness.contract import SelectionContext


def overall_reward(selected: Sequence[int], context: "SelectionContext") -> float:
    """Return ``accuracy − β·average-intra-subset-correlation`` for ``selected``.

    Accuracy is the shared probe's score on the validation partition; the correlation penalty is
    reused from the state block's correlation graph (DEC-003); β is
    ``context.config.correlation_penalty_weight``.
    """
    accuracy = context.probe.probe(selected, context.split.validation).accuracy
    average_correlation = average_pairwise_abs_correlation(build_correlation_graph(selected, context))
    beta = context.config.correlation_penalty_weight
    return float(accuracy - beta * average_correlation)
