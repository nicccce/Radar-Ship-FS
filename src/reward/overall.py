"""Overall reward calculator (COMP-008) — accuracy minus configured penalties.

Computes the single scalar reward the per-agent personalizer (COMP-009 / TASK-404) shapes into a
per-agent signal:

    reward = accuracy − β·correlation − λ·max(0, (|S|−budget)/budget)

where ``accuracy`` is the shared Decision-Tree probe's score for the subset on the **validation**
partition (the per-step, non-test evaluation surface, DEC-005) and the correlation penalty is the
mean absolute pairwise correlation of the selected features. The optional normalized over-budget term
is zero when ``feature_budget`` is unset or its weight λ is zero.

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

from reward.budget import over_budget_penalty
from state.graph import average_pairwise_abs_correlation, build_correlation_graph

if TYPE_CHECKING:
    from harness.contract import SelectionContext


def overall_reward(selected: Sequence[int], context: "SelectionContext") -> float:
    """Return accuracy minus correlation and optional normalized over-budget penalties.

    Accuracy is the shared probe's score on the validation partition; the correlation penalty is
    reused from the state block's correlation graph (DEC-003); β is
    ``context.config.correlation_penalty_weight``. The shared budget helper applies
    ``lambda*max(0,(|S|-budget)/budget)`` when enabled.
    """
    accuracy = context.probe.probe(selected, context.split.validation).accuracy
    average_correlation = average_pairwise_abs_correlation(build_correlation_graph(selected, context))
    beta = context.config.correlation_penalty_weight
    return float(accuracy - beta * average_correlation - over_budget_penalty(selected, context))
