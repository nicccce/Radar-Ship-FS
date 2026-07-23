"""Shared over-budget penalty for reinforced feature-selection rewards."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from harness.contract import SelectionContext


def over_budget_penalty(selected: Sequence[int], context: "SelectionContext") -> float:
    """Return ``lambda * max(0, (|S| - k) / k)`` for the configured budget.

    A missing budget or zero penalty weight preserves the historical reward exactly. The subset
    size is measured after de-duplication, matching the canonical selection contract.
    """
    budget = getattr(context.config, "feature_budget", None)
    weight = float(getattr(context.config, "over_budget_penalty_weight", 0.0))
    if budget is None or weight == 0.0:
        return 0.0
    if budget <= 0:
        raise ValueError("feature_budget must be positive when configured")
    if weight < 0.0:
        raise ValueError("over_budget_penalty_weight must be non-negative")

    selected_count = len({int(index) for index in selected})
    excess_ratio = max(0.0, (selected_count - budget) / budget)
    return float(weight * excess_ratio)
