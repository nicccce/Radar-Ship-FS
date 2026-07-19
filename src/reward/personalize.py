"""Per-agent reward personalizer (COMP-009) — splits the overall reward across agents.

Turns the single overall reward ``(Acc − βR)`` (COMP-008 / :func:`reward.overall.overall_reward`)
into a per-agent reward vector, implementing the reference method's two personalization schemes
(Section 3.3) selected by ``config.reward_scheme``:

- **Decision-Tree importance (3.3.1, headline / ASM-001).** ``r_i = I_i · (Acc − βR)`` for a selected
  agent, where ``I_i`` is the feature's Decision-Tree importance from the shared probe; ``0`` for a
  deselected agent.
- **Historical selection frequency (3.3.2, alternate / Q-001).** ``r_i = W_i · (Acc − βR)`` for a
  selected agent, where ``W_i = (Σ m_i) / Σ_j (Σ m_j)`` is the agent's share of all historical
  selections; ``0`` for a deselected agent.

In both schemes deselected agents receive **exactly zero** regardless of their weight — the result is a
zero vector with values written only at the selected positions, so a feature that is frequently selected
historically but *not* selected this step still earns nothing this step (the reference's ``a_i^t = 0``
case).

**Scope boundary.** This module owns the weighting math and scheme selection only. It does not store
rewards to the engine's experience memory, and it does not fetch the historical action records itself —
the ``selection_counts`` for the frequency scheme are supplied by the caller (TASK-405, which owns the
engine binding and has the experience memory). The correlation definition inside ``(Acc − βR)`` lives in
COMP-008, not here.

**Leakage safety (REQ-013).** Inherited from :func:`overall_reward` (validation accuracy, train-based
correlation) and the train-fit probe importances; no test partition is read. The transformation consumes
no randomness — pure and deterministic (CON-003).

Satisfies COMP-009 -> REQ-009 (AC-004 with COMP-008).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

from reward.overall import overall_reward

if TYPE_CHECKING:
    from harness.contract import SelectionContext


def _scheme_weights(
    scheme: str,
    selected: Sequence[int],
    context: "SelectionContext",
    selection_counts: Optional[Sequence[float]],
) -> np.ndarray:
    """Full-length per-agent weight vector for the configured scheme (zeros off-support).

    ``dt_importance`` reads Decision-Tree importances from the shared probe (already zero for
    unselected features); ``frequency`` normalizes the supplied historical selection counts into
    shares.
    """
    n_features = context.n_features

    if scheme == "dt_importance":
        return context.probe.probe(selected, context.split.validation).feature_importances

    if scheme == "frequency":
        if selection_counts is None:
            raise ValueError("frequency reward scheme requires selection_counts (historical action records)")
        counts = np.asarray(selection_counts, dtype=float)
        if counts.shape != (n_features,):
            raise ValueError(f"selection_counts must have length {n_features} (one per feature)")
        total = counts.sum()
        if total == 0.0:  # no history yet (e.g. the first step) — no agent has earned a share
            return np.zeros(n_features, dtype=float)
        return counts / total

    raise ValueError(f"unknown reward_scheme {scheme!r}; expected 'dt_importance' or 'frequency'")


def per_agent_reward_vector(
    selected: Sequence[int],
    context: "SelectionContext",
    *,
    selection_counts: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Per-agent reward vector: ``weight_i · overall`` for selected agents, ``0`` for deselected.

    The scheme is ``context.config.reward_scheme``. ``selection_counts`` (one historical selection
    count per feature) is required by the ``frequency`` scheme and ignored by ``dt_importance``.
    Returns a length-``context.n_features`` array.
    """
    overall = overall_reward(selected, context)
    weights = _scheme_weights(context.config.reward_scheme, selected, context, selection_counts)

    rewards = np.zeros(context.n_features, dtype=float)
    for agent in sorted({int(s) for s in selected}):  # zero stays for every deselected agent
        rewards[agent] = weights[agent] * overall
    return rewards
