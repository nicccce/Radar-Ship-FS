"""Overall reward calculator (COMP-009) — the provisional reward behind the seam.

Computes the single reward every selected agent learns from: downstream Decision-Tree accuracy
rewards good subsets, while redundancy among the chosen features is penalized. The value is

    reward = accuracy − β·correlation − λ·max(0, (|S|−budget)/budget)

where ``accuracy`` is the shared probe's score for the subset and ``average intra-subset
correlation`` is the mean absolute Pearson correlation over every pair of selected features. β is
the configurable correlation-penalty weight read from configuration (REQ-012). The optional budget
penalty is normalized by ``feature_budget``; with fewer than two selected features the correlation
term is ``0.0``.

Leakage invariant (REQ-010 / AC-007): both the accuracy and the correlation are computed **only**
from the validation partition. Test data is not used by this reward.

The reward is applied uniformly: the seam's ``agent`` argument is accepted (the engine passes it)
but ignored, so every selected agent receives the same value this phase (AC-006). PHASE-004's
personalized reward will read ``agent`` to differentiate — through the same signature, with no
engine change (RISK-004). This conforms to ``engine.seam.RewardFunction`` structurally.

Satisfies COMP-009 -> REQ-009, REQ-010, REQ-012, AC-006, AC-007. Conforms to REQ-014 (reward seam).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

from reward.budget import over_budget_penalty

if TYPE_CHECKING:
    from harness.contract import SelectionContext


def _abs_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Absolute Pearson correlation between two 1-D series, ``0.0`` if either is constant."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.std() == 0.0 or b.std() == 0.0:
        return 0.0
    return float(abs(np.corrcoef(a, b)[0, 1]))


def _average_pairwise_abs_corr(corr: np.ndarray, selected: Sequence[int]) -> float:
    """Mean of ``corr`` over every unordered pair of ``selected`` columns.

    ``corr`` is the precomputed ``|corr|`` matrix (``corr[i, j] == _abs_corr(X[:, i], X[:, j])``).
    Returns ``0.0`` when fewer than two features are selected (no pairs exist), so the penalty
    vanishes for singleton subsets rather than being undefined. The upper-triangle traversal visits
    the same pairs in the same (lexicographic) order as ``combinations`` did, so averaging the
    matrix entries reproduces the former per-pair result exactly.
    """
    idx = np.array(sorted({int(s) for s in selected}), dtype=int)
    if idx.size < 2:
        return 0.0
    sub = corr[np.ix_(idx, idx)]
    upper = np.triu_indices(idx.size, k=1)
    return float(np.mean(sub[upper]))


class OverallReward:
    """Uniform accuracy-minus-configured-penalties reward on the validation partition.

    Satisfies ``engine.seam.RewardFunction`` structurally: a single :meth:`reward` taking the shared
    context and an optional (ignored) ``agent``, returning one float applied to all agents.

    Because the value ignores ``agent`` (uniform this phase), it is a pure function of the subset,
    the validation partition, β, budget, and λ — so identical calls are memoized, exactly as
    :class:`~probe.DecisionTreeProbe` memoizes its fits. The exploration loop calls this once per
    feature-agent per step with the *same* subset (``n_features`` calls that all return one value);
    the first computes the pairwise-correlation penalty and the rest are cache hits, and a subset
    that recurs across steps is free. The cache lives on the instance — one per run — so it is
    bounded by distinct ``(subset, validation, β, budget, λ)`` keys and discarded with the reward.
    """

    def __init__(self) -> None:
        # Memo keyed on the subset bytes, the validation partition's row indices (content, not
        # object identity), and β — so a changed partition or penalty weight never returns a stale
        # value. Mirrors the probe's caching contract.
        self._cache: dict[tuple[bytes, bytes, float, Optional[int], float], float] = {}
        # Per validation-partition memo of the full feature ``|corr|`` matrix, built once from the
        # same pointwise :func:`_abs_corr` so the correlation penalty is bit-identical to the former
        # per-pair computation; keyed on the partition's row indices (content, not identity).
        self._corr_cache: dict[bytes, np.ndarray] = {}

    def _corr_matrix(self, validation) -> np.ndarray:
        """Full ``|corr|`` matrix of ``validation``'s feature columns, built once and memoized.

        ``corr[i, j] = _abs_corr(X[:, i], X[:, j])`` — the same estimator the per-pair penalty used,
        so :func:`_average_pairwise_abs_corr` reading from it reproduces the old values exactly.
        """
        key = np.asarray(validation.indices).tobytes()
        cached = self._corr_cache.get(key)
        if cached is not None:
            return cached

        X = validation.X
        n = X.shape[1]
        corr = np.zeros((n, n), dtype=float)
        for i in range(n):
            column_i = X[:, i]
            for j in range(i + 1, n):
                c = _abs_corr(column_i, X[:, j])
                corr[i, j] = corr[j, i] = c

        self._corr_cache[key] = corr
        return corr

    def reward(
        self,
        selected: Sequence[int],
        context: "SelectionContext",
        *,
        agent: Optional[int] = None,
    ) -> float:
        """Return accuracy minus correlation and optional normalized over-budget penalties.

        Accuracy is the shared probe's score for the subset on the validation partition; the
        correlation penalty is the mean absolute pairwise correlation of the selected features on
        that same partition; β is ``context.config.correlation_penalty_weight``. When configured,
        ``lambda*max(0,(|S|-budget)/budget)`` is also subtracted. ``agent`` is
        accepted for seam conformance but ignored — the reward is uniform across agents this phase.

        Memoized on ``(subset, validation indices, β, budget, λ)``: repeat calls with the same subset
        (every agent in a step) and recurring subsets across steps return the cached value, collapsing the
        per-agent recomputation of the O(k²) correlation penalty to once per distinct subset.
        """
        validation = context.split.validation
        beta = context.config.correlation_penalty_weight
        feature_budget = getattr(context.config, "feature_budget", None)
        budget_weight = float(getattr(context.config, "over_budget_penalty_weight", 0.0))

        subset_idx = np.asarray(selected, dtype=int)
        cache_key = (
            subset_idx.tobytes(),
            np.asarray(validation.indices).tobytes(),
            float(beta),
            feature_budget,
            budget_weight,
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        accuracy = context.probe.probe(selected, validation).accuracy
        average_correlation = _average_pairwise_abs_corr(self._corr_matrix(validation), selected)

        value = float(accuracy - beta * average_correlation - over_budget_penalty(selected, context))
        self._cache[cache_key] = value
        return value
