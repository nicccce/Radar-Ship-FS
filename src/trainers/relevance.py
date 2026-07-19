"""Relevance trainer (COMP-002) — advise hesitant agents toward features strong by training
relevance.

One of the two interactive teaching signals (REQ-002 / FLOW-002). Given the per-step agent partition
(COMP-001 / :mod:`trainers.classify`), it advises a *hesitant* agent to select its feature when that
feature looks comparatively strong — by relevance to the label, measured on the training partition —
against the features the *assertive* agents are keeping.

Relevance metric: mutual information with the label (scikit-learn ``mutual_info_classif``), the same
training-data relevance measure the classical relevance baseline uses (COMP-003 /
:mod:`methods.relevance_topk`), so the trainer and that baseline judge "relevance" identically. The
estimator's ``random_state`` is drawn once from the single shared RNG (CON-003); the relevance
vector is a fixed property of the training partition, so it is computed once and cached for the run
rather than recomputed (and re-drawn) every step.

Advice rule (the within-component reading of "comparatively strong relative to the assertive
features"; the reference fixes the framing, this is the recorded threshold choice — mirrors
COMP-003's above-median rule for consistency): advise a hesitant agent to ``ACTION_SELECT`` when its
feature's relevance is **strictly greater than the median relevance of the assertive features**.
With no assertive features there is no comparison baseline, so no advice is given. A hesitant
feature at or below the median is left alone.

Leakage safety (REQ-013 / AC-007): relevance is read only from ``context.split.train`` — the
validation and test partitions are never touched, so the advice is a non-test signal. No randomness
beyond the single seeded draw is consumed, so the advice is deterministic under the seed.

Satisfies COMP-002 -> REQ-002, REQ-013.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Sequence

import numpy as np
from sklearn.feature_selection import mutual_info_classif

from engine.policy import ACTION_SELECT

if TYPE_CHECKING:
    from harness.contract import SelectionContext
    from trainers.classify import AgentClassification


def _advise_from_relevance(
    relevance: Sequence[float],
    assertive: Sequence[int],
    hesitant: Sequence[int],
) -> Dict[int, int]:
    """Advise hesitant agents whose relevance strictly exceeds the assertive median to
    ``ACTION_SELECT``.

    Pure comparison rule, separated from the relevance computation so it can be checked on crafted
    relevance vectors. Returns ``{feature: ACTION_SELECT}`` for each advised hesitant feature; an
    empty ``assertive`` set yields no advice (no baseline), and a hesitant feature at or below the
    median is omitted (strict ``>``).
    """
    if len(assertive) == 0:
        return {}
    threshold = float(np.median([relevance[a] for a in assertive]))
    return {h: ACTION_SELECT for h in hesitant if relevance[h] > threshold}


class RelevanceTrainer:
    """Advise hesitant agents toward comparatively-relevant features versus the assertive set
    (COMP-002).

    Exposes :meth:`advise`, taking the per-step :class:`~trainers.classify.AgentClassification` and
    the shared context, and returning the advised actions for the hesitant agents it judges strong.
    The training-partition relevance vector is computed once per run and cached.
    """

    def __init__(self) -> None:
        self._relevance: Optional[np.ndarray] = None

    def _relevance_vector(self, context: "SelectionContext") -> np.ndarray:
        """Per-feature mutual-information relevance on the training partition (computed once,
        cached)."""
        if self._relevance is None:
            train = context.split.train
            random_state = int(context.rng.numpy.integers(0, 2**32))
            self._relevance = mutual_info_classif(train.X, train.y, random_state=random_state)
        return self._relevance

    def advise(self, classification: "AgentClassification", context: "SelectionContext") -> Dict[int, int]:
        """Return ``{feature: ACTION_SELECT}`` for hesitant agents strong by relevance vs the
        assertive median.

        Relevance is the training-partition mutual information; the threshold is the median
        relevance of the assertive features. Hesitant features strictly above it are advised to
        select; all others (and every non-hesitant agent) receive no advice.
        """
        relevance = self._relevance_vector(context)
        return _advise_from_relevance(relevance, classification.assertive, classification.hesitant)
