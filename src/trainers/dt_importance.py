"""Decision-Tree-importance trainer (COMP-003) — the structural teaching signal that uses the tree.

The second interactive teaching signal (REQ-003 / FLOW-002), the sibling of the relevance trainer
(COMP-002). Given the per-step agent partition (COMP-001 / :mod:`trainers.classify`), it advises a
*hesitant* agent to select its feature when that feature's Decision-Tree importance — from a probe
fit on the *participated* features — exceeds the median importance of the *assertive* features.
Where the relevance trainer judges a feature in isolation (relevance to the label), this one judges
it *structurally*, through the Decision Tree the whole study is built around.

Importances source: the one shared probe (DEC-002 / :class:`probe.DecisionTreeProbe`).
``probe(subset, eval_partition)`` fits the tree on the *training* partition restricted to ``subset``
and returns a full-length importance vector (zeros off-subset); the importances are therefore a
property of the participated subset and the training data, independent of ``eval_partition``.
Because ``hesitant`` and ``assertive`` are both subsets of ``participated`` (COMP-001's partition),
every feature this rule reads has a real, non-zero-eligible importance from that one fit.

Advice rule (the reference fixes "exceeds the median importance of the assertive features"; the
strictness is the recorded choice — strict ``>``, consistent with the relevance trainer): advise a
hesitant agent to ``ACTION_SELECT`` when its feature's importance is strictly greater than the
median importance of the assertive features. With no assertive features (no baseline) or no hesitant
features, no advice is given.

Leakage safety (REQ-013 / AC-007): the importances come from the train-fit probe; the
``eval_partition`` passed is the non-test validation partition and affects only accuracy, which this
trainer ignores — so no test data, and indeed no per-step decision data beyond train, reaches the
advice. No randomness is consumed (the probe's tree is deterministic), so the advice is
deterministic under the seed.

Satisfies COMP-003 -> REQ-003, REQ-013.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Sequence

import numpy as np

from engine.policy import ACTION_SELECT

if TYPE_CHECKING:
    from harness.contract import SelectionContext
    from trainers.classify import AgentClassification


def _advise_from_importances(
    importances: Sequence[float],
    assertive: Sequence[int],
    hesitant: Sequence[int],
) -> Dict[int, int]:
    """Advise hesitant agents whose importance strictly exceeds the assertive median to
    ``ACTION_SELECT``.

    Pure comparison rule, separated from the probe fit so it can be checked on crafted importance
    vectors. Returns ``{feature: ACTION_SELECT}`` for each advised hesitant feature; an empty
    ``assertive`` set yields no advice (no baseline), and a hesitant feature at or below the median
    is omitted (strict ``>``).
    """
    if len(assertive) == 0:
        return {}
    threshold = float(np.median([importances[a] for a in assertive]))
    return {h: ACTION_SELECT for h in hesitant if importances[h] > threshold}


class DTImportanceTrainer:
    """Advise hesitant agents whose tree importance beats the assertive median (COMP-003).

    Exposes :meth:`advise`, taking the per-step :class:`~trainers.classify.AgentClassification` and
    the shared context, fitting the shared probe on the participated features, and returning the
    advised actions for the hesitant agents it judges structurally strong. Stateless — importances
    depend on the per-step participated set, and the probe memoizes its own fits.
    """

    def advise(self, classification: "AgentClassification", context: "SelectionContext") -> Dict[int, int]:
        """Return ``{feature: ACTION_SELECT}`` for hesitant agents whose importance beats the
        assertive median.

        Fits the shared probe on the participated features for the importance vector, then advises
        every hesitant feature whose importance is strictly above the median importance of the
        assertive features. Returns no advice when there are no participated, no assertive, or no
        hesitant agents.
        """
        if not classification.participated or not classification.hesitant:
            return {}
        importances = context.probe.probe(
            classification.participated, context.split.validation
        ).feature_importances
        return _advise_from_importances(importances, classification.assertive, classification.hesitant)
