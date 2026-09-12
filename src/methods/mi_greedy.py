"""MI-ordered accept-if-improved selection.

This is deliberately not called forward greedy: it evaluates one precomputed
MI order and never enumerates all remaining candidates at a step.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_selection import mutual_info_classif

from harness.contract import SelectionContext, SubsetSelection, make_selection
from methods.sizing import configured_target_size


class MIOrderedImprovementSelector:
    """Walk the MI ranking once and accept a feature only on score improvement."""

    def select(self, context: SelectionContext) -> SubsetSelection:
        train = context.split.train
        random_state = int(context.rng.numpy.integers(0, 2**32))
        relevance = mutual_info_classif(train.X, train.y, random_state=random_state)
        ranked_features = np.argsort(-relevance, kind="stable").tolist()

        selected: list[int] = []
        best_accuracy = -1.0
        budget = configured_target_size(context)
        for feature in ranked_features:
            if len(selected) == budget:
                break
            candidate = tuple(sorted((*selected, feature)))
            accuracy = context.probe.probe(candidate, context.split.validation).accuracy
            if accuracy > best_accuracy + 1e-12:
                selected.append(feature)
                best_accuracy = float(accuracy)

        return make_selection(selected)


class MIGreedySelector(MIOrderedImprovementSelector):
    """Historical compatibility alias; use ``mi_ordered_accept`` in new work."""
