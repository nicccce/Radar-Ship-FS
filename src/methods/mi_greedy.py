"""Forward greedy selection guided by Mutual Information sorting.

Calculates Mutual Information for all features on the training set, sorts them
in descending order, and greedily adds them one by one if they improve the
cross-validated validation accuracy.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_selection import mutual_info_classif

from harness.contract import SelectionContext, SubsetSelection, make_selection

class MIGreedySelector:
    def select(self, context: SelectionContext) -> SubsetSelection:
        train = context.split.train
        random_state = int(context.rng.numpy.integers(0, 2**32))
        
        # 1. Mutual Information sorting
        relevance = mutual_info_classif(train.X, train.y, random_state=random_state)
        ranked_features = np.argsort(-relevance, kind="stable").tolist()
        
        # 2. Greedy forward selection
        selected = []
        best_acc = -1.0
        
        for feature in ranked_features:
            candidate = selected + [feature]
            # Probe evaluates on the development set's inner CV (handled by context.probe)
            result = context.probe.probe(candidate, context.split.validation)
            if result.accuracy > best_acc:
                selected.append(feature)
                best_acc = result.accuracy
                
        return make_selection(selected)
