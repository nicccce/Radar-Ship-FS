"""Relevance top-k classical selector (COMP-003).

The representative classical baseline for the thin equal-footing slice: rank every feature by its
relevance to the target label on the training partition and keep the top half, returned through the
common subset contract (TASK-201). This is the cheap reference the no-trainer reinforced method's
subset is compared against on equal footing.

Relevance metric: mutual information with the label (scikit-learn ``mutual_info_classif``), the
standard relevance-to-label measure in the feature-selection literature and consistent with the
relevance/redundancy framing the reinforced state uses. Its k-NN estimator is stochastic, so its
``random_state`` is drawn from the single shared RNG (CON-004) — the ranking is therefore
deterministic under the recorded seed. The specific classical baseline does not affect the harness
proof (ASM-001), so the metric is a reversible choice.

Leakage-safe (REQ-010): relevance is computed only on ``context.split.train``; the validation and
test partitions are never read here. Subset scoring is the orchestrator's job through the external
probe, not this selector's.

Satisfies COMP-003 -> REQ-003-A, REQ-003-B; AC-002.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_selection import mutual_info_classif

from harness.contract import SelectionContext, SubsetSelection, make_selection


class RelevanceTopKSelector:
    """Select the top-half features by mutual-information relevance to the label.

    Conforms structurally to :class:`harness.contract.Selector`: a single ``select`` taking the
    shared :class:`SelectionContext` and returning a :class:`SubsetSelection` with no per-step
    series (classical single-shot selection).
    """

    def select(self, context: SelectionContext) -> SubsetSelection:
        """Rank features by relevance on the training partition; keep the top half.

        Relevance is mutual information between each feature and the label, computed on the training
        partition only. The top ``n_features // 2`` features are selected; ties in relevance are
        broken by ascending feature index (stable sort), so the selected subset is fully determined
        by the recorded seed (REQ-003-B, AC-002).
        """
        train = context.split.train
        random_state = int(context.rng.numpy.integers(0, 2**32))
        relevance = mutual_info_classif(train.X, train.y, random_state=random_state)

        k = context.n_features // 2
        # Descending relevance; stable sort over index-ordered scores breaks ties by
        # ascending feature index, keeping the selection deterministic.
        ranked = np.argsort(-relevance, kind="stable")
        top_indices = ranked[:k]
        return make_selection(top_indices)
