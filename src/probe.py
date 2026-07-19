"""Decision-Tree probe (COMP-003) — the single shared scoring service (COMPAT-001).

Turns any feature subset plus an evaluation partition into classification accuracy, per-feature
importances, and the fitted tree. Every later-phase consumer (classical eliminator, trainers, state
representation, reward, exploration loop) reads its Decision-Tree outputs from this one probe rather
than fitting its own (DEC-002), so the call signature ``probe(subset, eval_partition)`` and the
``ProbeResult`` shape must stay stable across PHASE-002–005.

Accuracy contract (DEC-005): the tree is fit on the split's ``train`` partition (restricted to the
subset) and scored on the given ``eval_partition`` — validation during exploration (per-step
reward), test only for final metrics. The probe is bound to ``train`` at construction, so the per-
call signature carries only ``subset`` and the partition to score against.

Satisfies COMP-003 -> REQ-003.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from config import IrfsConfig
from data.splitter import Partition
from rng import SeededRng


class ProbeResult(NamedTuple):
    """One Decision-Tree scoring of a feature subset.

    ``feature_importances`` has length equal to the total feature count, with zeros at unselected
    positions and sklearn's importances (summing to 1.0) at selected ones. ``tree`` is the fitted
    estimator; ``tree.tree_`` is the raw sklearn tree structure.
    """

    accuracy: float
    feature_importances: np.ndarray
    tree: DecisionTreeClassifier


class DecisionTreeProbe:
    """Shared Decision-Tree probe bound to a training partition.

    The ``random_state`` is drawn once from the single shared RNG (CON-003) and reused
    for every call, so :meth:`probe` is a deterministic pure function of
    ``(subset, eval_partition)``.

    Because it is a pure function, identical ``(subset, eval_partition)`` calls are
    memoized: the fitted tree and its scoring are computed once and the same
    :class:`ProbeResult` is returned on repeat. This removes the redundant refit when one
    step both scores a subset and feeds it to the reward (each calls the probe with the
    same subset and validation partition). The cache lives on the probe instance — one per
    run — so it is bounded by the run's distinct ``(subset, partition)`` pairs and is
    discarded with the probe; results are read-only by every call site, so sharing one
    instance is safe.
    """

    def __init__(self, train: Partition, config: IrfsConfig, rng: SeededRng) -> None:
        self._train = train
        self._config = config
        self._random_state = int(rng.numpy.integers(0, 2**32))
        # Memo keyed on the exact subset bytes + the eval partition's row indices (content,
        # not object identity, so it is robust to id reuse). Train is fixed at construction.
        self._cache: dict[tuple[bytes, bytes], ProbeResult] = {}

    def probe(self, subset: Sequence[int], eval_partition: Partition) -> ProbeResult:
        """Score ``subset`` by fitting a Decision Tree on train and evaluating it.

        The tree is fit on the training partition restricted to ``subset`` and scored on
        ``eval_partition`` restricted to the same columns. Returns classification accuracy, a full-
        length per-feature importance vector (zeros for unselected features), and the fitted tree.
        ``subset`` is a non-empty sequence of column indices; an empty subset raises ``ValueError``.
        """
        subset_idx = np.asarray(subset, dtype=int)
        if subset_idx.size == 0:
            raise ValueError("subset must contain at least one feature index")

        cache_key = (subset_idx.tobytes(), np.asarray(eval_partition.indices).tobytes())
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        clf = DecisionTreeClassifier(random_state=self._random_state)
        clf.fit(self._train.X[:, subset_idx], self._train.y)

        accuracy = float(clf.score(eval_partition.X[:, subset_idx], eval_partition.y))

        n_features_total = self._train.X.shape[1]
        feature_importances = np.zeros(n_features_total, dtype=float)
        feature_importances[subset_idx] = clf.feature_importances_

        result = ProbeResult(
            accuracy=accuracy,
            feature_importances=feature_importances,
            tree=clf,
        )
        self._cache[cache_key] = result
        return result
