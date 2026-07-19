"""Minimal per-feature state encoder (COMP-008) — the provisional state behind the seam.

Gives each feature-agent the small, fixed-length summary it reasons over: how relevant its feature
is to the label, and how redundant it is against the features currently selected. The encoded vector
is length-2 — ``[relevance, redundancy]`` — and that length is fixed regardless of how many features
are already selected (CON-005), so the policy-input shape is stable across exploration steps.

Both statistics are deterministic absolute Pearson correlations computed on the *training* partition
(REQ-008 / AC-005): relevance is the feature's correlation with the label; redundancy is the mean of
its correlations with the other currently-selected features (zero when none are selected). A
constant (zero-variance) column yields a correlation of ``0.0`` rather than NaN, so the vector is
always finite. No randomness is consumed — the encoder is a pure function of the feature, the
selected subset, and the training data — so the recorded seed reproduces it exactly.

This is the deliberately minimal state replaced behind the seam in PHASE-004 by the Decision-Tree-
structured state. It conforms to ``engine.seam.StateEncoder`` structurally (no inheritance) and is
consumed by the agents (TASK-208) and the exploration loop (TASK-210).

Satisfies COMP-008 -> REQ-008, AC-005. Conforms to REQ-014 (the state seam).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:
    from harness.contract import SelectionContext


def _abs_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Absolute Pearson correlation between two 1-D series, ``0.0`` if either is constant.

    A zero-variance column makes the Pearson coefficient undefined (NaN); treating that as ``0.0``
    keeps every state entry finite without special-casing at the call sites.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.std() == 0.0 or b.std() == 0.0:
        return 0.0
    return float(abs(np.corrcoef(a, b)[0, 1]))


class MinimalStateEncoder:
    """Fixed length-2 ``[relevance, redundancy]`` per-feature state on the training partition.

    Satisfies ``engine.seam.StateEncoder`` structurally: exposes a constant :attr:`dimension` and an
    :meth:`encode` returning a vector of that length, independent of the selected-subset size.

    The two statistics are absolute correlations between *fixed* training columns, so they are
    invariant across the exploration loop's steps; only the *set* of columns averaged over (the
    redundancy term's ``others``) changes. The encoder therefore precomputes, once per training
    partition, the feature-vs-label relevance vector and the full feature-by-feature ``|corr|``
    matrix — built from the very same :func:`_abs_corr` used pointwise before — and ``encode`` reads
    from them. This collapses the loop's O(steps · n_features²) ``np.corrcoef`` calls to one
    O(n_features²) build, and because the matrix entry ``corr[i, j]`` *is* ``_abs_corr(X[:,i],
    X[:,j])`` (symmetric, so the access order is irrelevant), the encoded vectors are bit-identical
    to the per-call computation — the seed reproduces the same series (CON-003). The cache lives on
    the instance, keyed on the partition's row indices (content, not identity), and is discarded
    with the encoder.
    """

    dimension = 2

    def __init__(self) -> None:
        # Per training-partition memo: indices bytes -> (relevance_vector, |corr| matrix).
        self._cache: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}

    def _prepare(self, train) -> tuple[np.ndarray, np.ndarray]:
        """Relevance vector and ``|corr|`` matrix for ``train``, built once and memoized.

        ``relevance[f] = _abs_corr(X[:, f], y)`` and ``corr[i, j] = _abs_corr(X[:, i], X[:, j])``,
        each using the same pointwise estimator the per-call path used — so downstream reads
        reproduce the old values exactly.
        """
        key = np.asarray(train.indices).tobytes()
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        X = train.X
        n = X.shape[1]
        relevance = np.array([_abs_corr(X[:, f], train.y) for f in range(n)], dtype=float)
        corr = np.zeros((n, n), dtype=float)
        for i in range(n):
            column_i = X[:, i]
            for j in range(i + 1, n):
                c = _abs_corr(column_i, X[:, j])
                corr[i, j] = corr[j, i] = c

        prepared = (relevance, corr)
        self._cache[key] = prepared
        return prepared

    def encode(
        self,
        feature: int,
        selected: Sequence[int],
        context: "SelectionContext",
    ) -> np.ndarray:
        """Encode ``feature`` against the ``selected`` subset as ``[relevance, redundancy]``.

        ``relevance`` is the feature's absolute correlation with the label; ``redundancy`` is the
        mean absolute correlation against the *other* currently-selected features (``0.0`` when no
        other feature is selected). Both are read from the precomputed train-partition relevance
        vector and ``|corr|`` matrix (see :meth:`_prepare`), so they equal the former per-call
        ``_abs_corr`` results exactly.
        """
        relevance_vector, corr = self._prepare(context.split.train)

        relevance = float(relevance_vector[feature])

        others = [int(s) for s in selected if int(s) != int(feature)]
        if others:
            redundancy = float(np.mean(corr[feature, others]))
        else:
            redundancy = 0.0

        return np.array([relevance, redundancy], dtype=float)
