"""Stable random feature identifiers used as a weak PPO tie-breaker."""

from __future__ import annotations

import numpy as np


def random_feature_ids(n_features: int, seed: int) -> np.ndarray:
    """Return a seeded permutation of ``1..n_features`` indexed by feature.

    ``seed`` is deliberately independent from the PPO training seed.  Reusing the
    same identifier seed across experiment seeds therefore gives every original
    feature the same preference in every run.
    """
    if n_features <= 0:
        raise ValueError("n_features must be positive")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x464944]))
    return rng.permutation(np.arange(1, n_features + 1, dtype=np.int64))


def normalized_feature_ids(feature_ids: np.ndarray) -> np.ndarray:
    """Normalize a validated ``1..n`` permutation to the interval ``(0, 1]``."""
    values = np.asarray(feature_ids, dtype=np.int64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("feature_ids must be a non-empty one-dimensional array")
    expected = np.arange(1, values.size + 1, dtype=np.int64)
    if not np.array_equal(np.sort(values), expected):
        raise ValueError("feature_ids must be a permutation of 1..n_features")
    return values.astype(np.float64) / float(values.size)


def feature_id_score(mask: np.ndarray, feature_ids: np.ndarray) -> float:
    """Mean normalized identifier of a subset, or zero for the empty subset.

    The mean keeps this tie-breaker cardinality-neutral.  For PPO's fixed-budget
    search it is also equivalent to ranking subsets by the sum of their IDs.
    """
    selected = np.asarray(mask, dtype=bool)
    scores = normalized_feature_ids(feature_ids)
    if selected.shape != scores.shape:
        raise ValueError("mask and feature_ids must have the same shape")
    if not np.any(selected):
        return 0.0
    return float(np.mean(scores[selected]))
