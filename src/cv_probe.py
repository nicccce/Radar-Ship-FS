"""Cross-validated Decision-Tree probe used by the stage-2 RL protocol."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from config import IrfsConfig
from data.splitter import Partition
from probe import ProbeResult
from rng import SeededRng


class CrossValidatedDecisionTreeProbe:
    """Score subsets by stratified inner CV and expose one full-development feedback tree."""

    def __init__(
        self,
        development: Partition,
        config: IrfsConfig,
        rng: SeededRng,
        *,
        n_splits: int,
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        _, class_counts = np.unique(development.y, return_counts=True)
        if class_counts.size < 2 or int(class_counts.min()) < n_splits:
            raise ValueError("every class must contain at least n_splits development rows")

        self._development = development
        self._config = config
        self._n_splits = int(n_splits)
        self._random_state = int(rng.numpy.integers(0, 2**32))
        splitter = StratifiedKFold(
            n_splits=self._n_splits,
            shuffle=True,
            random_state=self._random_state,
        )
        self._folds = tuple(
            (train_idx.astype(int), held_out_idx.astype(int))
            for train_idx, held_out_idx in splitter.split(development.X, development.y)
        )
        self._cache: dict[bytes, tuple[ProbeResult, tuple[float, ...]]] = {}

    @property
    def n_splits(self) -> int:
        return self._n_splits

    def _validate_eval_partition(self, eval_partition: Partition) -> None:
        if not np.array_equal(eval_partition.indices, self._development.indices):
            raise ValueError(
                "cross-validated probe accepts only its bound development partition; "
                "held-out test scoring requires a separate final DecisionTreeProbe"
            )

    def probe(self, subset: Sequence[int], eval_partition: Partition) -> ProbeResult:
        """Return mean inner-CV accuracy plus a full-development tree for IRFS feedback."""
        self._validate_eval_partition(eval_partition)
        subset_idx = np.asarray(subset, dtype=int)
        if subset_idx.size == 0:
            raise ValueError("subset must contain at least one feature index")
        if subset_idx.min() < 0 or subset_idx.max() >= self._development.X.shape[1]:
            raise ValueError("subset contains an out-of-range feature index")

        cache_key = subset_idx.tobytes()
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached[0]

        fold_accuracies: list[float] = []
        for train_idx, held_out_idx in self._folds:
            tree = DecisionTreeClassifier(random_state=self._random_state)
            tree.fit(self._development.X[train_idx][:, subset_idx], self._development.y[train_idx])
            fold_accuracies.append(
                float(
                    tree.score(
                        self._development.X[held_out_idx][:, subset_idx],
                        self._development.y[held_out_idx],
                    )
                )
            )

        full_tree = DecisionTreeClassifier(random_state=self._random_state)
        full_tree.fit(self._development.X[:, subset_idx], self._development.y)
        feature_importances = np.zeros(self._development.X.shape[1], dtype=float)
        feature_importances[subset_idx] = full_tree.feature_importances_
        result = ProbeResult(
            accuracy=float(np.mean(fold_accuracies)),
            feature_importances=feature_importances,
            tree=full_tree,
        )
        self._cache[cache_key] = (result, tuple(fold_accuracies))
        return result

    def fold_indices(self) -> list[dict[str, list[int]]]:
        """Return original development-row indices for every fixed inner fold."""
        return [
            {
                "fit": self._development.indices[train_idx].astype(int).tolist(),
                "held_out": self._development.indices[held_out_idx].astype(int).tolist(),
            }
            for train_idx, held_out_idx in self._folds
        ]

    def fold_accuracies(
        self,
        subset: Sequence[int],
        eval_partition: Partition,
    ) -> tuple[float, ...]:
        """Return memoized per-fold accuracies for plotting and stability analysis."""
        self.probe(subset, eval_partition)
        subset_idx = np.asarray(subset, dtype=int)
        return self._cache[subset_idx.tobytes()][1]
