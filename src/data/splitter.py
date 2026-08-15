"""Train/validation/test splitting."""

from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np
from sklearn.model_selection import train_test_split

from config import IrfsConfig
from data.loader import LoadedDataset
from rng import SeededRng


class Partition(NamedTuple):
    """One disjoint slice of the dataset (samples, not features)."""
    X: np.ndarray
    y: np.ndarray
    indices: np.ndarray
    feature_names: list[str]
    groups: Optional[np.ndarray] = None
    metadata: Optional[dict] = None


class Split(NamedTuple):
    """The three dataset partitions."""
    train: Partition
    validation: Partition
    test: Partition

    def replace_development_for_inner_cv(self, development: Partition) -> "Split":
        """Use all development rows for components backed by inner cross-validation."""
        return Split(train=development, validation=development, test=self.test)


def _draw_split_seed(rng: SeededRng) -> int:
    """Draw one ``random_state`` integer from the single shared RNG."""
    return int(rng.numpy.integers(0, 2**32))


def make_split(dataset: LoadedDataset, config: IrfsConfig, rng: SeededRng) -> Split:
    """Partition a dataset into train, validation, and test rows."""
    indices = np.arange(dataset.X.shape[0])

    if dataset.test_indices is not None:
        test_idx = np.asarray(dataset.test_indices, dtype=int)
        train_pool_idx = np.setdiff1d(indices, test_idx, assume_unique=True)
    else:
        # Fallback just in case, though radar ship always has test_indices
        train_pool_idx, test_idx = train_test_split(
            indices,
            test_size=config.test_fraction,
            random_state=_draw_split_seed(rng),
            stratify=dataset.y,
        )

    train_idx, val_idx = train_test_split(
        train_pool_idx,
        test_size=config.validation_fraction,
        random_state=_draw_split_seed(rng),
        stratify=dataset.y[train_pool_idx],
    )

    def partition(idx: np.ndarray) -> Partition:
        return Partition(
            X=dataset.X[idx],
            y=dataset.y[idx],
            indices=idx,
            feature_names=dataset.feature_names,
            groups=(dataset.groups[idx] if dataset.groups is not None else None),
            metadata=dataset.metadata,
        )

    return Split(
        train=partition(train_idx),
        validation=partition(val_idx),
        test=partition(test_idx),
    )
