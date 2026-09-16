"""Train/validation/test splitting."""

from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from config import IrfsConfig
from data.loader import LoadedDataset
from rng import SeededRng


class Partition(NamedTuple):
    """One disjoint slice of the dataset (samples, not features).

    ``indices`` are positions into the original ``LoadedDataset`` rows, kept so a same-seed re-run
    can be checked for identical partitioning. ``X``/``y`` carry all features for the partition's
    samples; feature-subset restriction is the probe's job.
    """

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
    """Draw one ``random_state`` integer from the single shared RNG (CON-003).

    The split's only randomness originates here, so the partitioning is fully determined by the
    shared seed — no component seeds independently.
    """
    return int(rng.numpy.integers(0, 2**32))


def _group_split(
    idx: np.ndarray, groups: np.ndarray, test_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Two-way split of ``idx`` that keeps whole groups together — no group spans the two sides.

    ``groups`` is aligned to ``idx`` (one group label per entry). Returns ``(keep_idx, held_idx)``
    as original-index arrays, the held side being ~``test_fraction`` of the *groups* (group-level,
    so the row fractions are approximate). Consumes the single integer ``seed`` drawn from the
    shared RNG, so partitioning is reproducible (CON-003).
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    keep_pos, held_pos = next(splitter.split(idx, groups=groups))
    return idx[keep_pos], idx[held_pos]


def make_split(dataset: LoadedDataset, config: IrfsConfig, rng: SeededRng) -> Split:
    """Partition a dataset into train, validation, and test rows.

    Predefined test rows are kept as supplied. Otherwise they are sampled from the full dataset.
    Validation rows are sampled from the remaining training pool.
    """
    indices = np.arange(dataset.X.shape[0])

    if dataset.test_indices is not None:
        test_idx = np.asarray(dataset.test_indices, dtype=int)
        train_pool_idx = np.setdiff1d(indices, test_idx, assume_unique=True)
    elif dataset.groups is None:
        train_pool_idx, test_idx = train_test_split(
            indices,
            test_size=config.test_fraction,
            random_state=_draw_split_seed(rng),
            stratify=dataset.y,
        )
    else:
        train_pool_idx, test_idx = _group_split(
            indices, dataset.groups, config.test_fraction, _draw_split_seed(rng)
        )

    if dataset.groups is None:
        train_idx, val_idx = train_test_split(
            train_pool_idx,
            test_size=config.validation_fraction,
            random_state=_draw_split_seed(rng),
            stratify=dataset.y[train_pool_idx],
        )
    else:
        train_idx, val_idx = _group_split(
            train_pool_idx,
            dataset.groups[train_pool_idx],
            config.validation_fraction,
            _draw_split_seed(rng),
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
