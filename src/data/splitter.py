"""Leakage-safe splitter (COMP-002).

Produces disjoint train / validation / test partitions where the test partition is structurally —
not merely conventionally — unreachable by feature-search or reward code paths. The :class:`Split`
object exposes ``.train`` and ``.validation`` as plain attributes but has **no ``.test``
attribute**: the test partition is held privately and released only through
:meth:`Split.release_test_for_final_metrics`, the single, loudly named opt-in. The accidental-
leakage expression ``probe(split.test)`` therefore cannot be written (it raises ``AttributeError``),
and the one legitimate call site is greppable for the AC-002 inspection.

Per DEC-005 / REQ-002, the test partition feeds only the final reported metrics; per-step reward and
exploration draw on the validation partition carved from training data.

Satisfies COMP-002 -> REQ-002. Structural prevention mechanism for ROLLBACK-001.
"""

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


class Split:
    """Train / validation / test partitions with a structurally gated test slice.

    ``train`` and ``validation`` are freely accessible. The test partition has no public attribute;
    obtain it only via :meth:`release_test_for_final_metrics`.
    """

    def __init__(self, train: Partition, validation: Partition, test: Partition) -> None:
        self.train = train
        self.validation = validation
        self._test = test  # no public ``.test`` — see release_test_for_final_metrics

    def release_test_for_final_metrics(self) -> Partition:
        """Return the held-out test partition for final reported metrics only.

        This is the sole legitimate path to the test partition (REQ-002 / DEC-005). Calling it from
        a feature-search, policy-learning, or reward path is a leakage violation (ROLLBACK-001); the
        explicit name makes such a call auditable.
        """
        return self._test

    def replace_development_for_inner_cv(self, development: Partition) -> "Split":
        """Return an inner-CV view while keeping the existing test partition sealed.

        Both public partition handles intentionally point at the complete development partition:
        legacy selection components read ``train`` for state/relevance data and pass ``validation``
        to the probe, while the cross-validated probe performs the actual disjoint fold split.
        The held-out test object is transferred privately without calling the final-metrics release
        method, so feature search still has no test access path.
        """
        return Split(train=development, validation=development, test=self._test)


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
    """Partition ``dataset`` into disjoint train / validation / test slices.

    Two stages: first carve the test partition from the whole dataset by ``config.test_fraction``,
    then carve validation from the remaining training pool by ``config.validation_fraction`` (a
    fraction of the pool). For a dataset that defines a grouping variable (``dataset.groups``), both
    carves are group-aware — whole groups stay on one side (REQ-022) and are unstratified; otherwise
    they are stratified-random as before. Each stage consumes one integer drawn from ``rng`` (CON-003),
    so the same seed reproduces identical indices.
    """
    indices = np.arange(dataset.X.shape[0])

    if dataset.predefined_test_indices is not None:
        # File-based datasets may ship an official test split. Keep it structurally sealed and
        # carve only the source training rows into selector-train and reward-validation partitions.
        test_idx = np.asarray(dataset.predefined_test_indices, dtype=int)
        if test_idx.ndim != 1 or test_idx.size == 0:
            raise ValueError("predefined_test_indices must be a non-empty one-dimensional array")
        if np.unique(test_idx).size != test_idx.size:
            raise ValueError("predefined_test_indices must not contain duplicates")
        if test_idx.min() < 0 or test_idx.max() >= indices.size:
            raise ValueError("predefined_test_indices contains an out-of-range row")
        train_pool_idx = np.setdiff1d(indices, test_idx, assume_unique=True)
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
    elif dataset.groups is None:
        # Ungrouped datasets without an official test file retain the two-stage stratified split.
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
    else:
        # Group-aware (REQ-022): keep whole groups (e.g. subjects) on one side so no group spans
        # partitions. Unstratified (a pure group split cannot stratify by label).
        train_pool_idx, test_idx = _group_split(
            indices, dataset.groups, config.test_fraction, _draw_split_seed(rng)
        )
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
