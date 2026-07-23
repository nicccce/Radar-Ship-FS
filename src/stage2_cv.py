"""Build the stage-2 outer-test/inner-CV selection context."""

from __future__ import annotations

from typing import Optional

import numpy as np

from config import IrfsConfig
from cv_probe import CrossValidatedDecisionTreeProbe
from data.loader import load
from data.splitter import Partition, Split, make_split
from harness.contract import SelectionContext
from rng import init_rng


def combine_train_validation(split: Split) -> Partition:
    """Combine the ordinary split's two non-test partitions into development data."""
    train = split.train
    validation = split.validation
    groups: Optional[np.ndarray] = None
    if train.groups is not None and validation.groups is not None:
        groups = np.concatenate((train.groups, validation.groups))
    return Partition(
        X=np.vstack((train.X, validation.X)),
        y=np.concatenate((train.y, validation.y)),
        indices=np.concatenate((train.indices, validation.indices)),
        feature_names=train.feature_names,
        groups=groups,
        metadata=train.metadata,
    )


def build_stage2_cv_context(
    config: IrfsConfig,
    *,
    seed: int,
    n_splits: int,
) -> SelectionContext:
    """Build one context with a sealed outer test and a stratified inner-CV DT probe."""
    rng = init_rng(seed)
    dataset = load(config)
    ordinary_split = make_split(dataset, config, rng)
    development = combine_train_validation(ordinary_split)
    inner_cv_split = ordinary_split.replace_development_for_inner_cv(development)
    probe = CrossValidatedDecisionTreeProbe(
        development,
        config,
        rng,
        n_splits=n_splits,
    )
    return SelectionContext(split=inner_cv_split, probe=probe, config=config, rng=rng)
