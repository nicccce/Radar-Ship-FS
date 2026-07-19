"""Dataset loader (COMP-001).

Loads a config-named tabular classification dataset and reports its detected feature and class
counts, making no assumption about the number of features. Binary labels are the validation target;
the loader derives ``n_classes`` from the data and does not assume binary, so a multiclass dataset
would load without code changes. The dataset is selected by ``config.dataset`` (COMP-025), so
switching datasets is a configuration change, not a code change.

WDBC (``sklearn.datasets.load_breast_cancer``) is the sklearn-bundled validation target; Parkinson's
is the file-based, subject-grouped one.

Satisfies COMP-001 -> REQ-001.
"""

from __future__ import annotations

import os
from typing import Callable, NamedTuple, Optional

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

from config import IrfsConfig


class LoadedDataset(NamedTuple):
    """Result of loading a dataset.

    Field order ``(X, y, feature_names, n_features, n_classes, groups)``. Callers use attribute
    access (not positional unpacking), so the trailing optional ``groups`` is additive. ``groups``
    is a per-row grouping vector (e.g. subject ids) for datasets that define one, enabling group-
    aware splitting (REQ-022); it is ``None`` for datasets without grouping (the sklearn-bunch
    datasets).
    """

    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    n_features: int
    n_classes: int
    groups: Optional[np.ndarray] = None


# A loader returns (X, y, feature_names, groups); ``groups`` is the per-row grouping vector
# (e.g. subject ids) or ``None`` when the dataset defines no grouping.
_LoaderResult = tuple[np.ndarray, np.ndarray, list[str], Optional[np.ndarray]]


def _load_sklearn(loader: Callable[..., object]) -> _LoaderResult:
    """Adapt an sklearn ``load_*`` Bunch into (X, y, feature_names, groups=None)."""
    bunch = loader()
    X = np.asarray(bunch.data)
    y = np.asarray(bunch.target)
    feature_names = [str(name) for name in bunch.feature_names]
    return X, y, feature_names, None  # sklearn bunches define no grouping


def _load_parkinsons(data_dir: str) -> _LoaderResult:
    """Load UCI Parkinson's Disease Classification (Sakar 2018, UCI id=470) from
    ``<data_dir>/parkinsons/pd_speech_features.csv`` — subjects are the grouping (REQ-022).

    The 756 rows are voice recordings from 252 subjects with three sustained-/a/ phonations each
    (252x3=756), so the per-row subject ``id`` is exposed as ``groups`` to keep each subject wholly
    in one partition. Without this a subject's other two recordings leak across the split and
    inflate measured accuracy. The CSV's first physical line is a feature-group banner ("Baseline
    Features", ...), so the real header is row index 1 (``header=1``). ``id`` is the grouping key
    and is dropped from X (it is not a feature); ``gender`` is kept as a feature; ``class`` (1=PD,
    0=healthy) is the target.

    ``fetch_ucirepo(id=470)`` is deliberately NOT used: this dataset is not available via the UCI
    import API (it raises DatasetNotFoundError), so it is loaded from the statically downloaded CSV
    under ``config.data_dir``. Raises ``FileNotFoundError`` with the expected path when the data is
    not present locally.
    """
    path = os.path.join(data_dir, "parkinsons", "pd_speech_features.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Parkinson's dataset not found at {path!r}. fetch_ucirepo(id=470) is unavailable "
            f"via the UCI import API; download 'Parkinson's Disease Classification' (id=470) from "
            f"the UCI repository, extract pd_speech_features.csv (it is a .rar nested inside the "
            f".zip), and place it at {path!r}, or point config.data_dir at its grandparent."
        )
    df = pd.read_csv(path, header=1)
    missing = {"id", "class"} - set(df.columns)
    if missing:
        raise ValueError(
            f"Parkinson's CSV at {path!r} is missing expected column(s) {sorted(missing)}; "
            f"check that it is pd_speech_features.csv read with its banner first line skipped."
        )
    groups = df["id"].to_numpy().astype(int)
    y = df["class"].to_numpy().astype(int)
    feature_cols = [c for c in df.columns if c not in ("id", "class")]
    X = df[feature_cols].to_numpy(dtype=float)
    feature_names = [str(c) for c in feature_cols]
    return X, y, feature_names, groups


# Registry of config-selectable datasets. Adding a compliant dataset is a registry entry; the
# loading logic below is dataset-agnostic (counts are derived). Each loader takes the effective
# config (so file-based datasets can read config.data_dir) and returns a ``_LoaderResult``.
_LOADERS: dict[str, Callable[[IrfsConfig], _LoaderResult]] = {
    "wdbc": lambda config: _load_sklearn(load_breast_cancer),
    "parkinsons": lambda config: _load_parkinsons(config.data_dir),
}


def load(config: IrfsConfig) -> LoadedDataset:
    """Load the dataset named by ``config.dataset``.

    Returns a :class:`LoadedDataset` with the feature and class counts derived from the data — no
    assumption about the number of features — and ``groups`` set when the dataset defines a grouping
    variable (``None`` otherwise). Raises ``ValueError`` for an unknown dataset name.
    """
    name = config.dataset
    if name not in _LOADERS:
        raise ValueError(f"Unknown dataset {name!r}; available: {sorted(_LOADERS)}")
    X, y, feature_names, groups = _LOADERS[name](config)
    n_features = X.shape[1]  # derived from the data, not assumed
    n_classes = int(np.unique(y).size)
    return LoadedDataset(
        X=X,
        y=y,
        feature_names=feature_names,
        n_features=n_features,
        n_classes=n_classes,
        groups=groups,
    )
