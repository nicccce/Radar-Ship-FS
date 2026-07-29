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

import hashlib
import os
from pathlib import Path
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
    predefined_test_indices: Optional[np.ndarray] = None
    metadata: Optional[dict] = None


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


def _sha256(path: Path) -> str:
    """Return a stable content fingerprint without exposing machine-specific absolute paths."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_unique_columns(X: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    """Find exact duplicate columns using training data only."""
    keep_indices: list[int] = []
    duplicate_mapping: dict[int, int] = {}
    for current_idx in range(X.shape[1]):
        duplicate_of = next(
            (kept_idx for kept_idx in keep_indices if np.array_equal(X[:, current_idx], X[:, kept_idx])),
            None,
        )
        if duplicate_of is None:
            keep_indices.append(current_idx)
        else:
            duplicate_mapping[current_idx] = duplicate_of
    return np.asarray(keep_indices, dtype=int), duplicate_mapping


def _label_counts(y: np.ndarray) -> dict[str, int]:
    labels, counts = np.unique(y, return_counts=True)
    return {str(int(label)): int(count) for label, count in zip(labels, counts)}


def _load_radar_ship(
    data_dir: str,
    version: str,
) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    """Load and leakage-safely clean the supplied radar-ship SVM-light train/test files.

    Constant and exact-duplicate columns are identified on the first supplied file only, preserving
    the version-specific candidate pool. The same column mask is applied to the second file, then
    all rows are concatenated. Row splitting happens later over the combined rows through the
    ordinary stratified-random splitter. Only the source filenames vary by version; preprocessing
    is shared unchanged.
    """
    from sklearn.datasets import load_svmlight_file

    if not version or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for character in version
    ):
        raise ValueError(f"invalid radar_ship_version: {version!r}")
    root = Path(data_dir)
    train_path = root / f"sim_ship_cr_{version}.train.svm"
    test_path = root / f"sim_ship_cr_{version}.test.svm"
    missing = [str(path) for path in (train_path, test_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Radar-ship SVM-light file(s) not found: "
            f"{missing}. Set config.data_dir to the directory containing both files."
        )

    n_original_features = 75
    X_train_sparse, y_train = load_svmlight_file(
        train_path,
        n_features=n_original_features,
    )
    X_test_sparse, y_test = load_svmlight_file(
        test_path,
        n_features=n_original_features,
    )
    X_train = X_train_sparse.toarray().astype(np.float32)
    X_test = X_test_sparse.toarray().astype(np.float32)
    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)

    feature_min = X_train.min(axis=0)
    feature_max = X_train.max(axis=0)
    constant_mask = np.isclose(feature_max, feature_min)
    nonconstant_mask = ~constant_mask
    original_ids = np.arange(1, n_original_features + 1, dtype=int)
    nonconstant_ids = original_ids[nonconstant_mask]

    X_train_nonconstant = X_train[:, nonconstant_mask]
    X_test_nonconstant = X_test[:, nonconstant_mask]
    unique_indices, duplicate_positions = _find_unique_columns(X_train_nonconstant)
    X_train_final = X_train_nonconstant[:, unique_indices]
    X_test_final = X_test_nonconstant[:, unique_indices]
    final_ids = nonconstant_ids[unique_indices]
    duplicate_original_ids = {
        int(nonconstant_ids[removed]): int(nonconstant_ids[kept])
        for removed, kept in duplicate_positions.items()
    }

    X = np.vstack((X_train_final, X_test_final)).astype(np.float32, copy=False)
    y = np.concatenate((y_train, y_test)).astype(np.int64, copy=False)
    feature_names = [f"feature_{feature_id}" for feature_id in final_ids]
    metadata = {
        "source_format": "svmlight",
        "source_version": version,
        "source_files": {
            "train": {"name": train_path.name, "sha256": _sha256(train_path)},
            "test": {"name": test_path.name, "sha256": _sha256(test_path)},
        },
        "original_feature_count": n_original_features,
        "constant_feature_ids": original_ids[constant_mask].tolist(),
        "duplicate_feature_mapping": {str(removed): kept for removed, kept in duplicate_original_ids.items()},
        "final_feature_ids": final_ids.tolist(),
        "final_feature_count": int(final_ids.size),
        "source_train_rows": int(X_train_final.shape[0]),
        "source_test_rows": int(X_test_final.shape[0]),
        "source_train_label_counts": _label_counts(y_train),
        "source_test_label_counts": _label_counts(y_test),
        "preprocessing_fit_scope": "source_train_only",
    }
    return X, y, feature_names, metadata


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
    available = (*_LOADERS, "radar_ship")
    if name not in available:
        raise ValueError(f"Unknown dataset {name!r}; available: {sorted(available)}")
    predefined_test_indices = None
    metadata = None
    if name == "radar_ship":
        X, y, feature_names, metadata = _load_radar_ship(
            config.data_dir,
            config.radar_ship_version,
        )
        groups = None
        metadata = {
            **metadata,
            "row_split_protocol": "combine_source_files_then_stratified_random_split",
            "source_file_row_boundary_used": False,
            "candidate_feature_pool_note": (
                "the source-train-fitted feature mask is shared by every method; both files are "
                "combined before train/validation/test row splitting"
            ),
        }
    else:
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
        predefined_test_indices=predefined_test_indices,
        metadata=metadata,
    )
