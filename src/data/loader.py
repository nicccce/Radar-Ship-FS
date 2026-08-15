"""Dataset loader for radar ship data.

Loads the sim_ship_cr radar dataset.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Callable, NamedTuple, Optional

import numpy as np
from sklearn.datasets import load_svmlight_file

from config import IrfsConfig


class LoadedDataset(NamedTuple):
    """Result of loading a dataset."""
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    n_features: int
    n_classes: int
    groups: Optional[np.ndarray] = None
    test_indices: Optional[np.ndarray] = None
    metadata: Optional[dict] = None


def _sha256(path: Path) -> str:
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


def load_radar_ship(data_dir: str, version: str) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    """Load and clean the supplied radar-ship SVM-light train/test files."""
    if not version or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
        for character in version
    ):
        raise ValueError(f"invalid radar_ship_version: {version!r}")
    root = Path(data_dir)
    train_path = root / f"sim_ship_cr_{version}.train.svm"
    test_path = root / f"sim_ship_cr_{version}.test.svm"
    missing = [str(path) for path in (train_path, test_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Radar-ship SVM-light file(s) not found: {missing}."
        )

    n_original_features = 75
    X_train_sparse, y_train = load_svmlight_file(train_path, n_features=n_original_features)
    X_test_sparse, y_test = load_svmlight_file(test_path, n_features=n_original_features)
    
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


def load(config: IrfsConfig) -> LoadedDataset:
    """Load the radar ship dataset."""
    X, y, feature_names, metadata = load_radar_ship(
        config.data_dir,
        config.radar_ship_version,
    )
    test_start = int(metadata["source_train_rows"])
    test_indices = np.arange(test_start, X.shape[0], dtype=int)
    metadata = {
        **metadata,
        "row_split_protocol": "source_train_for_development_source_test_for_evaluation",
        "source_file_row_boundary_used": True,
        "candidate_feature_pool_note": (
            "the feature mask is fitted on the source train file and applied unchanged to test"
        ),
    }
    n_features = X.shape[1]
    n_classes = int(np.unique(y).size)
    return LoadedDataset(
        X=X,
        y=y,
        feature_names=feature_names,
        n_features=n_features,
        n_classes=n_classes,
        groups=None,
        test_indices=test_indices,
        metadata=metadata,
    )
