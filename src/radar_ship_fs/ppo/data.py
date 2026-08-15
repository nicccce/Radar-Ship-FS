from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.datasets import load_svmlight_file

DOMAIN_NAMES = ("time", "frequency", "time_frequency", "nonlinear", "fractional", "polarimetric")
DOMAIN_RANGES = ((1, 20), (21, 31), (32, 38), (39, 46), (47, 48), (49, 75))


@dataclass(frozen=True)
class RadarData:
    X_development: np.ndarray
    y_development: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    original_feature_ids: np.ndarray
    feature_names: tuple[str, ...]
    domains: np.ndarray
    metadata: dict

    @property
    def n_features(self) -> int:
        return int(self.X_development.shape[1])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_columns(X: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    kept: list[int] = []
    duplicates: dict[int, int] = {}
    for current in range(X.shape[1]):
        duplicate_of = next((prior for prior in kept if np.array_equal(X[:, current], X[:, prior])), None)
        if duplicate_of is None:
            kept.append(current)
        else:
            duplicates[current] = duplicate_of
    return np.asarray(kept, dtype=np.int64), duplicates


def _domain_id(feature_id: int) -> int:
    for domain, (lower, upper) in enumerate(DOMAIN_RANGES):
        if lower <= feature_id <= upper:
            return domain
    raise ValueError(f"feature id {feature_id} has no physical domain")


def _catalog(data_dir: Path) -> dict[int, str]:
    path = data_dir / "v16n_analysis" / "feature_catalog.csv"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            int(row["experiment_id_1based"]): row["feature_name"]
            for row in csv.DictReader(handle)
        }


def load_radar_data(data_dir: Path, version: str = "v16n") -> RadarData:
    """Match Radar-Ship-FS: clean on source train, then keep source test sealed."""
    train_path = data_dir / f"sim_ship_cr_{version}.train.svm"
    test_path = data_dir / f"sim_ship_cr_{version}.test.svm"
    for path in (train_path, test_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing radar dataset file: {path}")

    n_original = 75
    X_train_sparse, y_train = load_svmlight_file(train_path, n_features=n_original)
    X_test_sparse, y_test = load_svmlight_file(test_path, n_features=n_original)
    X_train = X_train_sparse.toarray().astype(np.float32)
    X_test = X_test_sparse.toarray().astype(np.float32)
    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)

    constant = np.isclose(X_train.max(axis=0), X_train.min(axis=0))
    nonconstant_ids = np.arange(1, n_original + 1, dtype=np.int64)[~constant]
    X_train_nc = X_train[:, ~constant]
    X_test_nc = X_test[:, ~constant]
    unique_positions, duplicate_positions = _unique_columns(X_train_nc)

    original_ids = nonconstant_ids[unique_positions]
    names = _catalog(data_dir)
    duplicate_ids = {
        int(nonconstant_ids[removed]): int(nonconstant_ids[kept])
        for removed, kept in duplicate_positions.items()
    }
    metadata = {
        "source_version": version,
        "source_files": {
            "train": {"name": train_path.name, "sha256": _sha256(train_path)},
            "test": {"name": test_path.name, "sha256": _sha256(test_path)},
        },
        "source_train_rows": int(X_train.shape[0]),
        "source_test_rows": int(X_test.shape[0]),
        "original_feature_count": n_original,
        "constant_feature_ids": np.arange(1, n_original + 1)[constant].astype(int).tolist(),
        "duplicate_feature_mapping": {str(key): value for key, value in duplicate_ids.items()},
        "final_feature_ids": original_ids.astype(int).tolist(),
        "final_feature_count": int(original_ids.size),
        "preprocessing_fit_scope": "source_train_only",
    }
    return RadarData(
        X_development=X_train_nc[:, unique_positions],
        y_development=y_train,
        X_test=X_test_nc[:, unique_positions],
        y_test=y_test,
        original_feature_ids=original_ids,
        feature_names=tuple(names.get(int(fid), f"feature_{fid}") for fid in original_ids),
        domains=np.asarray([_domain_id(int(fid)) for fid in original_ids], dtype=np.int64),
        metadata=metadata,
    )
