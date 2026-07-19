"""Data domain (D1): dataset loading and leakage-safe splitting.

Covers ``data/loader.py`` (config-selected datasets, feature/class counts derived from the data,
per-row grouping for datasets that define it) and ``data/splitter.py`` (disjoint
train/validation/test partitions with a structurally *sealed* test slice, group-aware so no group
spans partitions — REQ-022).

WDBC comes from ``sklearn.load_breast_cancer`` (library-bundled, not a repo artifact); the grouped-
dataset path is exercised by synthesising a ``pd_speech_features.csv`` in a tmp dir — no test reads
a pre-existing artifact. Split *determinism* (same seed → identical indices) is proven once in
``test_invariants.py`` (D8) and deliberately not re-asserted here.
"""

from __future__ import annotations

import numpy as np
import pytest

from config import load_config
from data.loader import load
from data.splitter import Partition, make_split
from rng import init_rng

# A small grouped dataset: several subjects × three phonations each, mirroring the real
# Parkinson's 252×3 structure. Columns: id (grouping key, not a feature), gender + two signal
# features, class target.
_N_SUBJECTS = 6
_RECORDINGS_PER_SUBJECT = 3


def _write_synthetic_parkinsons_csv(path) -> None:
    """Write a CSV matching pd_speech_features.csv's layout: banner line, then header + rows."""
    lines = [
        # First physical line is the feature-group banner UCI ships; the loader skips it via header=1.
        "banner,banner,Baseline Features,Baseline Features,banner",
        "id,gender,feat_a,feat_b,class",
    ]
    for subject in range(_N_SUBJECTS):
        label = subject % 2  # alternate PD/healthy so both classes are present
        for rec in range(_RECORDINGS_PER_SUBJECT):
            lines.append(f"{subject},{label},{subject + rec * 0.1},{rec * 1.0},{label}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture()
def parkinsons_data_dir(tmp_path):
    """A tmp ``data_dir`` holding a synthetic ``parkinsons/pd_speech_features.csv`` (grouped)."""
    subdir = tmp_path / "parkinsons"
    subdir.mkdir()
    _write_synthetic_parkinsons_csv(subdir / "pd_speech_features.csv")
    return str(tmp_path)


def _wdbc_split():
    """Load WDBC and split it from a freshly seeded RNG — the ungrouped (stratified) path."""
    config = load_config()  # defaults: dataset="wdbc"
    data = load(config)
    return config, data, make_split(data, config, init_rng(config.seeds[0]))


# === Loader =======================================================================================


def test_wdbc_loads_with_derived_shape() -> None:
    """WDBC reports 30 features and 2 classes, both derived from the data (not hardcoded), and
    defines no grouping (sklearn bunches are ungrouped)."""
    data = load(load_config())

    assert data.n_features == 30
    assert data.n_classes == 2
    assert data.n_features == data.X.shape[1]  # derived from the data width
    assert data.n_classes == int(np.unique(data.y).size)
    assert data.groups is None


def test_grouped_dataset_drops_id_and_exposes_subject_groups(parkinsons_data_dir: str) -> None:
    """The grouped loader keeps gender as a feature, drops the ``id`` grouping key from X, derives
    the class count, and exposes the per-row subject id as ``groups`` (REQ-022)."""

    data = load(load_config({"dataset": "parkinsons", "data_dir": parkinsons_data_dir}))

    n_rows = _N_SUBJECTS * _RECORDINGS_PER_SUBJECT
    assert data.X.shape == (n_rows, 3)  # gender + feat_a + feat_b; id and class excluded
    assert data.feature_names == ["gender", "feat_a", "feat_b"]
    assert "id" not in data.feature_names and "class" not in data.feature_names
    assert data.n_features == data.X.shape[1] and data.n_classes == 2
    assert data.X.dtype == float

    assert data.groups is not None and data.groups.shape == (n_rows,)
    _, counts = np.unique(data.groups, return_counts=True)
    assert set(counts.tolist()) == {_RECORDINGS_PER_SUBJECT}  # every subject's rows kept together


def test_missing_dataset_file_raises_with_guidance(tmp_path) -> None:
    """An absent file-based dataset fails loudly, naming the expected file — not a cryptic error."""
    config = load_config({"dataset": "parkinsons", "data_dir": str(tmp_path)})
    with pytest.raises(FileNotFoundError, match="pd_speech_features.csv"):
        load(config)


def test_unknown_dataset_name_is_rejected() -> None:
    """An unregistered dataset name is rejected with the list of available datasets."""
    with pytest.raises(ValueError, match="Unknown dataset"):
        load(load_config({"dataset": "does_not_exist"}))


# === Splitter =====================================================================================


def test_partitions_are_disjoint_and_cover_the_dataset() -> None:
    """Train / validation / test are pairwise disjoint, together cover every row exactly once, and
    the test partition is approximately the configured fraction (ungrouped stratified path)."""
    config, data, split = _wdbc_split()

    train = set(split.train.indices.tolist())
    val = set(split.validation.indices.tolist())
    test = set(split.release_test_for_final_metrics().indices.tolist())

    assert train.isdisjoint(val) and train.isdisjoint(test) and val.isdisjoint(test)
    total = data.X.shape[0]
    assert len(train) + len(val) + len(test) == total
    assert len(test) == pytest.approx(total * config.test_fraction, abs=2)


def test_test_partition_is_structurally_sealed() -> None:
    """The test partition has no public attribute — the accidental-leakage expression ``split.test``
    raises — and is reachable only through the loud ``release_test_for_final_metrics`` opt-in
    (REQ-002)."""
    _config, _data, split = _wdbc_split()

    assert not hasattr(split, "test")
    with pytest.raises(AttributeError):
        _ = split.test  # type: ignore[attr-defined]

    held_out = split.release_test_for_final_metrics()
    assert isinstance(held_out, Partition) and held_out.X.shape[0] > 0


def test_split_keeps_each_group_whole(parkinsons_data_dir: str) -> None:
    """Group-aware split (REQ-022): no subject's recordings span two partitions — the subject sets
    of train / validation / test are pairwise disjoint."""
    config = load_config({"dataset": "parkinsons", "data_dir": parkinsons_data_dir})
    data = load(config)
    split = make_split(data, config, init_rng(config.seeds[0]))

    def subjects(indices: np.ndarray) -> set[int]:
        return set(data.groups[indices].tolist())

    train = subjects(split.train.indices)
    val = subjects(split.validation.indices)
    test = subjects(split.release_test_for_final_metrics().indices)

    assert train.isdisjoint(val) and train.isdisjoint(test) and val.isdisjoint(test)
