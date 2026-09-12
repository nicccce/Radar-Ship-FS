"""Regression tests for raw/original/cleaned feature-coordinate conversions."""

from __future__ import annotations

import pytest

from radar_ship_fs.feature_mapping import (
    FeatureIndexMap,
    original_ids_1based_to_raw_indices_0based,
    raw_indices_0based_to_original_ids_1based,
)


def test_nontrivial_clean_mapping_does_not_confuse_original_ids_with_columns() -> None:
    mapping = FeatureIndexMap((1, 3, 7, 10), raw_feature_count=10)

    assert mapping.clean_indices_to_original_ids_1based((0, 2, 3)) == (1, 7, 10)
    assert mapping.clean_indices_to_raw_indices_0based((0, 2, 3)) == (0, 6, 9)
    assert mapping.original_ids_1based_to_clean_indices((10, 1, 7)) == (3, 0, 2)
    assert original_ids_1based_to_raw_indices_0based((1, 7, 10), raw_feature_count=10) == (
        0,
        6,
        9,
    )
    assert raw_indices_0based_to_original_ids_1based((0, 6, 9), raw_feature_count=10) == (
        1,
        7,
        10,
    )


def test_artifact_mapping_mismatch_fails_loudly() -> None:
    mapping = FeatureIndexMap((1, 3, 7, 10), raw_feature_count=10)
    valid = {
        "selected_clean_indices": [1, 3],
        "selected_original_feature_ids": [3, 10],
    }
    assert mapping.validate_artifact_selection(valid) == (1, 3)

    wrong_column = {**valid, "selected_original_feature_ids": [4, 10]}
    with pytest.raises(ValueError, match="mapping mismatch"):
        mapping.validate_artifact_selection(wrong_column)


@pytest.mark.parametrize(
    "values, message",
    [((0,), "outside"), ((11,), "outside"), ((1, 1), "duplicate")],
)
def test_original_id_conversion_rejects_ambiguous_or_invalid_ids(values, message) -> None:
    with pytest.raises(ValueError, match=message):
        original_ids_1based_to_raw_indices_0based(values, raw_feature_count=10)


def test_feature_coordinates_reject_fractional_values_instead_of_truncating() -> None:
    with pytest.raises(ValueError, match="integers"):
        original_ids_1based_to_raw_indices_0based((1.5,), raw_feature_count=10)
