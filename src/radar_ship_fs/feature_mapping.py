"""Explicit conversions between the three feature-coordinate systems.

The radar SVM-light files expose 75 raw columns.  Their scientific feature IDs
are 1-based, while NumPy and cleaned matrices are indexed from zero.  Keeping
these spaces explicit prevents a valid original ID from being silently used as
a NumPy column index.
"""

from __future__ import annotations

from dataclasses import dataclass
from operator import index
from typing import Mapping, Sequence


def _integers(values: Sequence[int], label: str) -> tuple[int, ...]:
    try:
        result = tuple(int(index(value)) for value in values)
    except TypeError as exc:
        raise ValueError(f"{label} must contain integers") from exc
    if len(result) != len(set(result)):
        raise ValueError(f"{label} must not contain duplicate values")
    return result


def original_ids_1based_to_raw_indices_0based(
    original_feature_ids: Sequence[int],
    *,
    raw_feature_count: int = 75,
) -> tuple[int, ...]:
    """Convert scientific 1-based IDs to raw NumPy/SVM-light column indices."""
    ids = _integers(original_feature_ids, "original_feature_ids_1based")
    invalid = [value for value in ids if not 1 <= value <= raw_feature_count]
    if invalid:
        raise ValueError(
            "original_feature_ids_1based contains IDs outside "
            f"[1, {raw_feature_count}]: {invalid}"
        )
    return tuple(value - 1 for value in ids)


def raw_indices_0based_to_original_ids_1based(
    raw_indices: Sequence[int],
    *,
    raw_feature_count: int = 75,
) -> tuple[int, ...]:
    """Convert raw NumPy/SVM-light columns to scientific 1-based IDs."""
    indices = _integers(raw_indices, "raw_indices_0based")
    invalid = [value for value in indices if not 0 <= value < raw_feature_count]
    if invalid:
        raise ValueError(
            f"raw_indices_0based contains indices outside [0, {raw_feature_count}): {invalid}"
        )
    return tuple(value + 1 for value in indices)


@dataclass(frozen=True)
class FeatureIndexMap:
    """Mapping defined by the ordered original IDs retained after cleaning."""

    final_original_ids_1based: tuple[int, ...]
    raw_feature_count: int = 75

    def __post_init__(self) -> None:
        ids = _integers(self.final_original_ids_1based, "final_original_ids_1based")
        original_ids_1based_to_raw_indices_0based(
            ids,
            raw_feature_count=self.raw_feature_count,
        )
        object.__setattr__(self, "final_original_ids_1based", ids)

    @classmethod
    def from_metadata(cls, metadata: Mapping) -> "FeatureIndexMap":
        if "final_feature_ids" not in metadata:
            raise ValueError("dataset metadata is missing final_feature_ids")
        return cls(
            tuple(int(value) for value in metadata["final_feature_ids"]),
            int(metadata.get("original_feature_count", 75)),
        )

    @property
    def clean_feature_count(self) -> int:
        return len(self.final_original_ids_1based)

    def clean_indices_to_original_ids_1based(
        self, clean_indices: Sequence[int]
    ) -> tuple[int, ...]:
        indices = _integers(clean_indices, "clean_indices_0based")
        invalid = [value for value in indices if not 0 <= value < self.clean_feature_count]
        if invalid:
            raise ValueError(
                "clean_indices_0based contains indices outside "
                f"[0, {self.clean_feature_count}): {invalid}"
            )
        return tuple(self.final_original_ids_1based[index] for index in indices)

    def original_ids_1based_to_clean_indices(
        self, original_feature_ids: Sequence[int]
    ) -> tuple[int, ...]:
        ids = _integers(original_feature_ids, "original_feature_ids_1based")
        lookup = {
            original_id: clean_index
            for clean_index, original_id in enumerate(self.final_original_ids_1based)
        }
        missing = [value for value in ids if value not in lookup]
        if missing:
            raise ValueError(f"original feature IDs were removed during cleaning: {missing}")
        return tuple(lookup[value] for value in ids)

    def clean_indices_to_raw_indices_0based(
        self, clean_indices: Sequence[int]
    ) -> tuple[int, ...]:
        return original_ids_1based_to_raw_indices_0based(
            self.clean_indices_to_original_ids_1based(clean_indices),
            raw_feature_count=self.raw_feature_count,
        )

    def validate_artifact_selection(self, payload: Mapping) -> tuple[int, ...]:
        """Return clean indices only after both artifact coordinate fields agree."""
        if "selected_clean_indices" not in payload:
            raise ValueError("selection artifact is missing selected_clean_indices")
        clean = _integers(payload["selected_clean_indices"], "selected_clean_indices")
        expected_original = self.clean_indices_to_original_ids_1based(clean)
        if "selected_original_feature_ids" not in payload:
            raise ValueError("selection artifact is missing selected_original_feature_ids")
        recorded_original = _integers(
            payload["selected_original_feature_ids"],
            "selected_original_feature_ids",
        )
        if recorded_original != expected_original:
            raise ValueError(
                "selection artifact feature mapping mismatch: "
                f"clean indices map to {list(expected_original)}, "
                f"recorded original IDs are {list(recorded_original)}"
            )
        return clean
