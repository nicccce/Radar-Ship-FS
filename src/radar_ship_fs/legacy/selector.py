"""Adapter exposing the frozen reinforced implementation as a public Selector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from methods.reinforced_run import REINFORCED_METHOD_NAMES, build_reinforced_engine

if TYPE_CHECKING:
    from config import IrfsConfig
    from harness.contract import SelectionContext, SubsetSelection


@dataclass(frozen=True)
class LegacySelector:
    """One historical reinforced method behind the common Selector boundary."""

    method: str
    config: "IrfsConfig"

    def __post_init__(self) -> None:
        if self.method not in REINFORCED_METHOD_NAMES:
            raise ValueError(
                f"unknown legacy method {self.method!r}; expected one of {REINFORCED_METHOD_NAMES}"
            )

    def select(self, context: "SelectionContext") -> "SubsetSelection":
        return build_reinforced_engine(self.method, self.config).select(context)


def build_legacy_selector(method: str, config: "IrfsConfig") -> LegacySelector:
    """Construct a fresh adapter without adding legacy branches to the stable trainer."""
    return LegacySelector(method=method, config=config)
