"""Small, batch-oriented feedback interfaces for the stable engine."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Sequence, runtime_checkable

import numpy as np
import torch

if TYPE_CHECKING:
    from harness.contract import SelectionContext


@runtime_checkable
class BatchStateEncoder(Protocol):
    dimension: int
    trainable: bool

    def encode_batch(
        self,
        subsets: Sequence[tuple[int, ...]],
        context: "SelectionContext",
    ) -> torch.Tensor: ...


@runtime_checkable
class RewardVector(Protocol):
    def evaluate(self, subset: tuple[int, ...], context: "SelectionContext") -> np.ndarray: ...

    def state_dict(self) -> dict: ...

    def load_state_dict(self, state: dict) -> None: ...
