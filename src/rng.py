"""Seed & determinism control (COMP-026).

Provides the single seeded random source that all stochastic components draw from, guaranteeing
identical subsets and metrics across runs sharing a seed, and exposes the seed for artifact
recording. Enforces CON-003 by construction: there is one shared RNG; components call
:func:`get_rng` rather than seeding independently.

This module intentionally does not import :mod:`config` (COMP-026 has ``Dependencies: none``). The
seed is passed in by the wiring layer, typically as ``init_rng(load_config().seed)``.

Satisfies COMP-026 -> REQ-021 (seed layer; end-to-end reproducibility proven in TASK-107).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class SeededRng:
    """The single seeded random source.

    Holds both a numpy ``Generator`` and a stdlib ``Random`` seeded from the same integer seed, plus
    the readable ``seed`` value for artifact recording.
    """

    seed: int
    numpy: np.random.Generator
    python: random.Random

    @classmethod
    def from_seed(cls, seed: int) -> "SeededRng":
        return cls(seed=seed, numpy=np.random.default_rng(seed), python=random.Random(seed))


_RNG: Optional[SeededRng] = None


def init_rng(seed: int) -> SeededRng:
    """Initialize (or re-initialize) the single shared RNG from ``seed``."""
    global _RNG
    _RNG = SeededRng.from_seed(seed)
    return _RNG


def get_rng() -> SeededRng:
    """Return the single shared RNG.

    Raises ``RuntimeError`` if called before :func:`init_rng`, so no component can silently fall
    back to an unseeded source (CON-003).
    """
    if _RNG is None:
        raise RuntimeError(
            "RNG not initialized; call init_rng(seed) first "
            "(CON-003: all stochastic behavior draws from the single seeded source)."
        )
    return _RNG


def reset_rng() -> None:
    """Clear the shared RNG (returns to the uninitialized state)."""
    global _RNG
    _RNG = None
