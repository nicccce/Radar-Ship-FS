"""IRFS method composition root (COMP-011) — re-export facade over the split submodules.

The reinforced-method wiring was split by concern into three cohesive modules; this module re-exports
their public surface so ``from methods.configure import …`` keeps resolving unchanged:

- :mod:`methods.seam_adapters` — the state/reward seam adapters that bridge the feedback modules onto the
  engine's per-agent seam.
- :mod:`methods.engine_builders` — encoder selection + engine assembly over the shared IRFS state/reward.
- :mod:`methods.reinforced_run` — the reinforced-method registry, per-method runners, and Selectors.

Import from the concrete submodules for new code; this facade exists to preserve the established import
path (and its test surface).
"""

from __future__ import annotations

from methods.engine_builders import (
    build_advised_engine,
    build_advised_engine_with_registration,
    build_no_trainer_engine,
    run_no_trainer,
    select_state_encoder,
)
from methods.reinforced_run import (
    _ADVISOR_FACTORIES,
    DIAGNOSTIC_REINFORCED_METHOD_NAMES,
    HEADLINE_REINFORCED_METHOD_NAMES,
    REINFORCED_METHOD_NAMES,
    _named_step,
    _ReinforcedSelector,
    _rng_from_snapshot,
    _rng_snapshot,
    build_reinforced_engine,
    build_reinforced_selectors,
    reinforced_method_names,
    run_reinforced_methods,
)
from methods.seam_adapters import (
    PersonalizedRewardSeamAdapter,
    TrainableGCNSeamAdapter,
    TreeStateSeamAdapter,
    _is_frequency,
)

__all__ = [
    # seam adapters
    "TreeStateSeamAdapter",
    "TrainableGCNSeamAdapter",
    "PersonalizedRewardSeamAdapter",
    "_is_frequency",
    # engine builders
    "select_state_encoder",
    "build_advised_engine",
    "build_advised_engine_with_registration",
    "build_no_trainer_engine",
    "run_no_trainer",
    # reinforced-method registry & runners
    "REINFORCED_METHOD_NAMES",
    "HEADLINE_REINFORCED_METHOD_NAMES",
    "DIAGNOSTIC_REINFORCED_METHOD_NAMES",
    "_ADVISOR_FACTORIES",
    "reinforced_method_names",
    "build_reinforced_engine",
    "run_reinforced_methods",
    "build_reinforced_selectors",
    "_ReinforcedSelector",
    "_named_step",
    "_rng_snapshot",
    "_rng_from_snapshot",
]
