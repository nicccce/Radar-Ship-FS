"""IRFS reinforced-method registry & runners (TASK-412/501 / COMP-011).

Names the reinforced methods, builds each method's engine, and runs the selected methods end-to-end —
both as a standalone ordered dict (:func:`run_reinforced_methods`) and as subset-contract Selectors for
the unified PHASE-005 comparison (:func:`build_reinforced_selectors`).

Every method EXCEPT ``marlfs`` is configured over ONE shared state representation and ONE shared reward
(built through :func:`~methods.engine_builders.build_advised_engine`) and differs ONLY by the active
trainer — the injected advisor, never by code fork (DEC-001, REQ-010 / AC-005).

MARLFS is the deliberate exception (the paper's §4.3/§4.4 baseline, §4.5 bottom rung): the faithful
vanilla multi-agent RL baseline that uses NONE of the paper's three contributions — no trainer, the
minimal ``[relevance, redundancy]`` state (not the tree-structured/GCN encoder), and the uniform overall
reward ``Acc−βR`` (not the personalized per-agent reward). It is therefore the bare
:class:`~engine.explore.ReinforcedEngine` substrate, built via its own branch in
:func:`build_reinforced_engine` — the one reinforced method that does not share the IRFS state+reward and
ignores ``config.state_encoder`` by design.

Fidelity note (Q1, user-confirmed): the reference's headline proposed method *is* Hybrid Teaching
(§3.1.3, §4.5 "IRFS with HT"), so ``full_irfs`` uses the hybrid advisor directly. There is no separate
``hybrid`` method — it would be byte-identical to ``full_irfs`` (same advisor), so it was removed.

Each method runs on its own clone of the context's RNG restored to the state it was handed in, so every
method starts from the identical exploration state, is independently reproducible under the seed (CON-003),
and leaves the caller's RNG untouched — the basis TASK-413's reproducibility proof rests on.
"""

from __future__ import annotations

import copy
import random
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

import numpy as np

from engine.explore import ReinforcedEngine
from engine.reward_overall import OverallReward
from engine.state_minimal import MinimalStateEncoder
from methods.advice import (
    build_dt_importance_advisor,
    build_hybrid_advisor,
    build_relevance_advisor,
)
from methods.engine_builders import build_advised_engine
from rng import SeededRng

if TYPE_CHECKING:
    from config import IrfsConfig
    from engine.seam import ActionAdvisor
    from harness.contract import SelectionContext, SubsetSelection


REINFORCED_METHOD_NAMES: tuple[str, ...] = (
    "marlfs",
    "no_trainer",
    "relevance",
    "dt_importance",
    "full_irfs",
)

HEADLINE_REINFORCED_METHOD_NAMES: tuple[str, ...] = ("marlfs", "no_trainer", "full_irfs")
DIAGNOSTIC_REINFORCED_METHOD_NAMES: tuple[str, ...] = (
    "relevance",
    "dt_importance",
)

# name -> factory turning the effective config into that method's advisor (None = no-trainer).
# ``full_irfs`` is Hybrid Teaching (the reference headline): it uses the hybrid advisor directly.
_ADVISOR_FACTORIES: Dict[str, Callable[["IrfsConfig"], "Optional[ActionAdvisor]"]] = {
    "no_trainer": lambda config: None,
    "relevance": lambda config: build_relevance_advisor(),
    "dt_importance": lambda config: build_dt_importance_advisor(),
    "full_irfs": lambda config: build_hybrid_advisor(config),  # Hybrid Teaching (reading A)
}


def reinforced_method_names(*, include_diagnostic_ablations: bool = False) -> tuple[str, ...]:
    """Return the reinforced methods for a run.

    The default is the headline study surface: no-trainer RL vs full IRFS. Diagnostic trainer ablations
    are opt-in so production runs and artifacts stay aligned with the current requirements.
    """
    if include_diagnostic_ablations:
        return REINFORCED_METHOD_NAMES
    return HEADLINE_REINFORCED_METHOD_NAMES


def build_reinforced_engine(name: str, config: "IrfsConfig") -> ReinforcedEngine:
    """Build the reinforced engine for method ``name``.

    ``marlfs`` is the faithful vanilla baseline and takes its own branch: the bare
    :class:`~engine.explore.ReinforcedEngine` substrate — the minimal ``[relevance, redundancy]``
    state, the uniform overall reward ``Acc−βR``, and no advisor — i.e. none of the three IRFS
    contributions. It is built explicitly (rather than relying on the engine defaults) so it is robust to
    any change in those defaults, and it ignores ``config.state_encoder`` by design.

    For every other (IRFS) method only the advisor varies with ``name``; the personalized reward is bound
    identically, and the state encoder is selected from ``config.state_encoder`` (TASK-003) — the SAME for
    every IRFS method in a run: ``"fixed"`` (default) keeps the baseline byte-identical (AC-007),
    ``"trained_gcn"`` drops the trainable encoder in behind the unchanged seam. A ``trained_gcn`` encoder
    seeds itself from ``context.rng`` on its first encode (CON-003), which is the per-method RNG clone the
    runner restores — so it stays reproducible under the seed. Unknown names raise ``ValueError``.
    """
    if name == "marlfs":
        return ReinforcedEngine(
            encoder=MinimalStateEncoder(),
            reward=OverallReward(),
            advisor=None,
        )
    if name not in _ADVISOR_FACTORIES:
        raise ValueError(f"Unknown reinforced method {name!r}; expected one of {REINFORCED_METHOD_NAMES}")
    return build_advised_engine(advisor=_ADVISOR_FACTORIES[name](config), config=config)


def _rng_snapshot(rng: SeededRng) -> tuple:
    """Capture the exact state of a :class:`SeededRng` so it can be restored later (deep-copied)."""
    return (copy.deepcopy(rng.numpy.bit_generator.state), rng.python.getstate())


def _rng_from_snapshot(snapshot: tuple, seed: int) -> SeededRng:
    """Build a fresh :class:`SeededRng` resumed at a captured snapshot — an independent clone."""
    np_state, py_state = snapshot
    generator = np.random.default_rng()
    generator.bit_generator.state = copy.deepcopy(np_state)
    python_rng = random.Random()
    python_rng.setstate(py_state)
    return SeededRng(seed=seed, numpy=generator, python=python_rng)


def run_reinforced_methods(
    context: "SelectionContext",
    *,
    include_diagnostic_ablations: bool = False,
    on_method=None,
    on_step=None,
) -> "Dict[str, SubsetSelection]":
    """Run the selected reinforced methods end-to-end, returning ``{name: SubsetSelection}`` in
    order.

    The split and the shared probe are reused unchanged; each method runs on its own clone of the
    context's RNG restored to the state it was handed in (after the split/probe were built — the
    same state the standalone no-trainer engine sees, so the no-trainer method reproduces TASK-406).
    Every method therefore starts from the identical exploration state and differs only by its
    trainer, and each is independently reproducible under the seed (CON-003, the basis TASK-413's
    proof rests on). The caller's RNG is left untouched (methods use clones).

    ``on_method`` and ``on_step`` are optional, observational progress hooks for long runs (default
    ``None`` → silent, so callers and tests are unaffected and the result is bit-identical):
    ``on_method(index, total, name, phase)`` fires with ``phase`` ``"start"`` before a method runs
    and ``"done"`` after; ``on_step(name, step, budget, accuracy, best_accuracy)`` forwards the
    engine's per-step hook with the active method name prepended.
    """
    config = context.config
    snapshot = _rng_snapshot(context.rng)
    method_names = reinforced_method_names(include_diagnostic_ablations=include_diagnostic_ablations)
    total = len(method_names)
    results: "Dict[str, SubsetSelection]" = {}
    for index, name in enumerate(method_names, start=1):
        if on_method is not None:
            on_method(index, total, name, "start")
        fresh = context._replace(rng=_rng_from_snapshot(snapshot, context.rng.seed))
        step_hook = (
            (lambda step, budget, acc, best, _name=name: on_step(_name, step, budget, acc, best))
            if on_step is not None
            else None
        )
        results[name] = build_reinforced_engine(name, config).select(fresh, on_step=step_hook)
        if on_method is not None:
            on_method(index, total, name, "done")
    return results


# --- Reinforced methods as Selectors (TASK-501) ------------------------------------------------
#
# The unified PHASE-005 comparison drives the classical methods plus the selected reinforced methods
# through MethodOrchestrator.run / run_comparison on the ONE shared context, so each reinforced method must
# be expressed as a subset-contract Selector. :func:`run_reinforced_methods` above runs the methods on
# their own RNG clones (each restored from a snapshot of the context RNG, the shared RNG untouched);
# :class:`_ReinforcedSelector` reproduces exactly that semantics inside the orchestrator loop, so the
# unified comparison yields the same per-method subsets/series as the standalone PHASE-004 run.


def _named_step(on_step, name):
    """Adapt the public per-step hook ``on_step(name, step, budget, acc, best)`` to the engine's
    ``(step, budget, acc, best)`` seam by binding the active method ``name`` (``None`` -> silent).

    Mirrors the wrapping :func:`run_reinforced_methods` applies, so both paths emit identical hooks.
    """
    if on_step is None:
        return None
    return lambda step, budget, acc, best: on_step(name, step, budget, acc, best)


class _ReinforcedSelector:
    """One reinforced method as a subset-contract Selector with snapshot-pinned RNG.

    Wraps :func:`build_reinforced_engine` for ``name`` and, on every ``select(context)``, discards
    the orchestrator's (possibly already-advanced) shared RNG and runs the engine on a fresh clone
    restored from the captured initial ``snapshot`` — exactly what :func:`run_reinforced_methods`
    does per method. The method is therefore independent of any selector run before it in the same
    pass and reproduces its TASK-406/412 numbers bit-for-bit (same snapshot, same active seed, same
    primitives). A fresh engine is built per ``select`` so each carries its own memoization /
    frequency state (as elsewhere here).
    """

    def __init__(self, name: str, config: "IrfsConfig", snapshot: tuple, *, on_step=None) -> None:
        self._name = name
        self._config = config
        self._snapshot = snapshot
        self._on_step = on_step

    def select(self, context: "SelectionContext") -> "SubsetSelection":
        """Run the reinforced engine on a clone of the pinned initial RNG, returning subset +
        series."""
        fresh = context._replace(rng=_rng_from_snapshot(self._snapshot, context.rng.seed))
        return build_reinforced_engine(self._name, self._config).select(
            fresh, on_step=_named_step(self._on_step, self._name)
        )


def build_reinforced_selectors(
    context: "SelectionContext",
    *,
    include_diagnostic_ablations: bool = False,
    on_step=None,
) -> "List[tuple[str, _ReinforcedSelector]]":
    """Build the selected reinforced methods as ``(name, Selector)`` pairs sharing one RNG snapshot.

    INVARIANT: the snapshot is captured HERE, from ``context.rng``, which must be the initial post-build
    state — so call this BEFORE the comparison runs (``MethodOrchestrator.run`` advances the shared RNG
    once selectors start). All returned selectors restore from this one snapshot on ``select``, so
    they are mutually order-independent and each reproduces :func:`run_reinforced_methods` (TASK-412/413).

    ``on_step``, when given, is the observational per-step hook ``on_step(name, step, budget, accuracy,
    best)`` forwarded verbatim to every engine (default ``None`` -> silent, bit-identical run).
    """
    snapshot = _rng_snapshot(context.rng)  # initial state — see the INVARIANT above
    config = context.config
    method_names = reinforced_method_names(include_diagnostic_ablations=include_diagnostic_ablations)
    return [(name, _ReinforcedSelector(name, config, snapshot, on_step=on_step)) for name in method_names]
