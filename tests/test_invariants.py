"""Cross-cutting invariants (test domain D8): determinism, leakage, seam.

This file is the single canonical home for the three system-wide guarantees that were
previously re-asserted across six phase/task files (``test_reproducibility``,
``test_phase004_reproducibility``, ``test_task004_parity_substrate_repro``,
``test_task006_leakage_reproducibility``, ``test_seam_swappable``, ``test_engine_seam``).
The per-domain suites (D1–D7) test their own behaviour; they no longer re-prove these
invariants. The guarantees:

1. **Determinism (CON-003 / CON-R-001).** One shared seeded RNG drives everything; two
   independent same-seed wirings reproduce byte-for-byte, and a different seed changes the
   result (so the identity is genuinely seed-driven, not a constant). Proven on the most
   randomness-heavy path (the trained GCN encoder: init + joint optimizer + mini-batch
   sampling + ε-greedy + trainer draw), across every reinforced method, and at the
   comparison/artifact level.
2. **Leakage (AC-004).** A full trained run never reaches the held-out test partition — proven
   at runtime by a sentinel Split that raises on any test access, plus a proof the sentinel is
   real (it fires the instant test is touched), so a green run cannot be a vacuous false pass.
3. **Seam (AC-011 / REQ-014 / RISK-004).** Alternate-shaped state/reward stand-ins satisfy the
   seam structurally (zero inheritance) and drive the unmodified production engine end to end.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from config import load_config
from data.loader import load
from data.splitter import Split, make_split
from engine.explore import ReinforcedEngine
from engine.seam import RewardFunction, StateEncoder
from harness.artifact import build_artifact
from harness.comparison import run_comparison
from harness.contract import SelectionContext, Selector, SubsetSelection
from harness.orchestrator import MethodOrchestrator
from methods.advice import build_hybrid_advisor
from methods.configure import (
    REINFORCED_METHOD_NAMES,
    build_advised_engine_with_registration,
    run_reinforced_methods,
)
from methods.relevance_topk import RelevanceTopKSelector
from probe import DecisionTreeProbe
from rng import SeededRng

# --- Constants -------------------------------------------------------------------------------------

_SEED = 42
_OTHER_SEED = 7

# Short budgets keep runs fast; these invariants are about determinism/isolation/substitution,
# not convergence. Several steps so per-step series are non-trivial and the joint optimizer updates
# encoder ∪ heads multiple times.
_TRAINED_BUDGET = 6
_REINFORCED_BUDGET = 8
_COMPARISON_BUDGET = 12
_SEAM_BUDGET = 12


# --- Wiring (independent each call) — one shared rewire pattern ------------------------------------


def _wire(*, seed: int = _SEED, budget: int, encoder: str = "fixed", **overrides) -> SelectionContext:
    """A real WDBC selection context wired from the seed (fresh RNG/split/probe each call).

    Each call builds its own ``SeededRng.from_seed``, split, and probe, so any identity proven
    across two calls is a genuine same-seed re-execution — not an artifact of one shared context.
    """
    config = load_config(
        {"seeds": (seed,), "exploration_step_budget": budget, "state_encoder": encoder, **overrides}
    )
    rng = SeededRng.from_seed(config.seeds[0])
    data = load(config)
    split = make_split(data, config, rng)
    probe = DecisionTreeProbe(split.train, config, rng)
    return SelectionContext(split=split, probe=probe, config=config, rng=rng)


def _series(selection: SubsetSelection) -> list[tuple[tuple[int, ...], float]]:
    """The full per-step series as comparable ``(subset, accuracy)`` pairs."""
    return [(step.subset, step.accuracy) for step in selection.per_step]


def _params_equal(first, second) -> bool:
    """Exact (bitwise on CPU) equality of two flat parameter snapshots."""
    return len(first) == len(second) and all(torch.equal(a, b) for a, b in zip(first, second))


def _heads_equal(first, second) -> bool:
    """Exact equality of two ``[(feature, [param, ...]), ...]`` head snapshots."""
    if len(first) != len(second):
        return False
    for (fa, layers_a), (fb, layers_b) in zip(first, second):
        if fa != fb or len(layers_a) != len(layers_b):
            return False
        if not all(torch.equal(x, y) for x, y in zip(layers_a, layers_b)):
            return False
    return True


def _trained_snapshot(seed: int = _SEED):
    """Run one trained_gcn ``full_irfs`` selection; snapshot subset, per-step, encoder & head params.

    ``full_irfs`` (hybrid teaching) is the headline advised method — it exercises the most randomness.
    The live encoder params are the tensors the joint optimizer owns (read after ``select`` = learned
    weights); the heads are captured by wrapping ``build_agents`` at the module the engine uses.
    """
    context = _wire(seed=seed, budget=_TRAINED_BUDGET, encoder="trained_gcn")
    built, live_encoder = build_advised_engine_with_registration(
        advisor=build_hybrid_advisor(context.config),
        config=context.config,
        rng=context.rng,  # eager build → params are the live tensors before any update
    )

    import engine.explore as explore_mod

    captured: dict = {}
    real_build_agents = explore_mod.build_agents

    def capturing_build_agents(ctx, enc):
        agents = real_build_agents(ctx, enc)
        captured["agents"] = agents
        return agents

    explore_mod.build_agents = capturing_build_agents
    try:
        selection = built.select(context)
    finally:
        explore_mod.build_agents = real_build_agents

    encoder_params = [p.detach().clone() for p in live_encoder]
    head_params = [
        (agent.feature, [p.detach().clone() for p in agent.policy.parameters()])
        for agent in captured["agents"]
    ]
    return selection, encoder_params, head_params


# === Determinism ==================================================================================


def test_full_trained_run_reproducible_same_seed() -> None:
    """Two independent same-seed trained_gcn runs are byte-identical: subset, full per-step metrics,
    learned ENCODER params, and every per-agent HEAD param.

    The strongest determinism path — encoder init, joint optimizer, and mini-batch sampling are all
    in play (AC-005 / AC-008 / CON-R-001).
    """
    sel_a, enc_a, heads_a = _trained_snapshot()
    sel_b, enc_b, heads_b = _trained_snapshot()

    assert sel_a.selected == sel_b.selected
    assert _series(sel_a) == _series(sel_b)
    assert len(sel_a.per_step) == _TRAINED_BUDGET  # a genuine full run, not an early-out
    assert len(enc_a) == 2 and _params_equal(enc_a, enc_b)  # one weight + one bias, bit-identical
    assert len(heads_a) == 30 and _heads_equal(heads_a, heads_b)  # one head per WDBC feature


def test_all_reinforced_methods_reproduce_same_seed() -> None:
    """Across the trainer dimension: every one of the five reinforced methods reproduces its subset
    and full per-step series across two independent same-seed runs of the whole runner (AC-008)."""
    first = run_reinforced_methods(_wire(budget=_REINFORCED_BUDGET), include_diagnostic_ablations=True)
    second = run_reinforced_methods(_wire(budget=_REINFORCED_BUDGET), include_diagnostic_ablations=True)

    assert set(first) >= set(REINFORCED_METHOD_NAMES)  # all five methods carried
    for name in REINFORCED_METHOD_NAMES:
        assert first[name].selected == second[name].selected
        assert _series(first[name]) == _series(second[name])


def test_comparison_run_reproducible_and_records_seed() -> None:
    """At the comparison/artifact level: two independent same-seed comparison runs produce identical
    selected subsets (both methods) and identical windowed Best/Average metrics, and the emitted
    artifact records the seed (REQ-013 / AC-010).

    Also covers the fixed/minimal engine path.
    """

    def _run():
        config = load_config({"seeds": (_SEED,), "exploration_step_budget": _COMPARISON_BUDGET})
        orchestrator = MethodOrchestrator(config)
        comparison = run_comparison(
            orchestrator,
            [("relevance_topk", RelevanceTopKSelector()), ("reinforced", ReinforcedEngine())],
        )
        artifact = build_artifact(comparison.runs, orchestrator.context, comparison=comparison)
        return comparison, artifact

    first, artifact = _run()
    second, _ = _run()

    subsets_first = {r.name: r.selected for r in first.runs}
    subsets_second = {r.name: r.selected for r in second.runs}
    assert set(subsets_first) == {"relevance_topk", "reinforced"}
    assert subsets_first == subsets_second

    assert "reinforced" in first.windowed
    assert "relevance_topk" not in first.windowed  # classical single-shot has no windowed metrics
    assert first.windowed == second.windowed

    assert artifact["seed"] == _SEED
    assert artifact["config"]["seeds"] == (_SEED,)


def test_different_seed_changes_run() -> None:
    """The determinism above is seed-driven, not a constant: a different seed yields a different
    learned trained run — different learned encoder params and/or a different subset/per-step
    series.

    Proves the encoder init, optimizer, and sampling are all driven by the single shared seed (no
    self-seeding).
    """
    sel_42, enc_42, _ = _trained_snapshot(seed=_SEED)
    sel_7, enc_7, _ = _trained_snapshot(seed=_OTHER_SEED)

    assert (
        sel_42.selected != sel_7.selected
        or _series(sel_42) != _series(sel_7)
        or not _params_equal(enc_42, enc_7)
    )


# === Leakage ======================================================================================


class HeldoutTestPartitionTouched(AssertionError):
    """Raised the instant a trained run reaches the held-out test partition (leakage tripwire).

    Subclasses ``AssertionError`` so a leak surfaces as a hard failure, not a silently-handled
    exception some defensive ``except Exception`` could swallow.
    """


class _SentinelSplit:
    """A split stand-in that raises if selection reads the test partition.

    Train and validation are real so the selection run otherwise behaves normally.
    """

    def __init__(self, real: Split) -> None:
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "train", real.train)
        object.__setattr__(self, "validation", real.validation)

    @property
    def test(self):
        raise HeldoutTestPartitionTouched("selection accessed the test partition")

    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, "_real"), name)


def _build_trained(context: SelectionContext):
    """A trained-encoder engine (hybrid ``full_irfs`` advisor) for ``context``, joint training
    on."""
    engine, _params = build_advised_engine_with_registration(
        advisor=build_hybrid_advisor(context.config),
        config=context.config,
        rng=context.rng,
    )
    return engine


def test_trained_run_never_touches_test_partition() -> None:
    """AC-004: a FULL trained_gcn run (joint training on) completes end-to-end against a sentinel
    context whose test partition RAISES on any access — proving encoder forward, joint optimizer
    updates, and produced state values read only train/validation, never test."""
    context = _wire(encoder="trained_gcn", budget=_TRAINED_BUDGET)
    sentinel_context = context._replace(split=_SentinelSplit(context.split))

    selection = _build_trained(sentinel_context).select(sentinel_context)

    assert len(selection.per_step) == _TRAINED_BUDGET  # a genuine full run, not an early-out
    assert 0 < len(selection.selected) < context.n_features  # non-degenerate real selection


def test_leakage_sentinel_is_real() -> None:
    """The sentinel raises when its test partition is accessed."""
    sentinel = _SentinelSplit(_wire(encoder="trained_gcn", budget=_TRAINED_BUDGET).split)

    with pytest.raises(HeldoutTestPartitionTouched):
        _ = sentinel.test


# === Seam =========================================================================================
# Stand-ins deliberately do NOT inherit from the seam Protocols — conformance must be structural.


class _TinyState:
    """A context-free 2-dimensional encoder; length is what matters for the structural check."""

    dimension = 2

    def encode(self, feature, selected, context) -> np.ndarray:
        return np.array([float(feature), float(len(selected))], dtype=float)


class _WideState:
    """A differently-shaped 5-dimensional encoder — proves the seam admits another length."""

    dimension = 5

    def encode(self, feature, selected, context) -> np.ndarray:
        return np.zeros(self.dimension, dtype=float)


class _NotAnEncoder:
    """Has no ``encode`` — must not satisfy the state seam."""


class _UniformReward:
    """Ignores ``agent`` and applies one value uniformly (the overall-reward shape)."""

    def reward(self, selected, context, *, agent=None) -> float:
        return float(len(selected))


class _PersonalizedReward:
    """Reads ``agent`` to return a per-agent value (the personalized-reward shape)."""

    def reward(self, selected, context, *, agent=None) -> float:
        return float(len(selected)) + float(agent or 0)


class _NotAReward:
    """Has no ``reward`` method — must not satisfy the reward seam."""


class _LiveState:
    """A context-reading 4-dimensional encoder (vs the minimal state's 2) with an unrelated feature
    set — proves the engine reads ``dimension`` and drives an alternate substrate, not the minimal
    one."""

    dimension = 4

    def encode(self, feature, selected, context) -> np.ndarray:
        column = context.split.train.X[:, feature]
        return np.array(
            [
                float(feature) / context.n_features,
                float(len(selected)),
                float(column.mean()),
                float(column.std()),
            ],
            dtype=float,
        )


class _CompactReward:
    """A reward with a completely different formula — compactness, ignoring accuracy/correlation."""

    def reward(self, selected, context, *, agent=None) -> float:
        return 1.0 / len(selected)


def test_seam_accepts_alternate_shaped_standins() -> None:
    """Structural conformance (zero inheritance): alternate-dimensioned encoders and both reward
    shapes are recognized by structure; the encoded length equals the declared ``dimension``
    independently of subset size (CON-005); non-conformers are rejected; and one call site serves
    uniform and per-agent rewards through the same signature (RISK-004)."""
    for encoder in (_TinyState(), _WideState()):
        assert isinstance(encoder, StateEncoder)
        assert StateEncoder not in type(encoder).__mro__  # purely structural
        small = encoder.encode(feature=0, selected=[1], context=None)
        large = encoder.encode(feature=0, selected=[1, 2, 3, 4], context=None)
        assert small.shape == (encoder.dimension,) and large.shape == (encoder.dimension,)
    assert _TinyState().dimension != _WideState().dimension  # the seam is not length-locked
    assert not isinstance(_NotAnEncoder(), StateEncoder)

    for reward in (_UniformReward(), _PersonalizedReward()):
        assert isinstance(reward, RewardFunction)
        assert RewardFunction not in type(reward).__mro__  # purely structural
    assert not isinstance(_NotAReward(), RewardFunction)

    selected = [0, 1, 2]
    assert _UniformReward().reward(selected, None, agent=0) == _UniformReward().reward(
        selected, None, agent=7
    )
    assert _PersonalizedReward().reward(selected, None, agent=0) != _PersonalizedReward().reward(
        selected, None, agent=7
    )


def test_engine_runs_unchanged_through_injected_seam() -> None:
    """AC-011: the unmodified production ``ReinforcedEngine`` (same class, no subclass) drives an
    alternate substrate — a 4-dim stand-in encoder and a compactness reward reaching it only through
    the public constructor seam — to a non-degenerate subset + full series, with zero engine/agent/
    contract edits.

    The same class also runs the minimal defaults (only the seam contents differ).
    """
    context = _wire(budget=_SEAM_BUDGET)
    engine = ReinforcedEngine(encoder=_LiveState(), reward=_CompactReward())

    assert isinstance(engine, Selector)
    assert type(engine) is ReinforcedEngine  # production class, no subclassing required

    selection = engine.select(context)
    assert isinstance(selection, SubsetSelection)
    assert len(selection.per_step) == _SEAM_BUDGET
    assert 0 < len(selection.selected) < context.n_features
    assert all(0.0 <= step.accuracy <= 1.0 for step in selection.per_step)

    # The identical engine class also runs the minimal default substrate.
    default = ReinforcedEngine().select(_wire(budget=_SEAM_BUDGET))
    assert len(default.per_step) == _SEAM_BUDGET
    assert 0 < len(default.selected) < context.n_features
