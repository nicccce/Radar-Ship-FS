"""Methods domain (D6): the classical baselines and the reinforced-method configurator —
``methods/*``.

Two halves:

- **Classical baselines** (``relevance_topk`` / ``dt_rfe`` / ``mrmr`` / ``l1``): the four selectors compose
  through one orchestrator on one shared leakage-safe split, each scored through the shared probe; the three
  fixed-size baselines select half the features while L1 is penalty-driven; mRMR surfaces a pinned identity.
- **Reinforced configurator** (``methods/configure.py`` + ``seam_adapters.py``): the five-method registry;
  each method runs end-to-end to a non-degenerate subset + full series; every IRFS method shares one
  tree-state adapter and one personalized-reward adapter, varying only the advisor (``marlfs`` is the bare
  MARLFS exception); the seam adapters conform structurally and carry per-agent state + personalized reward.

Run determinism and run-level leakage safety are proven once in ``test_invariants.py`` (D8) and not
re-asserted here.
"""

from __future__ import annotations

import numpy as np
import pytest

from config import load_config
from data.loader import load
from data.splitter import make_split
from engine.reward_overall import OverallReward
from engine.seam import RewardFunction, StateEncoder
from engine.state_minimal import MinimalStateEncoder
from harness.contract import SelectionContext, Selector, SubsetSelection
from harness.orchestrator import MethodOrchestrator
from methods.configure import (
    DIAGNOSTIC_REINFORCED_METHOD_NAMES,
    HEADLINE_REINFORCED_METHOD_NAMES,
    REINFORCED_METHOD_NAMES,
    PersonalizedRewardSeamAdapter,
    TreeStateSeamAdapter,
    build_no_trainer_engine,
    build_reinforced_engine,
    reinforced_method_names,
    run_no_trainer,
    run_reinforced_methods,
)
from methods.dt_rfe import DTImportanceEliminator
from methods.l1 import L1Selector
from methods.mrmr import MRMRSelector, implementation_identity
from methods.relevance_topk import RelevanceTopKSelector
from probe import DecisionTreeProbe
from reward.overall import overall_reward
from rng import SeededRng
from state.encoder import TreeStructuredStateEncoder

_RUN_BUDGET = 8


def wire_context(**overrides) -> SelectionContext:
    """A real WDBC selection context (fixed seed, short budget) via the PHASE-001 wiring chain."""
    config = load_config({"exploration_step_budget": _RUN_BUDGET, **overrides})
    rng = SeededRng.from_seed(config.seeds[0])
    data = load(config)
    split = make_split(data, config, rng)
    probe = DecisionTreeProbe(split.train, config, rng)
    return SelectionContext(split=split, probe=probe, config=config, rng=rng)


@pytest.fixture()
def context() -> SelectionContext:
    return wire_context()


def _classical_selectors() -> list:
    """The four classical baselines, fresh instances, in a stable order."""
    return [
        ("relevance_topk", RelevanceTopKSelector()),
        ("dt_rfe", DTImportanceEliminator()),
        ("mrmr", MRMRSelector()),
        ("l1", L1Selector()),
    ]


def _run_suite(config=None):
    """Run all four classical baselines through ONE orchestrator on the shared context."""
    orchestrator = MethodOrchestrator(config or load_config())
    return orchestrator, orchestrator.run(_classical_selectors())


# === Classical baselines ==========================================================================


def test_classical_suite_composes_through_one_orchestrator() -> None:
    """The four classical selectors run through one orchestrator on one shared context, each
    returning a canonical non-empty subset scored through the shared probe (single-shot: no per-step
    series)."""
    orchestrator, runs = _run_suite()
    n = orchestrator.context.n_features

    assert [r.name for r in runs] == ["relevance_topk", "dt_rfe", "mrmr", "l1"]
    assert n > 0
    for r in runs:
        assert r.size == len(r.selected) >= 1  # non-empty
        assert all(0 <= i < n for i in r.selected)  # valid column indices
        assert r.selected == tuple(sorted(set(r.selected)))  # canonical (contract)
        assert 0.0 <= r.accuracy <= 1.0  # scored through the DT probe
        assert r.per_step == ()  # classical single-shot — no series


def test_classical_sizing_fixed_half_vs_variable_l1() -> None:
    """The three fixed-size baselines select exactly half the features; L1 is variable-size and
    penalty-driven — a stronger penalty yields a different subset size (not the half rule)."""
    orchestrator, runs = _run_suite()
    n = orchestrator.context.n_features
    by_name = {r.name: r for r in runs}

    for name in ("relevance_topk", "dt_rfe", "mrmr"):
        assert by_name[name].size == n // 2, (name, by_name[name].size, n // 2)

    l1 = by_name["l1"]
    assert 1 <= l1.size <= n
    _, strong_runs = _run_suite(load_config({"l1_C": 0.01}))
    strong_l1 = next(r for r in strong_runs if r.name == "l1")
    assert strong_l1.size != l1.size  # size driven by l1_C, not n // 2


def test_mrmr_pinned_identity_is_surfaced() -> None:
    """The pinned mRMR implementation exposes a readable name + version record for the artifact."""
    identity = implementation_identity()
    assert identity["name"] == "mrmr-selection"
    assert isinstance(identity["version"], str) and identity["version"]


# === Reinforced-method registry & assembly ========================================================


def test_reinforced_registry_and_unknown_rejected(context: SelectionContext) -> None:
    """The registry is exactly the five ordered methods (headline vs diagnostic split), and an
    unknown method name is rejected."""
    assert REINFORCED_METHOD_NAMES == (
        "marlfs",
        "no_trainer",
        "relevance",
        "dt_importance",
        "full_irfs",
    )
    assert HEADLINE_REINFORCED_METHOD_NAMES == ("marlfs", "no_trainer", "full_irfs")
    assert DIAGNOSTIC_REINFORCED_METHOD_NAMES == ("relevance", "dt_importance")
    assert reinforced_method_names() == HEADLINE_REINFORCED_METHOD_NAMES
    assert reinforced_method_names(include_diagnostic_ablations=True) == REINFORCED_METHOD_NAMES

    with pytest.raises(ValueError, match="Unknown reinforced method"):
        build_reinforced_engine("kbest", context.config)


def test_methods_run_end_to_end_headline_and_diagnostic(context: SelectionContext) -> None:
    """The headline methods run end-to-end to a non-degenerate subset + full per-step series; the
    diagnostic flag additionally runs the two trainer ablations (all five)."""
    headline = run_reinforced_methods(context)
    assert list(headline) == list(HEADLINE_REINFORCED_METHOD_NAMES)
    for name in HEADLINE_REINFORCED_METHOD_NAMES:
        selection = headline[name]
        assert isinstance(selection, SubsetSelection)
        assert len(selection.per_step) == _RUN_BUDGET
        assert 0 < len(selection.selected) < context.n_features
        assert all(0.0 <= step.accuracy <= 1.0 for step in selection.per_step)

    diagnostic = run_reinforced_methods(wire_context(), include_diagnostic_ablations=True)
    assert list(diagnostic) == list(REINFORCED_METHOD_NAMES)
    for name in REINFORCED_METHOD_NAMES:
        assert len(diagnostic[name].per_step) == _RUN_BUDGET


def test_irfs_methods_share_state_reward_marlfs_is_bare(context: SelectionContext) -> None:
    """Every IRFS method's engine carries the same tree-state and personalized-reward adapters; only
    the advisor varies (no_trainer has none).

    MARLFS is the deliberate exception: minimal state, uniform
    overall reward, no advisor — none of the three IRFS contributions.
    """
    irfs = [name for name in REINFORCED_METHOD_NAMES if name != "marlfs"]
    engines = {name: build_reinforced_engine(name, context.config) for name in irfs}

    for engine in engines.values():
        assert type(engine._encoder) is TreeStateSeamAdapter
        assert type(engine._reward) is PersonalizedRewardSeamAdapter
    assert engines["no_trainer"]._advisor is None
    assert all(engines[name]._advisor is not None for name in irfs if name != "no_trainer")

    marlfs = build_reinforced_engine("marlfs", context.config)
    assert type(marlfs._encoder) is MinimalStateEncoder
    assert type(marlfs._reward) is OverallReward
    assert marlfs._advisor is None


def test_no_trainer_matches_standalone_and_entry_point() -> None:
    """The runner's no-trainer method, the standalone ``build_no_trainer_engine`` select, and the
    ``run_no_trainer`` entry point all produce the identical result (one contract Selector, three
    doors)."""
    assert isinstance(build_no_trainer_engine(), Selector)

    via_runner = run_reinforced_methods(wire_context())["no_trainer"]
    via_build = build_no_trainer_engine().select(wire_context())
    via_entry = run_no_trainer(wire_context())

    assert via_runner.selected == via_build.selected == via_entry.selected
    assert [s.accuracy for s in via_runner.per_step] == [s.accuracy for s in via_build.per_step]


# === Seam adapters ================================================================================


def test_seam_adapters_conform_and_state_is_per_agent(context: SelectionContext) -> None:
    """Both adapters are recognized by the engine seam structurally (no inheritance); the state
    adapter gives each agent its OWN feature-specific, fixed-width, standardized, memoized state row
    (the A+B fix)."""
    encoder = TreeStateSeamAdapter()
    reward = PersonalizedRewardSeamAdapter()

    assert isinstance(encoder, StateEncoder) and StateEncoder not in type(encoder).__mro__
    assert encoder.dimension == TreeStructuredStateEncoder().dimension
    assert isinstance(reward, RewardFunction) and RewardFunction not in type(reward).__mro__

    selected = [0, 3, 7]
    vectors = {}
    for feature in (0, 3, 7, 11):
        vec = encoder.encode(feature, selected, context)
        assert vec.shape == (encoder.dimension,)
        assert np.abs(vec).max() < 50.0  # standardized — no raw ~thousands scale
        np.testing.assert_array_equal(vec, encoder.encode(feature, selected, context))  # memoized
        vectors[feature] = vec

    distinct = {tuple(np.round(v, 6)) for v in vectors.values()}
    assert len(distinct) == len(vectors)  # per-agent identity (agents do not share one vector)


def test_reward_adapter_personalizes_and_frequency_binds(context: SelectionContext) -> None:
    """The personalized-reward adapter earns a deselected agent exactly zero, a selected agent a
    non-zero shaped reward, and falls back to the overall reward when no agent is supplied; the
    frequency scheme threads history end-to-end and still zeroes a deselected agent."""
    reward = PersonalizedRewardSeamAdapter()
    selected = (0, 1, 2)
    deselected = next(f for f in range(context.n_features) if f not in selected)

    assert reward.reward(selected, context, agent=deselected) == 0.0
    assert any(reward.reward(selected, context, agent=f) != 0.0 for f in selected)
    assert reward.reward(selected, context, agent=None) == pytest.approx(overall_reward(selected, context))

    ctx = wire_context(reward_scheme="frequency")
    selection = run_no_trainer(ctx)
    assert 0 < len(selection.selected) < ctx.n_features

    freq_reward = PersonalizedRewardSeamAdapter()
    freq_reward.reward(selected, ctx, agent=0)  # seed some history
    freq_deselected = next(f for f in range(ctx.n_features) if f not in selected)
    assert freq_reward.reward(selected, ctx, agent=freq_deselected) == 0.0
