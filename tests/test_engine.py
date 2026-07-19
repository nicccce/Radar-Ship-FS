"""Engine domain (D4): agent-action semantics and the trainer-advice seam.

Covers how the RL engine reads agent votes and consumes trainer advice:

- **Agent classification** (``trainers/classify.py``): partitioning feature-agents from their (prior,
  current) actions into participated / assertive / hesitant — the primitive the advice loop keys on.
- **Advice seam** (``engine/seam.py`` ``ActionAdvisor`` + ``methods/advice.py`` ``TrainerAdvisor``): the
  advisor conforms structurally and delegates to its advice source; the engine applies the override map
  to the votes *before* the SELECT-union so a forced override reaches the committed subset; running with
  no advisor (or an advisor that returns no overrides) reproduces the no-trainer run bit-for-bit; and
  advice is train-only.

The engine's deeper internals (``policy`` ε-greedy, ``memory`` replay, ``learner`` TD update, ``agents``
head init) are exercised end to end — determinism, leakage, seam-swap — in ``test_invariants.py`` (D8).
"""

from __future__ import annotations

import pytest

from config import load_config
from data.loader import load
from data.splitter import make_split
from engine.policy import ACTION_DESELECT, ACTION_SELECT
from engine.policy import ACTION_DESELECT as Dsel
from engine.policy import ACTION_SELECT as Sel
from engine.seam import ActionAdvisor
from harness.contract import SelectionContext
from methods.advice import (
    TrainerAdvisor,
    build_dt_importance_advisor,
    build_relevance_advisor,
)
from methods.configure import build_advised_engine, build_no_trainer_engine
from probe import DecisionTreeProbe
from rng import SeededRng
from trainers.classify import AgentClassification, classify_agents

_RUN_BUDGET = 12


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


# === Agent classification (participated / assertive / hesitant) ===================================


def test_action_partition_maps_prior_and_current_to_sets() -> None:
    """participated = prior-SELECT; assertive = SELECT→SELECT; hesitant = SELECT→DESELECT; a fresh
    DESELECT→SELECT vote is NOT a participant. participated == assertive ⊔ hesitant, as ascending
    typed tuples ready to index features (AC-001)."""
    #          feature:  0     1     2     3     4
    prior = [Sel, Sel, Dsel, Dsel, Sel]
    current = [Sel, Dsel, Sel, Dsel, Dsel]

    r = classify_agents(prior, current)

    assert isinstance(r, AgentClassification)
    assert r.assertive == (0,)  # selected in both steps
    assert r.hesitant == (1, 4)  # selected previously, deselected now
    assert r.participated == (
        0,
        1,
        4,
    )  # prior-selected; feature 2's new SELECT is not a participant
    assert set(r.assertive).isdisjoint(r.hesitant)
    assert set(r.participated) == set(r.assertive) | set(r.hesitant)
    assert list(r.hesitant) == sorted(r.hesitant)


def test_degenerate_action_populations() -> None:
    """The boundary populations: nobody selected → all sets empty; everybody selected both steps →
    all assertive; a full S→D flip → all hesitant, none assertive."""
    assert classify_agents([Dsel, Dsel, Dsel], [Dsel, Dsel, Dsel]) == AgentClassification((), (), ())

    all_selected = classify_agents([Sel, Sel, Sel], [Sel, Sel, Sel])
    assert all_selected.assertive == (0, 1, 2) and all_selected.hesitant == ()

    flipped = classify_agents([Sel, Dsel, Sel], [Dsel, Sel, Dsel])
    assert flipped.hesitant == (0, 2) and flipped.assertive == ()  # only S→D flips are hesitant


def test_classification_rejects_malformed_input() -> None:
    """Length mismatch and an out-of-space action both fail loudly."""
    with pytest.raises(ValueError, match="one entry per feature-agent"):
        classify_agents([Sel, Dsel], [Sel, Dsel, Sel])
    with pytest.raises(ValueError, match="not a valid action"):
        classify_agents([Sel, 7], [Sel, Dsel])


def test_classification_of_no_agents_is_empty() -> None:
    """Classifying empty action sequences yields an all-empty partition (no agents to place)."""
    assert classify_agents([], []) == AgentClassification((), (), ())


# === Advice seam (ActionAdvisor + TrainerAdvisor) =================================================


class _ForceAdvisor:
    """Forces one feature ON and one OFF every step — a fixed override map regardless of the
    votes."""

    def __init__(self, on: int, off: int) -> None:
        self._map = {on: ACTION_SELECT, off: ACTION_DESELECT}

    def advise(self, step, prior_actions, current_actions, context):
        return dict(self._map)


class _EmptyAdvisor:
    """A conforming advisor that runs the advice path but never overrides anything."""

    def advise(self, step, prior_actions, current_actions, context):
        return {}


class _FlipSpy:
    """Wraps a real advisor and records the (step, feature) pairs where advice actually changed a
    vote."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.flips: list[tuple[int, int]] = []

    def advise(self, step, prior_actions, current_actions, context):
        overrides = self._inner.advise(step, prior_actions, current_actions, context)
        for feature, action in overrides.items():
            if current_actions[feature] != action:
                self.flips.append((step, feature))
        return overrides


class _PerturbedValidationSplit:
    """Wraps a Split with its (non-test) validation features shifted; train is left untouched."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.train = inner.train
        self.validation = inner.validation._replace(X=inner.validation.X + 1000.0)

    def release_test_for_final_metrics(self):
        return self._inner.release_test_for_final_metrics()


def test_trainer_advisor_satisfies_seam_and_delegates(context: SelectionContext) -> None:
    """The trainer advisors are recognized by the ``ActionAdvisor`` seam structurally (no
    inheritance), and ``advise`` classifies the (prior, current) actions then returns its advice
    source's output verbatim."""
    for advisor in (build_relevance_advisor(), build_dt_importance_advisor()):
        assert isinstance(advisor, ActionAdvisor)
        assert ActionAdvisor not in type(advisor).__mro__

    seen: dict = {}

    def fake_advice_fn(step, classification, ctx):
        seen.update(step=step, classification=classification)
        return {7: ACTION_SELECT}

    prior = [ACTION_SELECT, ACTION_SELECT, ACTION_DESELECT, ACTION_DESELECT]
    current = [ACTION_SELECT, ACTION_DESELECT, ACTION_SELECT, ACTION_DESELECT]
    result = TrainerAdvisor(fake_advice_fn).advise(3, prior, current, context)

    assert result == {7: ACTION_SELECT}
    assert seen["step"] == 3
    assert seen["classification"] == classify_agents(prior, current)


def test_engine_applies_override_to_committed_subset() -> None:
    """A forced override lands on the votes before the SELECT-union, so it appears in the resulting
    subset.

    One step isolates the effect: the returned subset IS that step's overridden vote union.
    """
    ctx = wire_context(exploration_step_budget=1)
    selection = build_advised_engine(_ForceAdvisor(on=5, off=6)).select(ctx)

    assert 5 in selection.selected  # forced SELECT reached the union
    assert 6 not in selection.selected  # forced DESELECT reached the union


def test_no_advisor_and_empty_advice_reproduce_no_trainer() -> None:
    """``build_advised_engine(None)`` equals the no-trainer engine, and an advisor returning no
    overrides reproduces it bit-for-bit — proving the advice/classification path consumes no RNG
    (CON-003)."""
    none_run = build_advised_engine(None).select(wire_context())
    base_run = build_no_trainer_engine().select(wire_context())
    empty_run = build_advised_engine(_EmptyAdvisor()).select(wire_context())

    assert none_run.selected == base_run.selected == empty_run.selected
    series = lambda run: [s.accuracy for s in run.per_step]  # noqa: E731
    assert series(none_run) == series(base_run) == series(empty_run)


def test_real_advisors_flip_a_hesitant_agent() -> None:
    """AC-002: a real relevance run and a real DT-importance run each record at least one hesitant
    agent changing its action after advice (the advice genuinely reaches the engine's votes)."""
    for build in (build_relevance_advisor, build_dt_importance_advisor):
        spy = _FlipSpy(build())
        build_advised_engine(spy).select(wire_context())
        assert spy.flips, f"{build.__name__}: expected a hesitant agent to flip after advice"


def test_advice_is_train_only(context: SelectionContext) -> None:
    """AC-007: perturbing the (non-test) validation partition does not change the advice — it is
    train-fit — for both the relevance and DT-importance advisors."""
    n = context.n_features
    prior = [ACTION_SELECT if i < 10 else ACTION_DESELECT for i in range(n)]
    current = [ACTION_SELECT if i < 5 or 10 <= i < 15 else ACTION_DESELECT for i in range(n)]
    perturbed = context._replace(split=_PerturbedValidationSplit(context.split))

    for build in (build_relevance_advisor, build_dt_importance_advisor):
        assert build().advise(0, prior, current, context) == build().advise(0, prior, current, perturbed)
