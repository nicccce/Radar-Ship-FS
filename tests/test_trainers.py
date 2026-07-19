"""Trainers domain (D5): the teaching trainers and their hybrid scheduler — ``trainers/*``.

Two trainers share one comparison rule — advise a hesitant agent to SELECT iff its score is
*strictly* above the median score of the assertive agents (no assertive baseline → no advice). The
score is DT-importance from a probe fit on the participated features (``trainers/dt_importance.py``)
or training-partition mutual-information relevance (``trainers/relevance.py``). The hybrid scheduler
(``trainers/hybrid.py``) sequences relevance → DT-importance → withdraw across configurable
boundaries.

The rule is tested once over both score functions. That advice is *train-only* (leakage-safe) is
proven for both advisors in ``test_engine.py`` (D4); run determinism in ``test_invariants.py`` (D8)
— neither is re-asserted here. Agent classification lives in D4.
"""

from __future__ import annotations

import pytest

from config import load_config
from data.loader import load
from data.splitter import make_split
from engine.policy import ACTION_SELECT
from harness.contract import SelectionContext
from probe import DecisionTreeProbe
from rng import SeededRng
from trainers.classify import AgentClassification
from trainers.dt_importance import DTImportanceTrainer, _advise_from_importances
from trainers.hybrid import HybridTeachingScheduler, build_hybrid_scheduler
from trainers.relevance import RelevanceTrainer, _advise_from_relevance

# The two trainers' pure comparison rules share one signature ``(scores, assertive, hesitant)`` and one
# contract, so the rule is verified once over both.
_RULES = (_advise_from_importances, _advise_from_relevance)

# participated == assertive ⊔ hesitant (the COMP-001 invariant the trainers rely on).
_CLASSIFICATION = AgentClassification(
    participated=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    assertive=(0, 1, 2, 3, 4),
    hesitant=(5, 6, 7, 8, 9),
)


def _wire_context() -> SelectionContext:
    """Fully wire a WDBC context from the seed (identical RNG order each call)."""
    config = load_config()
    rng = SeededRng.from_seed(config.seeds[0])
    split = make_split(load(config), config, rng)
    probe = DecisionTreeProbe(split.train, config, rng)
    return SelectionContext(split=split, probe=probe, config=config, rng=rng)


@pytest.fixture()
def context() -> SelectionContext:
    return _wire_context()


# === The shared comparison rule (crafted scores, both trainers) ===================================


def test_rule_fires_strictly_above_the_assertive_median() -> None:
    """A hesitant feature strictly above the assertive median is advised SELECT; one at the median
    or below is not — identically for the DT-importance and relevance rules."""
    scores = [0.1, 0.3, 0.5, 0.8, 0.3, 0.05]  # assertive {0,1,2} → median 0.3
    #   hesitant: 3 (0.8 > 0.3 ✓), 4 (0.3 == median ✗), 5 (0.05 < 0.3 ✗)
    for rule in _RULES:
        assert rule(scores, assertive=[0, 1, 2], hesitant=[3, 4, 5]) == {3: ACTION_SELECT}


def test_rule_requires_assertive_baseline_and_only_advises_hesitant() -> None:
    """No assertive features → no comparison baseline → no advice; and a highly-scored non-hesitant
    feature is never itself advised (only hesitant agents can be)."""
    for rule in _RULES:
        assert rule([0.9, 0.8], assertive=[], hesitant=[0, 1]) == {}
        # Feature 2 scores highest but is not hesitant; feature 1 is hesitant but below the median.
        assert rule([0.9, 0.1, 0.95], assertive=[0], hesitant=[1]) == {}


def test_rule_gives_no_advice_when_there_are_no_hesitant_agents() -> None:
    """With an assertive baseline but no hesitant agents there is nobody to advise — both rules
    return no advice."""
    for rule in _RULES:
        assert rule([0.9, 0.1, 0.5], assertive=[0, 1], hesitant=[]) == {}


# === DT-importance trainer on a real context ======================================================


def test_dt_importance_trainer_applies_rule_over_participated_importances(
    context: SelectionContext,
) -> None:
    """``advise`` equals the rule applied to importances from a probe fit on the participated
    features; advice is a subset of hesitant, all SELECT; and degenerate partitions short-circuit to
    no advice."""
    trainer = DTImportanceTrainer()

    advice = trainer.advise(_CLASSIFICATION, context)
    importances = context.probe.probe(
        _CLASSIFICATION.participated, context.split.validation
    ).feature_importances
    expected = _advise_from_importances(importances, _CLASSIFICATION.assertive, _CLASSIFICATION.hesitant)

    assert advice == expected
    assert set(advice).issubset(set(_CLASSIFICATION.hesitant))
    assert all(action == ACTION_SELECT for action in advice.values())

    assert trainer.advise(AgentClassification((), (), ()), context) == {}
    assert trainer.advise(AgentClassification((0, 1), (0, 1), ()), context) == {}  # no hesitant


# === Relevance trainer on a real context ==========================================================


def test_relevance_trainer_applies_rule_and_caches_the_vector(context: SelectionContext) -> None:
    """``advise`` equals the rule applied to the trainer's training-relevance vector (subset of
    hesitant, all SELECT), and that relevance vector is cached per instance rather than
    recomputed."""
    trainer = RelevanceTrainer()

    advice = trainer.advise(_CLASSIFICATION, context)
    relevance = trainer._relevance_vector(context)
    expected = _advise_from_relevance(relevance, _CLASSIFICATION.assertive, _CLASSIFICATION.hesitant)

    assert advice == expected
    assert set(advice).issubset(set(_CLASSIFICATION.hesitant))
    assert all(action == ACTION_SELECT for action in advice.values())
    assert trainer._relevance_vector(context) is relevance  # cached, not recomputed


# === Hybrid scheduler =============================================================================


class _FakeTrainer:
    """A trainer stand-in returning a tagged advice dict and recording the calls it received."""

    def __init__(self, tag: int) -> None:
        self.tag = tag
        self.calls: list = []

    def advise(self, classification, context):
        self.calls.append((classification, context))
        return {self.tag: 1}


def _scheduler(switch: int = 2, withdraw: int = 5):
    relevance, dt = _FakeTrainer(100), _FakeTrainer(200)
    return HybridTeachingScheduler(relevance, dt, switch, withdraw), relevance, dt


def test_hybrid_switches_at_boundaries_and_delegates() -> None:
    """Relevance over [0, switch), DT-importance over [switch, withdraw), then None; ``advise``
    delegates to the active trainer in each stretch and returns {} once guidance is withdrawn."""
    scheduler, relevance, dt = _scheduler(switch=2, withdraw=5)
    assert [scheduler.active_trainer(step) for step in range(7)] == [
        relevance,
        relevance,
        dt,
        dt,
        dt,
        None,
        None,
    ]

    classification = AgentClassification(participated=(0,), assertive=(0,), hesitant=())
    ctx = object()  # opaque: a fake trainer never inspects it
    assert scheduler.advise(0, classification, ctx) == {100: 1}  # relevance stretch
    assert scheduler.advise(3, classification, ctx) == {200: 1}  # DT-importance stretch
    assert scheduler.advise(5, classification, ctx) == {}  # withdrawn

    assert relevance.calls == [(classification, ctx)]  # each active trainer saw exactly its own step
    assert dt.calls == [(classification, ctx)]


def test_hybrid_degenerate_boundaries_are_well_defined() -> None:
    """Switch=0 skips the relevance stretch; switch==withdraw skips the DT stretch; withdraw=0
    withdraws from the first step (the no-trainer schedule)."""
    skip_relevance, _, dt_a = _scheduler(switch=0, withdraw=3)
    assert [skip_relevance.active_trainer(s) for s in range(4)] == [dt_a, dt_a, dt_a, None]

    skip_dt, rel_b, _ = _scheduler(switch=2, withdraw=2)
    assert [skip_dt.active_trainer(s) for s in range(3)] == [rel_b, rel_b, None]

    no_trainer, _, _ = _scheduler(switch=0, withdraw=0)
    assert all(no_trainer.active_trainer(s) is None for s in range(4))


def test_hybrid_rejects_invalid_boundaries() -> None:
    """Non-monotonic or negative boundaries are rejected (require 0 <= switch <= withdraw)."""
    for switch, withdraw in [(5, 2), (-1, 3), (2, -1)]:
        with pytest.raises(ValueError, match="hybrid boundaries"):
            HybridTeachingScheduler(_FakeTrainer(1), _FakeTrainer(2), switch, withdraw)


def test_build_hybrid_scheduler_wires_real_trainers_from_config() -> None:
    """The builder reads the boundaries from config and binds the real relevance / DT-importance
    trainers."""
    scheduler = build_hybrid_scheduler(load_config({"hybrid_switch_step": 4, "hybrid_withdraw_step": 9}))

    assert isinstance(scheduler.active_trainer(0), RelevanceTrainer)
    assert isinstance(scheduler.active_trainer(4), DTImportanceTrainer)
    assert scheduler.active_trainer(9) is None
