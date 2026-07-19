"""Reward domain (D3): the overall reward and its per-agent personalization — ``reward/*``.

Covers ``reward/overall.py`` (scalar reward ``accuracy − β·avg-abs-correlation``, with the correlation
penalty *reused* from the training-based graph per DEC-003, not recomputed on validation) and
``reward/personalize.py`` (Section 3.3 schemes that split the overall reward across agents: importance
``r_i = I_i·overall`` and frequency ``r_i = W_i·overall``, with deselected agents earning exactly zero).

These are the seam-adapter reward functions (``reward/*``); the bare-engine ``engine/reward_overall.py``
``OverallReward`` class is exercised by the reinforced-method runs in ``test_invariants.py`` (D8). The
"reward never touches the held-out test partition" invariant is likewise proven at run level in D8, so it
is not re-asserted here.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from config import load_config
from data.loader import load
from data.splitter import make_split
from harness.contract import SelectionContext
from probe import DecisionTreeProbe
from reward.overall import overall_reward
from reward.personalize import per_agent_reward_vector
from rng import SeededRng
from state.graph import average_pairwise_abs_correlation, build_correlation_graph


@pytest.fixture(scope="module")
def context() -> SelectionContext:
    """A real WDBC selection context: shared split + probe under the single seeded RNG."""
    config = load_config()
    rng = SeededRng.from_seed(config.seeds[0])
    dataset = load(config)
    split = make_split(dataset, config, rng)
    probe = DecisionTreeProbe(split.train, config, rng)
    return SelectionContext(split=split, probe=probe, config=config, rng=rng)


class _StubProbe:
    """Probe stand-in with fixed accuracy (and optional importances), isolating the reward math."""

    def __init__(self, accuracy: float, importances=None) -> None:
        self._accuracy = accuracy
        self._importances = None if importances is None else np.asarray(importances, dtype=float)

    def probe(self, subset, eval_partition):  # noqa: ARG002 - signature parity with the real probe
        if self._importances is None:
            return SimpleNamespace(accuracy=self._accuracy)
        return SimpleNamespace(accuracy=self._accuracy, feature_importances=self._importances)


def _stub_context(train_x, accuracy, *, beta, importances=None, scheme=None) -> SimpleNamespace:
    """A minimal context: correlations come from ``train_x``; accuracy/β/scheme are fixed for a
    worked example (validation aliases train so the stub needs only one array)."""
    split = SimpleNamespace(train=SimpleNamespace(X=train_x), validation=SimpleNamespace(X=train_x))
    return SimpleNamespace(
        split=split,
        probe=_StubProbe(accuracy, importances),
        config=SimpleNamespace(correlation_penalty_weight=beta, reward_scheme=scheme),
        n_features=train_x.shape[1],
    )


# Three features; col1 = 2·col0 (|r01| = 1), col2 deselected in the worked examples.
_TRAIN_X = np.array([[0.0, 0.0, 9.0], [1.0, 2.0, 1.0], [2.0, 4.0, 4.0], [3.0, 6.0, 1.0]])


# === Overall reward (COMP-008) ====================================================================


def test_overall_reward_worked_example() -> None:
    """reward = accuracy − β·avg-abs-correlation on a hand computation, and the penalty uses |r|: two
    perfectly correlated features (r=+1) and two perfectly anti-correlated (r=−1) both give 0.8−0.5·1=0.3."""
    positive = _stub_context(
        np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]), accuracy=0.8, beta=0.5
    )
    negative = _stub_context(
        np.array([[0.0, 0.0], [1.0, -2.0], [2.0, -4.0], [3.0, -6.0]]), accuracy=0.8, beta=0.5
    )

    assert overall_reward([0, 1], positive) == pytest.approx(0.3)  # type: ignore[arg-type]
    assert overall_reward([0, 1], negative) == pytest.approx(0.3)  # |−1|, not −1 # type: ignore[arg-type]


def test_penalty_reuses_training_graph_not_validation(context: SelectionContext) -> None:
    """DEC-003 reuse: the penalty equals the train-based graph average (not a validation
    recompute)."""
    selected = [0, 1, 2, 3]
    beta = context.config.correlation_penalty_weight

    accuracy = context.probe.probe(selected, context.split.validation).accuracy
    train_penalty = average_pairwise_abs_correlation(build_correlation_graph(selected, context))
    assert overall_reward(selected, context) == pytest.approx(accuracy - beta * train_penalty)

    # A validation-based recompute genuinely differs — proving the source is train (REQ-013).
    val_x = context.split.validation.X
    pairs = [(i, j) for i in selected for j in selected if i < j]
    val_penalty = np.mean([abs(np.corrcoef(val_x[:, i], val_x[:, j])[0, 1]) for i, j in pairs])
    assert train_penalty != pytest.approx(val_penalty)


def test_beta_is_configurable(context: SelectionContext) -> None:
    """Raising β strictly lowers the reward — the penalty is genuinely weighted by config β."""
    selected = [0, 1, 2, 3]
    low = overall_reward(selected, context._replace(config=load_config({"correlation_penalty_weight": 0.0})))
    high = overall_reward(selected, context._replace(config=load_config({"correlation_penalty_weight": 2.0})))
    assert low > high


def test_singleton_subset_has_no_penalty(context: SelectionContext) -> None:
    """A one-feature subset has no pairs, so the reward is the bare validation accuracy."""
    accuracy = context.probe.probe([7], context.split.validation).accuracy
    assert overall_reward([7], context) == pytest.approx(accuracy)


# === Per-agent personalization (COMP-009) =========================================================


def test_importance_scheme_worked_example() -> None:
    """r_i = importance·overall for selected, 0 for deselected. overall = 0.9 − 0.5·1.0 = 0.4."""
    ctx = _stub_context(_TRAIN_X, accuracy=0.9, beta=0.5, importances=[0.3, 0.7, 0.0], scheme="dt_importance")

    rewards = per_agent_reward_vector([0, 1], ctx)  # type: ignore[arg-type]
    assert rewards == pytest.approx([0.12, 0.28, 0.0])  # [0.3·0.4, 0.7·0.4, deselected]


def test_importance_rewards_sum_to_overall(context: SelectionContext) -> None:
    """Probe importances sum to 1 over the selected set, so Σ personalized = overall (AC-004), and every
    deselected agent is exactly zero."""
    selected = [0, 5, 10, 15]
    rewards = per_agent_reward_vector(selected, context)

    assert rewards.shape == (context.n_features,)
    assert rewards.sum() == pytest.approx(overall_reward(selected, context))
    for agent in range(context.n_features):
        if agent not in selected:
            assert rewards[agent] == 0.0


def test_frequency_scheme_zeroes_deselected_despite_history() -> None:
    """A deselected feature with a large historical count still earns 0 this step (the a_i^t = 0 case)."""
    ctx = _stub_context(_TRAIN_X, accuracy=0.9, beta=0.5, importances=[0.0, 0.0, 0.0], scheme="frequency")
    counts = [
        3.0,
        1.0,
        6.0,
    ]  # total 10 -> W = [0.3, 0.1, 0.6]; feature 2 selected most but not this step

    rewards = per_agent_reward_vector([0, 1], ctx, selection_counts=counts)  # type: ignore[arg-type]
    assert rewards == pytest.approx([0.12, 0.04, 0.0])  # deselected despite W=0.6


def test_frequency_counts_required_and_zero_history_is_zero() -> None:
    """The frequency scheme fails loudly without historical counts, and yields all-zero rewards
    before any selections have happened (all counts zero → no agent has a share)."""
    ctx = _stub_context(_TRAIN_X, accuracy=0.9, beta=0.5, importances=[0.0, 0.0, 0.0], scheme="frequency")

    with pytest.raises(ValueError, match="selection_counts"):
        per_agent_reward_vector([0, 1], ctx)  # type: ignore[arg-type]

    zero_history = per_agent_reward_vector(
        [0, 1],
        ctx,
        selection_counts=[0.0, 0.0, 0.0],  # type: ignore[arg-type]
    )
    assert zero_history == pytest.approx([0.0, 0.0, 0.0])


def test_schemes_are_selectable_and_differ(context: SelectionContext) -> None:
    """The two schemes are config-selectable and produce different per-agent vectors."""
    selected = [0, 5, 10, 15]
    counts = np.zeros(context.n_features)
    counts[selected] = [1.0, 2.0, 3.0, 4.0]

    importance = per_agent_reward_vector(
        selected, context._replace(config=load_config({"reward_scheme": "dt_importance"}))
    )
    frequency = per_agent_reward_vector(
        selected,
        context._replace(config=load_config({"reward_scheme": "frequency"})),
        selection_counts=counts,
    )
    assert not np.allclose(importance, frequency)


def test_unknown_scheme_raises(context: SelectionContext) -> None:
    """An unsupported reward scheme is rejected rather than silently defaulting."""
    with pytest.raises(ValueError, match="reward_scheme"):
        per_agent_reward_vector(
            [0, 1, 2], context._replace(config=load_config({"reward_scheme": "nonsense"}))
        )
