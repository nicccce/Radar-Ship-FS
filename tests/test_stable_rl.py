"""Stable joint replay, environment, batch encoders, and Double-DQN trainer."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from config import load_config
from engine.state_minimal import MinimalStateEncoder
from harness.orchestrator import build_run_context
from radar_ship_fs.feedback.encoders import (
    FixedBatchStateEncoder,
    MinimalBatchStateEncoder,
    TrainableGCNBatchStateEncoder,
)
from radar_ship_fs.rl.environment import SubsetEnvironment
from radar_ship_fs.rl.q_system import IndependentQSystem
from radar_ship_fs.rl.replay import JointReplayBuffer
from radar_ship_fs.rl.schedule import LinearEpsilonSchedule
from radar_ship_fs.rl.trainer import MultiAgentDQNTrainer
from radar_ship_fs.rl.transition import JointTransition
from rng import SeededRng
from state.gcn_encoder import TrainableGCNEncoder


class _LengthEncoder:
    dimension = 1
    trainable = False

    def encode_batch(self, subsets, context) -> torch.Tensor:
        values = [[[float(len(subset))] for _feature in range(context.n_features)] for subset in subsets]
        return torch.tensor(values, dtype=torch.float32)


class _Probe:
    def probe(self, subset, partition):  # noqa: ARG002
        return SimpleNamespace(accuracy=float(len(subset) / 4.0))


class _Reward:
    def evaluate(self, subset, context):
        return np.full(context.n_features, len(subset) / 4.0, dtype=np.float32)

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        assert state == {}


def _tiny_context(seed: int = 42):
    return SimpleNamespace(
        n_features=4,
        rng=SeededRng.from_seed(seed),
        config=SimpleNamespace(feature_budget=None),
        probe=_Probe(),
        split=SimpleNamespace(validation=object()),
    )


def _linear_q_system(n_features: int = 2) -> IndependentQSystem:
    heads = [nn.Linear(1, 2, bias=False) for _ in range(n_features)]
    system = IndependentQSystem(_LengthEncoder(), heads)
    with torch.no_grad():
        for head in system.heads:
            head.weight.copy_(torch.tensor([[1.0], [2.0]]))
    return system


def test_replay_is_bounded_and_rejects_non_applied_transition() -> None:
    replay = JointReplayBuffer(capacity=2)

    def transition(value: int, *, applied: bool = True):
        return JointTransition(
            subset=(value,),
            actions=np.array([0, 1]),
            rewards=np.array([0.0, 1.0]),
            next_subset=(value + 1,),
            applied=applied,
            done=False,
        )

    with pytest.raises(ValueError, match="must not enter"):
        replay.add(transition(0, applied=False))
    replay.add(transition(0))
    replay.add(transition(1))
    replay.add(transition(2))
    assert [item.subset for item in replay] == [(1,), (2,)]


def test_environment_records_but_does_not_apply_degenerate_population_vote() -> None:
    context = _tiny_context()
    environment = SubsetEnvironment(context, _Reward())
    result = environment.step(
        step=0,
        committed=(0, 1),
        proposed_actions=np.zeros(context.n_features, dtype=int),
        done=False,
    )
    assert result.transition.applied is False
    assert result.transition.next_subset == (0, 1)
    assert result.proposed_select_count == 0


def test_epsilon_schedule_has_exact_boundaries() -> None:
    schedule = LinearEpsilonSchedule(1.0, 0.05, total_steps=100, decay_fraction=0.7)
    assert schedule.value(0) == 1.0
    assert schedule.value(35) == pytest.approx(0.525)
    assert schedule.value(70) == pytest.approx(0.05)
    assert schedule.value(1000) == pytest.approx(0.05)


def test_double_dqn_uses_online_argmax_and_target_value() -> None:
    context = SimpleNamespace(n_features=2)
    online = _linear_q_system()
    trainer = MultiAgentDQNTrainer(
        online,
        learning_rate=1e-3,
        discount=0.9,
        target_sync_interval=10,
        gradient_clip_norm=10.0,
    )
    with torch.no_grad():
        for head in trainer.target.heads:
            head.weight.copy_(torch.tensor([[10.0], [3.0]]))
    transition = JointTransition(
        subset=(0,),
        actions=np.array([0, 0]),
        rewards=np.array([1.0, 1.0]),
        next_subset=(0, 1),
        applied=True,
        done=False,
    )
    metrics = trainer.update([transition], context)
    # Online chooses action 1 at next-state input 2; target action-1 value is 2*3 = 6.
    assert metrics.target_q_mean == pytest.approx(1.0 + 0.9 * 6.0)
    assert metrics.loss >= 0.0
    assert trainer.update_count == 1


def test_terminal_transition_does_not_bootstrap_and_target_sync_is_periodic() -> None:
    context = SimpleNamespace(n_features=2)
    trainer = MultiAgentDQNTrainer(
        _linear_q_system(),
        learning_rate=1e-3,
        discount=0.9,
        target_sync_interval=2,
        gradient_clip_norm=0.1,
    )
    transition = JointTransition(
        subset=(0,),
        actions=np.array([0, 1]),
        rewards=np.array([0.25, 0.75]),
        next_subset=(0, 1),
        applied=True,
        done=True,
    )
    first = trainer.update([transition], context)
    assert first.target_q_mean == pytest.approx(0.5)
    assert first.target_synced is False
    second = trainer.update([transition], context)
    assert second.target_synced is True
    for online, target in zip(trainer.online.parameters(), trainer.target.parameters()):
        assert torch.equal(online, target)
    squared = sum(float((parameter.grad**2).sum()) for parameter in trainer.online.parameters())
    assert squared**0.5 <= 0.100001


def test_batch_encoders_match_existing_single_subset_paths() -> None:
    context = build_run_context(load_config({"seeds": (42,)}), seed=42)
    subsets = [(0, 1, 2), (3, 4, 5, 6)]

    minimal = MinimalBatchStateEncoder()
    minimal_batch = minimal.encode_batch(subsets, context)
    legacy_minimal = MinimalStateEncoder()
    for batch, subset in enumerate(subsets):
        expected = np.stack(
            [legacy_minimal.encode(feature, subset, context) for feature in range(context.n_features)]
        )
        assert minimal_batch[batch].numpy() == pytest.approx(expected)

    fixed = FixedBatchStateEncoder()
    fixed_batch = fixed.encode_batch(subsets, context)
    for batch, subset in enumerate(subsets):
        assert fixed_batch[batch].numpy() == pytest.approx(
            fixed._encoder.encode_all(subset, context), abs=1e-6
        )


def test_trainable_gcn_has_fresh_online_state_and_frozen_target() -> None:
    context = build_run_context(load_config({"seeds": (42,), "state_encoder": "trained_gcn"}), seed=42)
    encoder = TrainableGCNBatchStateEncoder(4, 1, "relu", random_state=7)
    online = IndependentQSystem.build(
        encoder,
        n_features=context.n_features,
        hidden_sizes=(4,),
        activation="relu",
        rng=context.rng,
    )
    target = online.clone_target()
    subset = (0, 1, 2, 3, 4)
    before_online = encoder.encode_batch([subset], context).detach().clone()
    before_target = target.encoder.encode_batch([subset], context).detach().clone()
    with torch.no_grad():
        encoder.bias.add_(1.0)
    after_online = encoder.encode_batch([subset], context).detach()
    after_target = target.encoder.encode_batch([subset], context).detach()
    assert not torch.equal(before_online, after_online)
    assert torch.equal(before_target, after_target)


def test_stable_gcn_batch_one_matches_existing_encoder_at_same_weights() -> None:
    context = build_run_context(load_config({"seeds": (42,), "state_encoder": "trained_gcn"}), seed=42)
    legacy = TrainableGCNEncoder(4, 1, "relu", random_state=7)
    stable = TrainableGCNBatchStateEncoder(4, 1, "relu", random_state=99)
    with torch.no_grad():
        stable.W.copy_(legacy.W.float())
        stable.bias.copy_(legacy.bias.float())

    subset = (0, 1, 2, 3, 4)
    expected = legacy.encode_all(subset, context).detach().float()
    actual = stable.encode_batch([subset], context)[0].detach()
    assert actual.dtype == torch.float32
    np.testing.assert_allclose(
        actual.numpy(),
        expected.numpy(),
        rtol=1e-5,
        atol=1e-5,
    )


def test_deterministic_terminal_mdp_learns_the_known_better_action() -> None:
    context = SimpleNamespace(n_features=1)
    head = nn.Linear(1, 2, bias=False)
    with torch.no_grad():
        head.weight.zero_()
    online = IndependentQSystem(_LengthEncoder(), [head])
    trainer = MultiAgentDQNTrainer(
        online,
        learning_rate=0.05,
        discount=0.0,
        target_sync_interval=10,
        gradient_clip_norm=10.0,
    )
    deselect = JointTransition(
        subset=(0,),
        actions=np.array([0]),
        rewards=np.array([0.0]),
        next_subset=(0,),
        applied=True,
        done=True,
    )
    select = JointTransition(
        subset=(0,),
        actions=np.array([1]),
        rewards=np.array([1.0]),
        next_subset=(0,),
        applied=True,
        done=True,
    )
    losses = []
    for _ in range(80):
        metrics = trainer.update([deselect, select], context)
        losses.append(metrics.loss)

    with torch.no_grad():
        values = trainer.online.q_values([(0,)], context)[0, 0]
    assert values[1] > values[0]
    assert losses[-1] < losses[0]
