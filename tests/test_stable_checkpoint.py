"""Bit-identical stable session resumption from a joint replay checkpoint."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from config import load_config
from harness.orchestrator import build_run_context
from radar_ship_fs.feedback.advisors import build_stable_advisor
from radar_ship_fs.feedback.encoders import build_batch_encoder
from radar_ship_fs.feedback.rewards import build_reward_vector
from radar_ship_fs.rl.checkpoint import CheckpointStore
from radar_ship_fs.rl.environment import SubsetEnvironment
from radar_ship_fs.rl.q_system import IndependentQSystem
from radar_ship_fs.rl.replay import JointReplayBuffer
from radar_ship_fs.rl.schedule import LinearEpsilonSchedule
from radar_ship_fs.rl.session import SessionSettings, TrainingSession
from radar_ship_fs.rl.trainer import MultiAgentDQNTrainer
from rng import SeededRng


class _Encoder:
    dimension = 2
    trainable = False

    def encode_batch(self, subsets, context):
        rows = []
        for subset in subsets:
            chosen = set(subset)
            rows.append(
                [
                    [float(feature in chosen), float(len(subset)) / context.n_features]
                    for feature in range(context.n_features)
                ]
            )
        return torch.tensor(rows, dtype=torch.float32)


class _Probe:
    def probe(self, subset, partition):  # noqa: ARG002
        score = 0.5 + 0.1 * (0 in subset) - 0.02 * abs(len(subset) - 2)
        return SimpleNamespace(accuracy=float(score))


class _Reward:
    def evaluate(self, subset, context):
        score = context.probe.probe(subset, context.split.validation).accuracy
        return np.full(context.n_features, score, dtype=np.float32)

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        assert state == {}


def _make_session(path: Path) -> TrainingSession:
    rng = SeededRng.from_seed(91)
    context = SimpleNamespace(
        n_features=4,
        rng=rng,
        config=SimpleNamespace(feature_budget=None),
        probe=_Probe(),
        split=SimpleNamespace(validation=object()),
    )
    online = IndependentQSystem.build(
        _Encoder(),
        n_features=4,
        hidden_sizes=(8,),
        activation="relu",
        rng=rng,
    )
    trainer = MultiAgentDQNTrainer(
        online,
        learning_rate=3e-4,
        discount=0.9,
        target_sync_interval=2,
        gradient_clip_norm=10.0,
    )
    return TrainingSession(
        context=context,
        environment=SubsetEnvironment(context, _Reward()),
        trainer=trainer,
        replay=JointReplayBuffer(32),
        epsilon=LinearEpsilonSchedule(1.0, 0.1, 6, 0.5),
        settings=SessionSettings(steps=6, batch_size=2, warmup_steps=2, checkpoint_interval=3),
        checkpoint=CheckpointStore(path),
        identity={"config_hash": "same", "method": "tiny", "seed": 91},
    )


def _without_elapsed(result):
    rows = []
    for metric in result.metrics:
        row = metric.as_dict()
        row.pop("elapsed_seconds")
        rows.append(row)
    return rows


def test_interrupted_resume_matches_continuous_training(tmp_path: Path) -> None:
    continuous = _make_session(tmp_path / "continuous.pt")
    continuous_result = continuous.run(resume=False)

    interrupted = _make_session(tmp_path / "resumed.pt")
    interrupted.run(resume=False, stop_after=3)
    resumed = _make_session(tmp_path / "resumed.pt")
    resumed_result = resumed.run(resume=True)

    assert continuous_result.selection == resumed_result.selection
    assert _without_elapsed(continuous_result) == _without_elapsed(resumed_result)
    assert continuous_result.learner_updates == resumed_result.learner_updates
    for left, right in zip(
        continuous.trainer.online.state_dict().values(),
        resumed.trainer.online.state_dict().values(),
    ):
        assert torch.equal(left, right)
    for left, right in zip(
        continuous.trainer.target.state_dict().values(),
        resumed.trainer.target.state_dict().values(),
    ):
        assert torch.equal(left, right)


@pytest.mark.parametrize(
    ("encoder_name", "reward_name", "advisor_name"),
    [
        ("minimal", "uniform", None),
        ("fixed", "personalized", "hybrid"),
        ("trained_gcn", "personalized", "hybrid"),
    ],
)
def test_each_stable_encoder_completes_a_short_training_run(
    encoder_name: str,
    reward_name: str,
    advisor_name: str | None,
    tmp_path: Path,
) -> None:
    config = load_config(
        {
            "seeds": (42,),
            "exploration_step_budget": 4,
            "mini_batch_size": 2,
            "state_encoder": "fixed" if encoder_name == "minimal" else encoder_name,
            "hybrid_switch_step": 1,
            "hybrid_withdraw_step": 3,
        }
    )
    context = build_run_context(config, seed=42)
    encoder = build_batch_encoder(encoder_name, context)
    online = IndependentQSystem.build(
        encoder,
        n_features=context.n_features,
        hidden_sizes=(8,),
        activation="relu",
        rng=context.rng,
    )
    trainer = MultiAgentDQNTrainer(
        online,
        learning_rate=3e-4,
        discount=0.9,
        target_sync_interval=2,
        gradient_clip_norm=10.0,
    )
    session = TrainingSession(
        context=context,
        environment=SubsetEnvironment(
            context,
            build_reward_vector(reward_name),
            build_stable_advisor(advisor_name, context),
        ),
        trainer=trainer,
        replay=JointReplayBuffer(16),
        epsilon=LinearEpsilonSchedule(1.0, 0.05, 4, 0.75),
        settings=SessionSettings(steps=4, batch_size=2, warmup_steps=2, checkpoint_interval=2),
        checkpoint=CheckpointStore(tmp_path / f"{encoder_name}.pt"),
        identity={"encoder": encoder_name, "seed": 42},
    )

    result = session.run(resume=False)

    assert len(result.metrics) == 4
    assert len(result.selection.selected) > 0
    encoded = trainer.online.encoder.encode_batch([result.selection.selected], context)
    assert encoded.shape == (1, context.n_features, encoder.dimension)
    assert torch.isfinite(encoded).all()
    assert all(torch.isfinite(parameter).all() for parameter in trainer.online.parameters())
