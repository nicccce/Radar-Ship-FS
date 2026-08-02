"""Stable training session: environment interaction, replay, updates, and recovery."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np

from harness.contract import StepRecord, make_selection
from radar_ship_fs.rl.checkpoint import CheckpointStore, capture_rng_state, restore_rng_state
from radar_ship_fs.rl.events import (
    CheckpointSaved,
    EventBus,
    StepCompleted,
    TrainingObserver,
    UpdateCompleted,
)
from radar_ship_fs.selection.archive import SelectionArchive
from radar_ship_fs.selection.types import StableTrainingResult, TrainingMetrics

if TYPE_CHECKING:
    from harness.contract import SelectionContext
    from radar_ship_fs.rl.environment import SubsetEnvironment
    from radar_ship_fs.rl.replay import JointReplayBuffer
    from radar_ship_fs.rl.schedule import LinearEpsilonSchedule
    from radar_ship_fs.rl.trainer import MultiAgentDQNTrainer, UpdateMetrics


@dataclass(frozen=True)
class SessionSettings:
    steps: int
    batch_size: int
    warmup_steps: int
    checkpoint_interval: int


class TrainingSession:
    """Coordinate stable training while keeping every subsystem independently testable."""

    def __init__(
        self,
        *,
        context: "SelectionContext",
        environment: "SubsetEnvironment",
        trainer: "MultiAgentDQNTrainer",
        replay: "JointReplayBuffer",
        epsilon: "LinearEpsilonSchedule",
        settings: SessionSettings,
        checkpoint: CheckpointStore,
        identity: dict,
        observers: Sequence[TrainingObserver] = (),
    ) -> None:
        self.context = context
        self.environment = environment
        self.trainer = trainer
        self.replay = replay
        self.epsilon = epsilon
        self.settings = settings
        self.checkpoint = checkpoint
        self.identity = dict(identity)
        self.events = EventBus(observers)

    def _new_state(self) -> dict:
        initial = self.environment.initial_subset()
        feature_budget = self.context.config.feature_budget
        initial_accuracy = self.environment.score(initial) if feature_budget is not None else None
        archive = SelectionArchive(
            best_subset=initial,
            best_accuracy=float(initial_accuracy) if initial_accuracy is not None else -1.0,
            feature_budget=feature_budget,
        )
        return {
            "next_step": 0,
            "committed": initial,
            "initial_subset": initial,
            "initial_accuracy": initial_accuracy,
            "archive": archive,
            "metrics": [],
            "rejected_transitions": 0,
            "elapsed_before": 0.0,
        }

    def _restore_state(self) -> dict:
        payload = self.checkpoint.load()
        if payload["identity"] != self.identity:
            raise ValueError("checkpoint identity does not match config/data/method/seed")
        self.trainer.load_state_dict(payload["trainer"])
        self.replay.load_state_dict(payload["replay"])
        self.epsilon.load_state_dict(payload["epsilon"])
        self.environment.load_state_dict(payload["environment"])
        restore_rng_state(self.context.rng, payload["rng"])
        metrics = [TrainingMetrics.from_dict(item) for item in payload["metrics"]]
        return {
            "next_step": int(payload["next_step"]),
            "committed": tuple(int(value) for value in payload["committed"]),
            "initial_subset": tuple(int(value) for value in payload["initial_subset"]),
            "initial_accuracy": payload["initial_accuracy"],
            "archive": SelectionArchive.from_state_dict(payload["archive"]),
            "metrics": metrics,
            "rejected_transitions": int(payload["rejected_transitions"]),
            "elapsed_before": float(metrics[-1].elapsed_seconds) if metrics else 0.0,
        }

    def _save_state(self, state: dict) -> None:
        payload = {
            "identity": self.identity,
            "next_step": state["next_step"],
            "committed": state["committed"],
            "initial_subset": state["initial_subset"],
            "initial_accuracy": state["initial_accuracy"],
            "archive": state["archive"].state_dict(),
            "metrics": [metric.as_dict() for metric in state["metrics"]],
            "rejected_transitions": state["rejected_transitions"],
            "trainer": self.trainer.state_dict(),
            "replay": self.replay.state_dict(),
            "epsilon": self.epsilon.state_dict(),
            "environment": self.environment.state_dict(),
            "rng": capture_rng_state(self.context.rng),
        }
        self.checkpoint.save(payload)
        self.events.emit_checkpoint(CheckpointSaved(step=int(state["next_step"]), path=self.checkpoint.path))

    @staticmethod
    def _training_metrics(
        *,
        step: int,
        env_step,
        epsilon: float,
        best_accuracy: float,
        replay_size: int,
        update: "UpdateMetrics | None",
        elapsed: float,
    ) -> TrainingMetrics:
        rewards = env_step.transition.rewards
        return TrainingMetrics(
            step=step + 1,
            subset=env_step.transition.next_subset,
            subset_size=len(env_step.transition.next_subset),
            accuracy=float(env_step.accuracy),
            best_accuracy=float(best_accuracy),
            epsilon=float(epsilon),
            proposed_select_count=env_step.proposed_select_count,
            transition_applied=env_step.transition.applied,
            reward_min=float(np.min(rewards)),
            reward_mean=float(np.mean(rewards)),
            reward_max=float(np.max(rewards)),
            replay_size=replay_size,
            update_performed=update is not None,
            loss=None if update is None else update.loss,
            td_error_mean=None if update is None else update.td_error_mean,
            td_error_max=None if update is None else update.td_error_max,
            q_mean=None if update is None else update.q_mean,
            q_std=None if update is None else update.q_std,
            q_max=None if update is None else update.q_max,
            target_q_mean=None if update is None else update.target_q_mean,
            gradient_norm=None if update is None else update.gradient_norm,
            target_synced=False if update is None else update.target_synced,
            advisor_override_count=env_step.advisor_override_count,
            elapsed_seconds=float(elapsed),
        )

    def run(self, *, resume: bool, stop_after: int | None = None) -> StableTrainingResult:
        state = self._restore_state() if resume and self.checkpoint.exists else self._new_state()
        if state["next_step"] > self.settings.steps:
            raise ValueError("checkpoint step exceeds configured training budget")
        end_step = self.settings.steps if stop_after is None else min(int(stop_after), self.settings.steps)
        if end_step < state["next_step"]:
            raise ValueError("stop_after cannot precede the restored checkpoint step")
        started = time.perf_counter()

        for step in range(state["next_step"], end_step):
            epsilon = self.epsilon.value(step)
            actions = self.trainer.online.select_actions(
                state["committed"], self.context, epsilon=epsilon, rng=self.context.rng
            )
            env_step = self.environment.step(
                step=step,
                committed=state["committed"],
                proposed_actions=actions,
                done=step + 1 == self.settings.steps,
            )
            update = None
            if env_step.transition.applied:
                self.replay.add(env_step.transition)
                if len(self.replay) >= self.settings.warmup_steps:
                    batch = self.replay.sample(self.settings.batch_size, self.context.rng)
                    update = self.trainer.update(batch, self.context)
                    self.events.emit_update(UpdateCompleted(step=step + 1, metrics=update))
            else:
                state["rejected_transitions"] += 1

            state["archive"].consider(env_step.transition.next_subset, env_step.accuracy)
            elapsed = state["elapsed_before"] + time.perf_counter() - started
            metric = self._training_metrics(
                step=step,
                env_step=env_step,
                epsilon=epsilon,
                best_accuracy=state["archive"].best_accuracy,
                replay_size=len(self.replay),
                update=update,
                elapsed=elapsed,
            )
            state["metrics"].append(metric)
            state["committed"] = env_step.transition.next_subset
            state["next_step"] = step + 1
            self.events.emit_step(StepCompleted(metric))

            if state["next_step"] % self.settings.checkpoint_interval == 0:
                self._save_state(state)

        if state["next_step"] and state["next_step"] % self.settings.checkpoint_interval != 0:
            self._save_state(state)

        per_step = tuple(
            StepRecord(subset=metric.subset, accuracy=metric.accuracy) for metric in state["metrics"]
        )
        selection = make_selection(state["archive"].best_subset, per_step=per_step)
        return StableTrainingResult(
            selection=selection,
            metrics=tuple(state["metrics"]),
            initial_subset=state["initial_subset"],
            initial_accuracy=state["initial_accuracy"],
            learner_updates=self.trainer.update_count,
            rejected_transitions=state["rejected_transitions"],
        )
