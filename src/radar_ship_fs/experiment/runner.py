"""Configuration-driven stable experiment runner."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np

from radar_ship_fs.experiment.artifact import ArtifactObserver, ArtifactStore, development_fingerprint
from radar_ship_fs.feedback.advisors import build_stable_advisor
from radar_ship_fs.feedback.encoders import build_batch_encoder
from radar_ship_fs.feedback.rewards import build_reward_vector
from radar_ship_fs.rl.checkpoint import CheckpointStore
from radar_ship_fs.rl.environment import SubsetEnvironment
from radar_ship_fs.rl.events import CheckpointSaved, StepCompleted
from radar_ship_fs.rl.q_system import IndependentQSystem
from radar_ship_fs.rl.replay import JointReplayBuffer
from radar_ship_fs.rl.schedule import LinearEpsilonSchedule
from radar_ship_fs.rl.session import SessionSettings, TrainingSession
from radar_ship_fs.rl.trainer import MultiAgentDQNTrainer
from rng import SeededRng
from stage2_cv import build_stage2_cv_context


def _rng_snapshot(rng: SeededRng) -> tuple:
    return copy.deepcopy(rng.numpy.bit_generator.state), rng.python.getstate()


def _rng_from_snapshot(snapshot: tuple, seed: int) -> SeededRng:
    numpy_state, python_state = snapshot
    generator = np.random.default_rng()
    generator.bit_generator.state = copy.deepcopy(numpy_state)
    python_rng = random.Random()
    python_rng.setstate(python_state)
    return SeededRng(seed=seed, numpy=generator, python=python_rng)


class ConsoleObserver:
    def __init__(self, method: str, seed: int, steps: int) -> None:
        self.method = method
        self.seed = seed
        self.every = max(1, steps // 10)

    def on_step(self, event: StepCompleted) -> None:
        metric = event.metrics
        if metric.step % self.every == 0:
            loss = "-" if metric.loss is None else f"{metric.loss:.5f}"
            print(
                f"seed={self.seed} method={self.method:<24} "
                f"step={metric.step:>3} acc={metric.accuracy:.4f} "
                f"best={metric.best_accuracy:.4f} eps={metric.epsilon:.3f} loss={loss}",
                flush=True,
            )

    def on_checkpoint(self, event: CheckpointSaved) -> None:
        print(f"checkpoint step={event.step}: {event.path}", flush=True)


def _method_defaults(method) -> tuple[str, str | None]:
    reward = method.reward
    advisor = method.advisor
    if reward is None:
        reward = "uniform" if method.encoder == "minimal" else "personalized"
    if advisor is None:
        advisor = "none" if method.encoder == "minimal" else "hybrid"
    return reward, advisor


class ExperimentRunner:
    """Expand one immutable spec and run every selected stable method/seed."""

    def __init__(self, spec, *, seed_filter: Iterable[int] = (), method_filter: Iterable[str] = ()) -> None:
        if spec.algorithm_version != "stable_v1":
            raise ValueError("the new ExperimentRunner accepts only algorithm_version='stable_v1'")
        self.spec = spec
        requested_seeds = set(int(value) for value in seed_filter)
        requested_methods = set(method_filter)
        self.seeds = tuple(
            seed for seed in spec.dataset.seeds if not requested_seeds or seed in requested_seeds
        )
        self.methods = tuple(
            method
            for method in spec.enabled_methods
            if not requested_methods or method.name in requested_methods
        )
        missing_seeds = requested_seeds - set(self.seeds)
        missing_methods = requested_methods - {method.name for method in self.methods}
        if missing_seeds or missing_methods:
            raise ValueError(
                f"filters reference values absent from TOML: seeds={sorted(missing_seeds)}, "
                f"methods={sorted(missing_methods)}"
            )
        if not self.seeds or not self.methods:
            raise ValueError("experiment filters produced an empty run matrix")

    def matrix(self) -> list[dict]:
        root = Path(self.spec.output.root)
        return [
            {
                "seed": seed,
                "method": method.name,
                "encoder": method.encoder,
                "output": str(root / f"seed-{seed}" / method.name),
                "config_hash": self.spec.config_hash,
            }
            for seed in self.seeds
            for method in self.methods
        ]

    def dry_run(self) -> list[dict]:
        return self.matrix()

    def _identity(self, *, seed: int, method, data_hash: str) -> dict:
        method_payload = asdict(method)
        method_json = json.dumps(method_payload, sort_keys=True, separators=(",", ":"))
        return {
            "artifact_schema_version": 1,
            "algorithm_version": self.spec.algorithm_version,
            "config_hash": self.spec.config_hash,
            "method": method_payload,
            "method_hash": hashlib.sha256(method_json.encode("utf-8")).hexdigest(),
            "seed": int(seed),
            "development_fingerprint": data_hash,
        }

    def _run_method(self, *, base_context, snapshot: tuple, seed: int, method, resume: bool):
        config = self.spec.irfs_config(method)
        context = base_context._replace(
            config=config,
            rng=_rng_from_snapshot(snapshot, seed),
        )
        data_hash = development_fingerprint(context)
        identity = self._identity(seed=seed, method=method, data_hash=data_hash)
        run_dir = Path(self.spec.output.root) / f"seed-{seed}" / method.name
        artifacts = ArtifactStore(run_dir)
        existing = artifacts.existing_selection(identity, self.spec.training.steps)
        if resume and existing is not None:
            print(f"seed={seed} method={method.name} resume: completed selection", flush=True)
            return existing
        artifacts.write_manifest(spec=self.spec, method=method, seed=seed, context=context, identity=identity)

        reward_name, advisor_name = _method_defaults(method)
        encoder = build_batch_encoder(method.encoder, context)
        reward = build_reward_vector(reward_name)
        advisor = build_stable_advisor(advisor_name, context)
        environment = SubsetEnvironment(context, reward, advisor)
        online = IndependentQSystem.build(
            encoder,
            n_features=context.n_features,
            hidden_sizes=self.spec.training.hidden_layer_sizes,
            activation=self.spec.training.activation,
            rng=context.rng,
        )
        trainer = MultiAgentDQNTrainer(
            online,
            learning_rate=self.spec.training.learning_rate,
            discount=self.spec.training.discount,
            target_sync_interval=self.spec.training.target_sync_interval,
            gradient_clip_norm=self.spec.training.gradient_clip_norm,
        )
        replay = JointReplayBuffer(self.spec.training.replay_capacity)
        epsilon = LinearEpsilonSchedule(
            self.spec.training.epsilon_start,
            self.spec.training.epsilon_end,
            self.spec.training.steps,
            self.spec.training.epsilon_decay_fraction,
        )
        session = TrainingSession(
            context=context,
            environment=environment,
            trainer=trainer,
            replay=replay,
            epsilon=epsilon,
            settings=SessionSettings(
                steps=self.spec.training.steps,
                batch_size=self.spec.training.batch_size,
                warmup_steps=self.spec.training.warmup_steps,
                checkpoint_interval=self.spec.training.checkpoint_interval,
            ),
            checkpoint=CheckpointStore(artifacts.checkpoint_path),
            identity=identity,
            observers=(
                ConsoleObserver(method.name, seed, self.spec.training.steps),
                ArtifactObserver(artifacts),
            ),
        )
        result = session.run(resume=resume)
        payload = artifacts.write_result(result, identity=identity, context=context)
        print(
            f"seed={seed} method={method.name} done features={len(result.selection.selected)} "
            f"best={payload['best_dt_inner_cv_accuracy']:.4f}",
            flush=True,
        )
        return payload

    def run(self, *, resume: bool | None = None) -> list[dict]:
        should_resume = self.spec.output.resume if resume is None else bool(resume)
        ArtifactStore.prepare_root(self.spec.output.root, self.spec)
        results: list[dict] = []
        first_method = self.methods[0]
        base_config = self.spec.irfs_config(first_method)
        for seed in self.seeds:
            context = build_stage2_cv_context(
                base_config,
                seed=seed,
                n_splits=self.spec.dataset.inner_cv_folds,
            )
            expected = self.spec.dataset.expected_clean_features
            if expected is not None and context.n_features != expected:
                raise ValueError(f"expected {expected} cleaned features, got {context.n_features}")
            snapshot = _rng_snapshot(context.rng)
            for method in self.methods:
                results.append(
                    self._run_method(
                        base_context=context,
                        snapshot=snapshot,
                        seed=seed,
                        method=method,
                        resume=should_resume,
                    )
                )
        return results
