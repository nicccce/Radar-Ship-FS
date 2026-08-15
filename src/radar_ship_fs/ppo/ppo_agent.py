"""Clipped on-policy PPO for the topology-aware selector."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .config import ExperimentConfig
from .ppo_env import Observation
from .ppo_model import GraphActorCritic


@dataclass
class RolloutBuffer:
    selected: list[np.ndarray]
    progress: list[float]
    valid_actions: list[np.ndarray]
    actions: list[int]
    log_probabilities: list[float]
    values: list[float]
    rewards: list[float]
    dones: list[bool]

    @classmethod
    def empty(cls) -> "RolloutBuffer":
        return cls([], [], [], [], [], [], [], [])

    def add(
        self,
        observation: Observation,
        action: int,
        log_probability: float,
        value: float,
        reward: float,
        done: bool,
    ) -> None:
        self.selected.append(observation.selected)
        self.progress.append(observation.progress)
        self.valid_actions.append(observation.valid_actions)
        self.actions.append(int(action))
        self.log_probabilities.append(float(log_probability))
        self.values.append(float(value))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))

    def __len__(self) -> int:
        return len(self.actions)


class PPOAgent:
    def __init__(
        self,
        model: GraphActorCritic,
        config: ExperimentConfig,
        device: torch.device,
    ) -> None:
        self.ppo_model = model.to(device)
        self.config = config
        self.device = device
        self.optimizer = torch.optim.Adam(
            self.ppo_model.parameters(), lr=config.learning_rate, eps=1e-5
        )

    def set_learning_rate(self, fraction_remaining: float) -> None:
        learning_rate = self.config.learning_rate * max(0.05, fraction_remaining)
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate

    def act(
        self, observation: Observation, deterministic: bool = False
    ) -> tuple[int, float, float]:
        selected = torch.from_numpy(observation.selected[None, :]).to(self.device)
        progress = torch.tensor(
            [observation.progress], dtype=torch.float32, device=self.device
        )
        valid = torch.from_numpy(observation.valid_actions[None, :]).to(self.device)
        with torch.no_grad():
            distribution, value = self.ppo_model.distribution(selected, progress, valid)
            action = distribution.probs.argmax(dim=-1) if deterministic else distribution.sample()
            log_probability = distribution.log_prob(action)
        return int(action.item()), float(log_probability.item()), float(value.item())

    def _advantages(self, buffer: RolloutBuffer) -> tuple[np.ndarray, np.ndarray]:
        rewards = np.asarray(buffer.rewards, dtype=np.float32)
        values = np.asarray(buffer.values, dtype=np.float32)
        dones = np.asarray(buffer.dones, dtype=np.float32)
        advantages = np.zeros_like(rewards)
        gae = 0.0
        for index in range(len(rewards) - 1, -1, -1):
            next_value = 0.0 if dones[index] else values[index + 1]
            delta = rewards[index] + self.config.gamma * next_value - values[index]
            gae = (
                delta
                + self.config.gamma
                * self.config.gae_lambda
                * (1.0 - dones[index])
                * gae
            )
            advantages[index] = gae
        return advantages, advantages + values

    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        advantages_np, returns_np = self._advantages(buffer)
        selected = torch.from_numpy(np.stack(buffer.selected)).to(self.device)
        progress = torch.tensor(buffer.progress, dtype=torch.float32, device=self.device)
        valid = torch.from_numpy(np.stack(buffer.valid_actions)).to(self.device)
        actions = torch.tensor(buffer.actions, dtype=torch.long, device=self.device)
        old_log_probabilities = torch.tensor(
            buffer.log_probabilities, dtype=torch.float32, device=self.device
        )
        returns = torch.from_numpy(returns_np).to(self.device)
        advantages = torch.from_numpy(advantages_np).to(self.device)
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )

        metrics: list[tuple[float, float, float, float, float]] = []
        stop_early = False
        for _ in range(self.config.ppo_epochs):
            permutation = torch.randperm(len(buffer), device=self.device)
            for indices in permutation.split(self.config.minibatch_size):
                distribution, values = self.ppo_model.distribution(
                    selected[indices], progress[indices], valid[indices]
                )
                new_log_probabilities = distribution.log_prob(actions[indices])
                log_ratio = new_log_probabilities - old_log_probabilities[indices]
                ratio = log_ratio.exp()
                unclipped = ratio * advantages[indices]
                clipped = ratio.clamp(
                    1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio
                )
                policy_loss = -torch.minimum(
                    unclipped, clipped * advantages[indices]
                ).mean()
                value_loss = 0.5 * (returns[indices] - values).pow(2).mean()
                entropy = distribution.entropy().mean()
                loss = policy_loss + self.config.value_coef * value_loss
                loss -= self.config.entropy_coef * entropy

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.ppo_model.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()

                approximate_kl = (
                    old_log_probabilities[indices] - new_log_probabilities
                ).mean()
                clip_fraction = (
                    (ratio - 1.0).abs() > self.config.clip_ratio
                ).float().mean()
                metrics.append(
                    (
                        float(policy_loss.item()),
                        float(value_loss.item()),
                        float(entropy.item()),
                        float(approximate_kl.item()),
                        float(clip_fraction.item()),
                    )
                )
                if float(approximate_kl.item()) > self.config.target_kl:
                    stop_early = True
                    break
            if stop_early:
                break

        values = np.asarray(metrics, dtype=float)
        return {
            "policy_loss": float(values[:, 0].mean()),
            "value_loss": float(values[:, 1].mean()),
            "entropy": float(values[:, 2].mean()),
            "approximate_kl": float(values[:, 3].mean()),
            "clip_fraction": float(values[:, 4].mean()),
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            "transitions": float(len(buffer)),
        }
