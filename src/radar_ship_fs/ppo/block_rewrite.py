"""Conditional macro-action PPO used by the block-rewrite accuracy experiment.

The existing PPO implementation stores a scalar action.  This module is kept
separate so an ordered remove/remove/add/add action remains one PPO transition
whose log probability is the sum of its teacher-forced conditional heads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .ppo_graph import FeatureGraph
from .ppo_model import GraphActorCritic


@dataclass(frozen=True)
class MacroObservation:
    selected: np.ndarray
    progress: float
    current_score: float
    episode_best_score: float


@dataclass(frozen=True)
class MacroPPOConfig:
    learning_rate: float = 3e-4
    gamma: float = 1.0
    gae_lambda: float = 1.0
    clip_ratio: float = 0.2
    ppo_epochs: int = 4
    minibatch_size: int = 64
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.05


def macro_action_size(mode: str) -> int:
    if mode == "pair":
        return 4
    if mode == "single":
        return 2
    raise ValueError(f"unknown macro action mode: {mode}")


def action_mask(
    selected: torch.Tensor,
    previous_actions: torch.Tensor,
    head_index: int,
    mode: str,
) -> torch.Tensor:
    """Return the legal feature mask for one conditional pointer head."""
    selected = selected.bool()
    if mode == "pair":
        if head_index == 0:
            return selected
        if head_index == 1:
            mask = selected.clone()
            mask.scatter_(1, previous_actions[:, 0:1], False)
            return mask
        if head_index == 2:
            return ~selected
        if head_index == 3:
            mask = ~selected
            mask.scatter_(1, previous_actions[:, 2:3], False)
            return mask
    elif mode == "single":
        if head_index == 0:
            return selected
        if head_index == 1:
            return ~selected
    raise ValueError(f"invalid {mode} head index: {head_index}")


def apply_macro_action(subset: Sequence[int], action: Sequence[int], n_features: int) -> tuple[int, ...]:
    """Apply an ordered pair or single swap, rejecting every illegal action."""
    chosen = {int(value) for value in subset}
    ordered = tuple(int(value) for value in action)
    if not chosen or len(chosen) != len(tuple(subset)):
        raise ValueError("subset must be non-empty and unique")
    if min(chosen) < 0 or max(chosen) >= n_features:
        raise ValueError("subset feature outside cleaned feature space")
    if len(ordered) == 4:
        remove = ordered[:2]
        add = ordered[2:]
    elif len(ordered) == 2:
        remove = ordered[:1]
        add = ordered[1:]
    else:
        raise ValueError("macro action must contain two or four ordered features")
    if len(set(remove)) != len(remove) or any(value not in chosen for value in remove):
        raise ValueError("remove action is not a without-replacement selection from S")
    if len(set(add)) != len(add) or any(value in chosen for value in add):
        raise ValueError("add action is not a without-replacement selection from the complement")
    result = tuple(sorted((chosen - set(remove)) | set(add)))
    if len(result) != len(chosen):
        raise AssertionError("macro action changed the subset size")
    return result


def best_so_far_reward(previous_best: float, new_score: float) -> tuple[float, float]:
    new_best = max(float(previous_best), float(new_score))
    return 100.0 * (new_best - float(previous_best)), new_best


def signed_increment_reward(current_score: float, new_score: float) -> tuple[float, float]:
    return 100.0 * (float(new_score) - float(current_score)), float(new_score)


class ConditionalMacroActorCritic(nn.Module):
    """Graph encoder plus autoregressive feature-pointer heads."""

    def __init__(
        self,
        graph: FeatureGraph,
        *,
        mode: str,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.action_size = macro_action_size(mode)
        # prior_scale is explicitly zero and the legacy actor is never used.
        self.encoder = GraphActorCritic(graph, hidden_dim=hidden_dim, prior_scale=0.0)
        graph_state_dim = hidden_dim * 3 + 2
        context_dim = graph_state_dim + 2
        self.context = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.Tanh(),
        )
        self.node_projections = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(self.action_size)
        )
        self.query_projections = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim) for _ in range(self.action_size)
        )
        self.pointer_heads = nn.ModuleList(
            nn.Linear(hidden_dim, 1, bias=False) for _ in range(self.action_size)
        )
        self.action_positions = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(self.action_size)
        )
        self.critic = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self._reset_new_parameters()

    @property
    def n_features(self) -> int:
        return self.encoder.n_features

    @property
    def prior_scale(self) -> float:
        return self.encoder.prior_scale

    def _reset_new_parameters(self) -> None:
        modules = [
            self.context,
            self.node_projections,
            self.query_projections,
            self.pointer_heads,
            self.action_positions,
            self.critic,
        ]
        for collection in modules:
            for module in collection.modules():
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, gain=2**0.5)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        for head in self.pointer_heads:
            nn.init.orthogonal_(head.weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def _encoded(
        self,
        selected: torch.Tensor,
        progress: torch.Tensor,
        current_score: torch.Tensor,
        episode_best_score: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, graph_state = self.encoder._encode(selected, progress)
        full_state = torch.cat(
            [
                graph_state,
                current_score.view(-1, 1),
                episode_best_score.view(-1, 1),
            ],
            dim=1,
        )
        return hidden, self.context(full_state), self.critic(full_state).squeeze(-1)

    def evaluate_actions(
        self,
        selected: torch.Tensor,
        progress: torch.Tensor,
        current_score: torch.Tensor,
        episode_best_score: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Teacher-force the exact ordered sequence used during sampling."""
        hidden, base_context, value = self._encoded(selected, progress, current_score, episode_best_score)
        joint_log_probability = torch.zeros(selected.shape[0], device=selected.device)
        entropies: list[torch.Tensor] = []
        action_context = torch.zeros_like(base_context)
        for head_index in range(self.action_size):
            mask = action_mask(selected, actions, head_index, self.mode)
            query = self.query_projections[head_index](base_context + action_context)
            logits = self.pointer_heads[head_index](
                torch.tanh(self.node_projections[head_index](hidden) + query.unsqueeze(1))
            ).squeeze(-1)
            distribution = Categorical(logits=logits.masked_fill(~mask, -1e9))
            chosen = actions[:, head_index]
            joint_log_probability = joint_log_probability + distribution.log_prob(chosen)
            entropies.append(distribution.entropy())
            chosen_hidden = hidden.gather(1, chosen.view(-1, 1, 1).expand(-1, 1, hidden.shape[-1])).squeeze(1)
            action_context = action_context + self.action_positions[head_index](chosen_hidden)
        mean_entropy = torch.stack(entropies, dim=1).mean(dim=1)
        return joint_log_probability, mean_entropy, value

    def sample_actions(
        self,
        selected: torch.Tensor,
        progress: torch.Tensor,
        current_score: torch.Tensor,
        episode_best_score: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, base_context, value = self._encoded(selected, progress, current_score, episode_best_score)
        actions = torch.empty((selected.shape[0], self.action_size), dtype=torch.long, device=selected.device)
        joint_log_probability = torch.zeros(selected.shape[0], device=selected.device)
        entropies: list[torch.Tensor] = []
        action_context = torch.zeros_like(base_context)
        for head_index in range(self.action_size):
            mask = action_mask(selected, actions, head_index, self.mode)
            query = self.query_projections[head_index](base_context + action_context)
            logits = self.pointer_heads[head_index](
                torch.tanh(self.node_projections[head_index](hidden) + query.unsqueeze(1))
            ).squeeze(-1)
            distribution = Categorical(logits=logits.masked_fill(~mask, -1e9))
            chosen = distribution.probs.argmax(dim=1) if deterministic else distribution.sample()
            actions[:, head_index] = chosen
            joint_log_probability = joint_log_probability + distribution.log_prob(chosen)
            entropies.append(distribution.entropy())
            chosen_hidden = hidden.gather(1, chosen.view(-1, 1, 1).expand(-1, 1, hidden.shape[-1])).squeeze(1)
            action_context = action_context + self.action_positions[head_index](chosen_hidden)
        mean_entropy = torch.stack(entropies, dim=1).mean(dim=1)
        return actions, joint_log_probability, value, mean_entropy


@dataclass
class MacroRolloutBuffer:
    selected: list[np.ndarray]
    progress: list[float]
    current_score: list[float]
    episode_best_score: list[float]
    actions: list[np.ndarray]
    log_probabilities: list[float]
    values: list[float]
    rewards: list[float]
    dones: list[bool]

    @classmethod
    def empty(cls) -> "MacroRolloutBuffer":
        return cls([], [], [], [], [], [], [], [], [])

    def add(
        self,
        observation: MacroObservation,
        action: Sequence[int],
        log_probability: float,
        value: float,
        reward: float,
        done: bool,
    ) -> None:
        self.selected.append(np.asarray(observation.selected, dtype=np.bool_).copy())
        self.progress.append(float(observation.progress))
        self.current_score.append(float(observation.current_score))
        self.episode_best_score.append(float(observation.episode_best_score))
        self.actions.append(np.asarray(action, dtype=np.int64).copy())
        self.log_probabilities.append(float(log_probability))
        self.values.append(float(value))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))

    def __len__(self) -> int:
        return len(self.actions)


def generalized_advantages(
    rewards: Sequence[float],
    values: Sequence[float],
    dones: Sequence[bool],
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    rewards_array = np.asarray(rewards, dtype=np.float32)
    values_array = np.asarray(values, dtype=np.float32)
    dones_array = np.asarray(dones, dtype=np.float32)
    advantages = np.zeros_like(rewards_array)
    gae = 0.0
    for index in range(len(rewards_array) - 1, -1, -1):
        if index == len(rewards_array) - 1 or dones_array[index]:
            next_value = 0.0
        else:
            next_value = values_array[index + 1]
        delta = rewards_array[index] + gamma * next_value - values_array[index]
        gae = delta + gamma * gae_lambda * (1.0 - dones_array[index]) * gae
        advantages[index] = gae
    return advantages, advantages + values_array


class MacroPPOAgent:
    def __init__(
        self,
        model: ConditionalMacroActorCritic,
        config: MacroPPOConfig,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate, eps=1e-5)

    def act_many(
        self, observations: Sequence[MacroObservation], *, deterministic: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        selected = torch.from_numpy(np.stack([row.selected for row in observations])).to(self.device)
        progress = torch.tensor(
            [row.progress for row in observations], dtype=torch.float32, device=self.device
        )
        current = torch.tensor(
            [row.current_score for row in observations], dtype=torch.float32, device=self.device
        )
        best = torch.tensor(
            [row.episode_best_score for row in observations],
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            actions, log_probabilities, values, entropies = self.model.sample_actions(
                selected, progress, current, best, deterministic=deterministic
            )
        return (
            actions.cpu().numpy(),
            log_probabilities.cpu().numpy(),
            values.cpu().numpy(),
            entropies.cpu().numpy(),
        )

    def update(self, buffer: MacroRolloutBuffer) -> dict[str, float]:
        advantages_np, returns_np = generalized_advantages(
            buffer.rewards,
            buffer.values,
            buffer.dones,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )
        selected = torch.from_numpy(np.stack(buffer.selected)).to(self.device)
        progress = torch.tensor(buffer.progress, dtype=torch.float32, device=self.device)
        current = torch.tensor(buffer.current_score, dtype=torch.float32, device=self.device)
        best = torch.tensor(buffer.episode_best_score, dtype=torch.float32, device=self.device)
        actions = torch.from_numpy(np.stack(buffer.actions)).to(self.device)
        old_log_probabilities = torch.tensor(
            buffer.log_probabilities, dtype=torch.float32, device=self.device
        )
        returns = torch.from_numpy(returns_np).to(self.device)
        advantages = torch.from_numpy(advantages_np).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        before = torch.cat([parameter.detach().flatten() for parameter in self.model.parameters()])
        metrics: list[tuple[float, float, float, float, float, float]] = []
        stop_early = False
        for _ in range(self.config.ppo_epochs):
            permutation = torch.randperm(len(buffer), device=self.device)
            for indices in permutation.split(self.config.minibatch_size):
                new_log_probabilities, entropy_values, values = self.model.evaluate_actions(
                    selected[indices],
                    progress[indices],
                    current[indices],
                    best[indices],
                    actions[indices],
                )
                log_ratio = new_log_probabilities - old_log_probabilities[indices]
                ratio = log_ratio.exp()
                unclipped = ratio * advantages[indices]
                clipped = (
                    ratio.clamp(1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio)
                    * advantages[indices]
                )
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = 0.5 * (returns[indices] - values).pow(2).mean()
                entropy = entropy_values.mean()
                loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                approximate_kl = (old_log_probabilities[indices] - new_log_probabilities).mean()
                clip_fraction = ((ratio - 1.0).abs() > self.config.clip_ratio).float().mean()
                metrics.append(
                    (
                        float(policy_loss.item()),
                        float(value_loss.item()),
                        float(entropy.item()),
                        float(approximate_kl.item()),
                        float(clip_fraction.item()),
                        float(grad_norm.item()),
                    )
                )
                if float(approximate_kl.item()) > self.config.target_kl:
                    stop_early = True
                    break
            if stop_early:
                break
        after = torch.cat([parameter.detach().flatten() for parameter in self.model.parameters()])
        values_array = np.asarray(metrics, dtype=float)
        return {
            "policy_loss": float(values_array[:, 0].mean()),
            "value_loss": float(values_array[:, 1].mean()),
            "entropy": float(values_array[:, 2].mean()),
            "approximate_kl": float(values_array[:, 3].mean()),
            "clip_fraction": float(values_array[:, 4].mean()),
            "gradient_norm": float(values_array[:, 5].mean()),
            "parameter_delta_l2": float(torch.linalg.vector_norm(after - before).item()),
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            "transitions": float(len(buffer)),
            "epochs_stopped_early": float(stop_early),
        }
