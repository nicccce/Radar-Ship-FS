"""One stable Double-DQN optimizer over all online heads and an optional GCN."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch
from torch.nn import functional as F

if TYPE_CHECKING:
    from harness.contract import SelectionContext
    from radar_ship_fs.rl.q_system import IndependentQSystem
    from radar_ship_fs.rl.transition import JointTransition


@dataclass(frozen=True)
class UpdateMetrics:
    loss: float
    td_error_mean: float
    td_error_max: float
    q_mean: float
    q_std: float
    q_max: float
    target_q_mean: float
    gradient_norm: float
    target_synced: bool


class MultiAgentDQNTrainer:
    """Double-DQN learner independent of environment, probe, and artifact concerns."""

    def __init__(
        self,
        online: "IndependentQSystem",
        *,
        learning_rate: float,
        discount: float,
        target_sync_interval: int,
        gradient_clip_norm: float,
    ) -> None:
        self.online = online
        self.target = online.clone_target()
        self.discount = float(discount)
        self.target_sync_interval = int(target_sync_interval)
        self.gradient_clip_norm = float(gradient_clip_norm)
        parameters = [parameter for parameter in online.parameters() if parameter.requires_grad]
        ids = [id(parameter) for parameter in parameters]
        if len(ids) != len(set(ids)):
            raise ValueError("online optimizer parameter list contains duplicates")
        self.optimizer = torch.optim.Adam(parameters, lr=float(learning_rate))
        self.update_count = 0

    def update(
        self,
        batch: Sequence["JointTransition"],
        context: "SelectionContext",
    ) -> UpdateMetrics:
        if not batch:
            raise ValueError("DQN update requires a non-empty batch")
        if any(not transition.applied for transition in batch):
            raise ValueError("stable replay batch contains a rejected transition")
        subsets = [transition.subset for transition in batch]
        next_subsets = [transition.next_subset for transition in batch]
        actions = torch.as_tensor(np.stack([transition.actions for transition in batch]), dtype=torch.int64)
        rewards = torch.as_tensor(np.stack([transition.rewards for transition in batch]), dtype=torch.float32)
        done = torch.as_tensor([transition.done for transition in batch], dtype=torch.float32).reshape(-1, 1)

        q_all = self.online.q_values(subsets, context)
        q_taken = q_all.gather(dim=2, index=actions.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            next_online = self.online.q_values(next_subsets, context)
            next_actions = next_online.argmax(dim=2, keepdim=True)
            next_target = self.target.q_values(next_subsets, context)
            bootstrap = next_target.gather(dim=2, index=next_actions).squeeze(-1)
            targets = rewards + (1.0 - done) * self.discount * bootstrap

        loss = F.smooth_l1_loss(q_taken, targets, reduction="mean")
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite stable DQN loss")
        self.optimizer.zero_grad()
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.gradient_clip_norm)
        if not torch.isfinite(torch.as_tensor(gradient_norm)):
            raise FloatingPointError("non-finite stable DQN gradient norm")
        self.optimizer.step()

        self.update_count += 1
        synced = self.update_count % self.target_sync_interval == 0
        if synced:
            self.target.sync_from(self.online)

        td_error = (targets - q_taken.detach()).abs()
        return UpdateMetrics(
            loss=float(loss.detach()),
            td_error_mean=float(td_error.mean()),
            td_error_max=float(td_error.max()),
            q_mean=float(q_all.detach().mean()),
            q_std=float(q_all.detach().std(unbiased=False)),
            q_max=float(q_all.detach().max()),
            target_q_mean=float(targets.mean()),
            gradient_norm=float(gradient_norm),
            target_synced=synced,
        )

    def state_dict(self) -> dict:
        return {
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "update_count": self.update_count,
        }

    def load_state_dict(self, state: dict) -> None:
        self.online.load_state_dict(state["online"])
        self.target.load_state_dict(state["target"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.update_count = int(state["update_count"])
