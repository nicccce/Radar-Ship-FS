"""Topology-aware graph actor-critic for pointer-style feature selection."""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical

from .ppo_graph import FeatureGraph


class GraphActorCritic(nn.Module):
    """Encode the feature graph and score every unselected feature plus STOP."""

    def __init__(
        self,
        graph: FeatureGraph,
        hidden_dim: int = 64,
        prior_scale: float = 2.0,
    ):
        super().__init__()
        static = torch.as_tensor(graph.static_node_features, dtype=torch.float32)
        self.register_buffer("static_node_features", static)
        self.register_buffer(
            "correlation_adjacency",
            torch.as_tensor(graph.correlation_adjacency, dtype=torch.float32),
        )
        self.register_buffer(
            "dependency_adjacency",
            torch.as_tensor(graph.dependency_adjacency, dtype=torch.float32),
        )
        self.register_buffer(
            "absolute_correlation",
            torch.as_tensor(graph.absolute_correlation, dtype=torch.float32),
        )
        mi = torch.as_tensor(graph.mutual_information, dtype=torch.float32)
        order = torch.argsort(mi)
        selection_prior = torch.empty_like(mi)
        selection_prior[order] = torch.linspace(-1.0, 1.0, steps=mi.numel())
        self.register_buffer("selection_prior", selection_prior)
        self.prior_scale = float(prior_scale)

        input_dim = static.shape[1] + 3
        self.self_1 = nn.Linear(input_dim, hidden_dim)
        self.corr_1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.dep_1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.norm_1 = nn.LayerNorm(hidden_dim)

        self.self_2 = nn.Linear(hidden_dim, hidden_dim)
        self.corr_2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dep_2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm_2 = nn.LayerNorm(hidden_dim)

        self.feature_actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        graph_state_dim = hidden_dim * 3 + 2
        self.stop_actor = nn.Sequential(
            nn.Linear(graph_state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(graph_state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self._reset_parameters()

    @property
    def n_features(self) -> int:
        return int(self.static_node_features.shape[0])

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=2**0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.feature_actor[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.stop_actor[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def _node_features(
        self, selected: torch.Tensor, progress: torch.Tensor
    ) -> torch.Tensor:
        batch_size = selected.shape[0]
        static = self.static_node_features.unsqueeze(0).expand(batch_size, -1, -1)
        selected_column = selected.unsqueeze(-1)

        selected_count = selected.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean_correlation = torch.matmul(
            self.absolute_correlation.unsqueeze(0), selected_column
        ).squeeze(-1)
        mean_correlation = mean_correlation / selected_count
        mean_correlation = mean_correlation * (selected.sum(dim=1, keepdim=True) > 0)

        progress_column = progress.view(-1, 1, 1).expand(-1, self.n_features, 1)
        return torch.cat(
            [static, selected_column, mean_correlation.unsqueeze(-1), progress_column],
            dim=-1,
        )

    def _encode(
        self, selected: torch.Tensor, progress: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        selected = selected.float()
        node_features = self._node_features(selected, progress)
        corr_messages = torch.matmul(self.correlation_adjacency, node_features)
        dep_messages = torch.matmul(self.dependency_adjacency, node_features)
        hidden = torch.relu(
            self.norm_1(
                self.self_1(node_features)
                + self.corr_1(corr_messages)
                + self.dep_1(dep_messages)
            )
        )

        corr_messages = torch.matmul(self.correlation_adjacency, hidden)
        dep_messages = torch.matmul(self.dependency_adjacency, hidden)
        update = self.self_2(hidden) + self.corr_2(corr_messages) + self.dep_2(dep_messages)
        hidden = torch.relu(self.norm_2(hidden + update))

        selected_count = selected.sum(dim=1, keepdim=True)
        selected_pool = (hidden * selected.unsqueeze(-1)).sum(dim=1)
        selected_pool = selected_pool / selected_count.clamp_min(1.0)

        available = 1.0 - selected
        available_pool = (hidden * available.unsqueeze(-1)).sum(dim=1)
        available_pool = available_pool / available.sum(dim=1, keepdim=True).clamp_min(1.0)

        mean_pool = hidden.mean(dim=1)
        selected_fraction = selected_count / self.n_features
        graph_state = torch.cat(
            [
                mean_pool,
                selected_pool,
                available_pool,
                selected_fraction,
                progress.view(-1, 1),
            ],
            dim=1,
        )
        return hidden, graph_state

    def forward(
        self,
        selected: torch.Tensor,
        progress: torch.Tensor,
        valid_actions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, graph_state = self._encode(selected, progress)
        feature_logits = self.feature_actor(hidden).squeeze(-1)
        contextual_prior = self.selection_prior.unsqueeze(0)
        contextual_prior = contextual_prior * (1.0 - 2.0 * selected.float())
        feature_logits = feature_logits + self.prior_scale * contextual_prior
        stop_logit = self.stop_actor(graph_state)
        logits = torch.cat([feature_logits, stop_logit], dim=1)
        if valid_actions is not None:
            logits = logits.masked_fill(~valid_actions.bool(), -1e9)
        value = self.critic(graph_state).squeeze(-1)
        return logits, value

    def distribution(
        self,
        selected: torch.Tensor,
        progress: torch.Tensor,
        valid_actions: torch.Tensor | None = None,
    ) -> tuple[Categorical, torch.Tensor]:
        logits, value = self(selected, progress, valid_actions)
        return Categorical(logits=logits), value
