"""Independent feature-agent Q heads behind one batch-oriented system."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch
from torch import nn

from engine.policy import N_ACTIONS

if TYPE_CHECKING:
    from harness.contract import SelectionContext
    from radar_ship_fs.feedback.protocols import BatchStateEncoder


_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "logistic": nn.Sigmoid,
}


def _make_head(state_dim: int, hidden_sizes: Sequence[int], activation: str) -> nn.Sequential:
    if activation not in _ACTIVATIONS:
        raise ValueError(f"unsupported Q-head activation {activation!r}")
    layers: list[nn.Module] = []
    width = int(state_dim)
    for hidden in hidden_sizes:
        layers.extend((nn.Linear(width, int(hidden)), _ACTIVATIONS[activation]()))
        width = int(hidden)
    layers.append(nn.Linear(width, N_ACTIONS))
    return nn.Sequential(*layers)


def _seed_module(module: nn.Module, seed: int) -> None:
    """Initialize without permanently changing the process-global Torch RNG."""
    global_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(int(seed) & 0xFFFFFFFF)
        with torch.no_grad():
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_uniform_(layer.weight, a=5**0.5)
                    fan_in = layer.weight.shape[1]
                    bound = 1.0 / (fan_in**0.5) if fan_in else 0.0
                    nn.init.uniform_(layer.bias, -bound, bound)
    finally:
        torch.random.set_rng_state(global_state)


class IndependentQSystem(nn.Module):
    """One independent head per feature, evaluated as a single ``[B,N,2]`` system."""

    def __init__(self, encoder: "BatchStateEncoder", heads: Sequence[nn.Module]) -> None:
        super().__init__()
        self.encoder = encoder
        self.heads = nn.ModuleList(heads)
        if not self.heads:
            raise ValueError("Q system requires at least one feature head")

    @classmethod
    def build(
        cls,
        encoder: "BatchStateEncoder",
        *,
        n_features: int,
        hidden_sizes: Sequence[int],
        activation: str,
        rng,
    ) -> "IndependentQSystem":
        heads: list[nn.Module] = []
        for _feature in range(n_features):
            head = _make_head(encoder.dimension, hidden_sizes, activation)
            _seed_module(head, int(rng.numpy.integers(0, 2**32)))
            heads.append(head)
        return cls(encoder, heads)

    @property
    def n_features(self) -> int:
        return len(self.heads)

    def q_values(
        self,
        subsets: Sequence[tuple[int, ...]],
        context: "SelectionContext",
    ) -> torch.Tensor:
        states = self.encoder.encode_batch(subsets, context)
        if states.ndim != 3 or states.shape[1] != self.n_features:
            raise ValueError("encoder must return [batch, n_features, state_dim]")
        return torch.stack(
            [head(states[:, feature, :]) for feature, head in enumerate(self.heads)],
            dim=1,
        )

    def select_actions(
        self,
        subset: tuple[int, ...],
        context: "SelectionContext",
        *,
        epsilon: float,
        rng,
    ) -> np.ndarray:
        with torch.no_grad():
            greedy = self.q_values([subset], context)[0].argmax(dim=1).cpu().numpy()
        actions = np.asarray(greedy, dtype=np.int64).copy()
        for feature in range(self.n_features):
            if rng.numpy.random() < epsilon:
                actions[feature] = int(rng.numpy.integers(0, N_ACTIONS))
        return actions

    def clone_target(self) -> "IndependentQSystem":
        encoder = copy.deepcopy(self.encoder) if self.encoder.trainable else self.encoder
        target = IndependentQSystem(encoder, copy.deepcopy(list(self.heads)))
        target.eval()
        for parameter in target.parameters():
            parameter.requires_grad_(False)
        return target

    def sync_from(self, online: "IndependentQSystem") -> None:
        self.load_state_dict(online.state_dict())
        self.eval()
