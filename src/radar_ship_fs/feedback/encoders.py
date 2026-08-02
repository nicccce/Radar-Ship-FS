"""Batch state encoders with one canonical stable-engine interface."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch
from torch import nn

from radar_ship_fs.feedback.protocols import BatchStateEncoder
from state.encoder import TreeStructuredStateEncoder, _node_features, _standardize
from state.graph import build_augmented_graph, train_feature_correlation

if TYPE_CHECKING:
    from harness.contract import SelectionContext


def _absolute_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.std() == 0.0 or right.std() == 0.0:
        return 0.0
    return float(abs(np.corrcoef(left, right)[0, 1]))


class MinimalBatchStateEncoder:
    """Vectorized stable implementation of relevance/redundancy state."""

    dimension = 2
    trainable = False

    def __init__(self) -> None:
        self._cache: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}

    def _prepare(self, train) -> tuple[np.ndarray, np.ndarray]:
        key = np.asarray(train.indices).tobytes()
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        n_features = train.X.shape[1]
        relevance = np.array(
            [_absolute_correlation(train.X[:, feature], train.y) for feature in range(n_features)],
            dtype=float,
        )
        correlation = np.zeros((n_features, n_features), dtype=float)
        for left in range(n_features):
            for right in range(left + 1, n_features):
                value = _absolute_correlation(train.X[:, left], train.X[:, right])
                correlation[left, right] = correlation[right, left] = value
        prepared = (relevance, correlation)
        self._cache[key] = prepared
        return prepared

    def encode_batch(
        self,
        subsets: Sequence[tuple[int, ...]],
        context: "SelectionContext",
    ) -> torch.Tensor:
        if not subsets:
            return torch.empty((0, context.n_features, self.dimension), dtype=torch.float32)
        relevance, correlation = self._prepare(context.split.train)
        n_features = context.n_features
        matrices = np.empty((len(subsets), n_features, self.dimension), dtype=np.float32)
        matrices[:, :, 0] = relevance
        feature_ids = np.arange(n_features)
        for batch, subset in enumerate(subsets):
            selected = np.asarray(tuple(sorted({int(value) for value in subset})), dtype=np.int64)
            if selected.size == 0:
                matrices[batch, :, 1] = 0.0
                continue
            redundancy_sum = correlation[:, selected].sum(axis=1)
            selected_mask = np.isin(feature_ids, selected)
            denominator = selected.size - selected_mask.astype(np.int64)
            matrices[batch, :, 1] = np.divide(
                redundancy_sum,
                denominator,
                out=np.zeros(n_features, dtype=np.float64),
                where=denominator > 0,
            )
        return torch.from_numpy(matrices)


class FixedBatchStateEncoder:
    """Batch adapter for the fixed tree/correlation encoder."""

    dimension = TreeStructuredStateEncoder.dimension
    trainable = False

    def __init__(self) -> None:
        self._encoder = TreeStructuredStateEncoder()

    def encode_batch(
        self,
        subsets: Sequence[tuple[int, ...]],
        context: "SelectionContext",
    ) -> torch.Tensor:
        matrices = [self._encoder.encode_all(subset, context) for subset in subsets]
        if not matrices:
            return torch.empty((0, context.n_features, self.dimension), dtype=torch.float32)
        return torch.as_tensor(np.stack(matrices), dtype=torch.float32)


_GCN_ACTIVATIONS = {
    "relu": torch.relu,
    "tanh": torch.tanh,
    "logistic": torch.sigmoid,
}


def _augmented_adjacency(graph, mix: float) -> torch.Tensor:
    """Existing graph semantics, represented once as a float32 constant tensor."""
    correlation = graph.correlation
    position = {node: index for index, node in enumerate(graph.nodes)}
    adjacency = np.zeros_like(correlation, dtype=np.float32)
    for parent, child in graph.tree_edges:
        adjacency[position[parent], position[child]] = 1.0
    neighbour = correlation.astype(np.float32, copy=False) * adjacency
    combined = float(mix) * neighbour.T + (1.0 - float(mix)) * correlation
    return torch.as_tensor(np.ascontiguousarray(combined), dtype=torch.float32)


class TrainableGCNBatchStateEncoder(nn.Module):
    """Float32 trainable GCN with no parameter-dependent forward cache."""

    trainable = True

    def __init__(self, output_dim: int, layers: int, activation: str, random_state: int) -> None:
        super().__init__()
        if output_dim <= 0:
            raise ValueError("GCN output_dim must be positive")
        if layers != 1:
            raise ValueError("stable_v1 currently supports one GCN layer")
        if activation not in _GCN_ACTIVATIONS:
            raise ValueError(f"unsupported GCN activation {activation!r}")
        self.dimension = int(output_dim)
        self._activation_name = activation
        self._activation = _GCN_ACTIVATIONS[activation]

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(random_state) & 0xFFFFFFFF)
        weight = torch.empty((4, self.dimension), dtype=torch.float32)
        bound = float(np.sqrt(6.0 / (4 + self.dimension)))
        weight.uniform_(-bound, bound, generator=generator)
        self.W = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.zeros(self.dimension, dtype=torch.float32))

    def _encode_one(self, subset: tuple[int, ...], context: "SelectionContext") -> torch.Tensor:
        if context.config.per_node_features != "summary_statistics":
            raise ValueError("stable GCN supports only summary_statistics node features")
        chosen = tuple(sorted({int(value) for value in subset}))
        chosen_set = set(chosen)
        graph = build_augmented_graph(chosen, context)
        raw = _node_features(graph.nodes, context)
        standardized, mean, safe = _standardize(raw)
        h_selected = torch.as_tensor(np.ascontiguousarray(standardized), dtype=torch.float32)
        adjacency = _augmented_adjacency(graph, context.config.neighbor_global_mix)
        selected_rows = self._activation(adjacency @ h_selected @ self.W + self.bias)

        rows = [torch.zeros(self.dimension, dtype=torch.float32) for _ in range(context.n_features)]
        positions = {node: index for index, node in enumerate(graph.nodes)}
        for feature in chosen:
            rows[feature] = selected_rows[positions[feature]]

        deselected = [feature for feature in range(context.n_features) if feature not in chosen_set]
        if deselected:
            full_corr = train_feature_correlation(context)
            corr = full_corr[np.ix_(deselected, graph.nodes)]
            raw_deselected = _node_features(tuple(deselected), context)
            standardized_deselected = (raw_deselected - mean) / safe
            corr_tensor = torch.as_tensor(np.ascontiguousarray(corr), dtype=torch.float32)
            deselected_tensor = torch.as_tensor(
                np.ascontiguousarray(standardized_deselected), dtype=torch.float32
            )
            global_pre = (1.0 - float(context.config.neighbor_global_mix)) * (
                corr_tensor @ h_selected + deselected_tensor
            )
            deselected_rows = self._activation(global_pre @ self.W + self.bias)
            for index, feature in enumerate(deselected):
                rows[feature] = deselected_rows[index]
        return torch.stack(rows)

    def encode_batch(
        self,
        subsets: Sequence[tuple[int, ...]],
        context: "SelectionContext",
    ) -> torch.Tensor:
        if not subsets:
            return torch.empty((0, context.n_features, self.dimension), dtype=torch.float32)
        # Reuse identical subsets only within this forward graph. Nothing survives an optimizer step.
        encoded: dict[tuple[int, ...], torch.Tensor] = {}
        rows: list[torch.Tensor] = []
        for subset in subsets:
            key = tuple(int(value) for value in subset)
            if key not in encoded:
                encoded[key] = self._encode_one(key, context)
            rows.append(encoded[key])
        return torch.stack(rows)


def build_batch_encoder(name: str, context: "SelectionContext") -> BatchStateEncoder:
    if name == "minimal":
        return MinimalBatchStateEncoder()
    if name == "fixed":
        return FixedBatchStateEncoder()
    if name == "trained_gcn":
        seed = int(context.rng.numpy.integers(0, 2**32))
        return TrainableGCNBatchStateEncoder(
            output_dim=context.config.gcn_hidden_dim,
            layers=context.config.gcn_layers,
            activation=context.config.activation,
            random_state=seed,
        )
    raise ValueError(f"unknown stable encoder {name!r}")


assert isinstance(MinimalBatchStateEncoder(), BatchStateEncoder)
