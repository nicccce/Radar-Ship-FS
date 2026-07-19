"""Trainable GCN state encoder (COMP-001) — the learned replacement for the fixed aggregator.

Builds the learned counterpart to :mod:`state.encoder` (the fixed-weight
:class:`~state.encoder.TreeStructuredStateEncoder`): a graph convolution over the **same**
augmented feature graph (COMP-005/006, :mod:`state.graph`) whose linear transform is a
**trainable** PyTorch parameter, pooled to a fixed-length per-feature state. This is the central
fidelity gain of the feature — the state representation gains the capacity to adapt — implementing
the reference method's Section 3.2 Steps 4–5 (SRC-R-001) with a learned weight transform faithful
to the GCN family (Q-001 working direction: ``H' = Â · H · W``).

**Reused unchanged (CON-R-003).** The augmented-graph builder (:func:`build_augmented_graph` — signed
correlation + Decision-Tree tree edges), the Decision-Tree probe (importances for pooling), and the
per-node helpers ``_node_features``/``_standardize`` are imported from the fixed encoder and graph
modules and are *not* modified. Only the aggregation weights change from fixed signed-correlation
folding to a learned ``W``.

**Convolution form (Step 4 — learned).** Per build over the selected set ``S`` (``k = |S|`` nodes):

- ``H`` is the ``(k, L)`` standardized per-node summary-statistics matrix (``L = 4``: mean/std/min/max),
  identical to the fixed encoder's input (train partition only).
- ``Â`` is the ``(k, k)`` normalized augmented adjacency: signed correlation combined with the directed
  tree-edge adjacency via ``config.neighbor_global_mix`` (λ), mirroring the fixed encoder's
  neighbour-vs-global split — neighbour term over tree in-edges, global term over all nodes — but here
  ``Â`` only *carries* the graph; the learnable transform is ``W``.
- ``H' = activation(Â · H · W)``, where ``W`` is a learnable ``(L, dimension)`` parameter (plus a
  learnable ``(dimension,)`` bias) and ``activation`` defaults to ReLU. The whole forward is built from
  autograd-connectable tensors, so a downstream loss yields gradients on ``W``/``bias`` (the optimizer
  that consumes them is TASK-005 — **no backward/optimizer here**).

**Pooling (Step 5).** ``dt_importance`` weighting by default (Method 1: ``s = Σ_v I_v h'_v``) honouring
``config.state_pooling`` (``"average"`` = Method 2). The pooled vector has length ``dimension``,
constant across exploration steps and independent of subset size (CON-004 / REQ-007 / AC-003).

**Per-agent state (mirrors ``encode_all``).** :meth:`encode_all` returns the ``(n_features, dimension)``
matrix: each *selected* agent reads its convolved/pooled row from the one build over ``S``; each
*deselected* agent ``i`` attaches to that fixed ``S`` graph through the learned global term only — the
same correlation-only attachment shape :class:`~state.encoder.TreeStructuredStateEncoder` documents
(REQ-019-B) — kept differentiable through ``W``.

**Determinism (CON-R-001).** ``W``/``bias`` are initialized from a per-encoder
``torch.Generator().manual_seed(random_state)`` (``random_state`` drawn once from the single shared RNG
by the selector, TASK-003) **without disturbing the global torch RNG** — so the same seed reproduces
identical initial parameters. CPU device pinned (CPU reference platform; reproducibility requires it).

**Leakage safety (REQ-013 / REQ-004).** Node features and correlations come from ``split.train``;
importances come from the train-fit probe (validation only reuses the probe memo); the test partition is
never read. The weights are static in PHASE-001 (the learning signal is TASK-005).

Satisfies COMP-001 -> REQ-001, REQ-003; partial REQ-004 (train-only inputs), REQ-005 (seeded init)
summed with TASK-005 and verified in TASK-004/TASK-006.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch
from torch import nn

from state.encoder import _node_features, _standardize
from state.graph import (
    CorrelationGraph,
    build_augmented_graph,
    train_feature_correlation,
)

if TYPE_CHECKING:
    from harness.contract import SelectionContext

# Per-node summary-statistics width (mean/std/min/max). The convolution input ``H`` is ``(k, L)``;
# this is the learned transform's input feature dimension, NOT the output state width (which is the
# configurable ``gcn_hidden_dim``). Mirrors :data:`state.encoder._STATE_DIMENSION`.
_INPUT_FEATURE_WIDTH = 4

_ACTIVATIONS = {
    "relu": torch.relu,
    "tanh": torch.tanh,
    "identity": lambda x: x,
}


def _normalized_adjacency(graph: CorrelationGraph, mix: float) -> torch.Tensor:
    """Build the ``(k, k)`` augmented adjacency ``Â`` as a float64 CPU tensor (no grad).

    Mirrors the fixed encoder's Step-4 neighbour-vs-global split: the neighbour term restricts the
    signed-correlation weights to the directed tree in-edges (``W_{u,v}`` only where a tree edge
    ``u -> v`` exists), the global term spans every node (correlation, symmetric). ``Â = λ·Aₙ + (1−λ)·A_g``
    where ``Aₙ`` is applied left-multiplied as ``neighbour_weights.T`` (row ``v`` = Σ_u W_{u,v} h_u, the
    same orientation as :func:`state.encoder._aggregate`) and ``A_g`` is the full correlation
    matrix. Pure graph structure — carries no learnable parameter — so it is a constant in the autograd
    graph; the learnable transform ``W`` is applied separately.
    """
    correlation = graph.correlation  # (k, k) signed, symmetric, 1.0 diagonal
    position = {node: i for i, node in enumerate(graph.nodes)}
    k = len(graph.nodes)

    adjacency = np.zeros((k, k), dtype=float)
    for (
        parent,
        child,
    ) in graph.tree_edges:  # directed parent -> child == in-neighbour parent of child
        adjacency[position[parent], position[child]] = 1.0

    neighbor_weights = correlation * adjacency  # W_{u,v} only on tree edges u->v
    # row v = Σ_u W_{u,v} h_u  (neighbour term, matching _aggregate's neighbor_weights.T @ H)
    a_hat = mix * neighbor_weights.T + (1.0 - mix) * correlation
    return torch.from_numpy(np.ascontiguousarray(a_hat, dtype=np.float64))


class TrainableGCNEncoder(nn.Module):
    """Learned graph-convolution state encoder over the augmented feature graph (COMP-001).

    A single learned weight transform ``W`` (plus bias) applied as ``H' = activation(Â · H · W)`` over the
    reused augmented graph, pooled to a fixed-length per-feature state. Exposes :attr:`dimension` (the
    output state width = ``gcn_hidden_dim``) and the learnable parameters via :meth:`parameters` (PyTorch
    ``nn.Module``), consumed by the selector/registration (TASK-003) and the joint optimizer (TASK-005).

    Constructed from primitives so it stays decoupled from :mod:`config`/:mod:`rng` (the
    selector reads config and draws ``random_state`` from the shared RNG). Weights are CPU-pinned and
    seeded deterministically from a private ``torch.Generator`` — the global torch RNG is never touched.
    Forward is differentiable but holds no optimizer/backward (TASK-005 owns learning).
    """

    def __init__(
        self,
        output_dim: int,
        layers: int,
        activation: str,
        random_state: int,
    ) -> None:
        super().__init__()
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if layers != 1:
            # Q-001 working direction defers depth/width tuning; only the single-layer transform is
            # implemented in PHASE-001 (epic Non-Goal forbids architecture search). Recorded, not silent.
            raise ValueError(
                f"gcn_layers={layers} unsupported; only a single learned layer is implemented "
                "(depth tuning is a deferred Non-Goal, Q-001)"
            )
        if activation not in _ACTIVATIONS:
            raise ValueError(f"unknown activation {activation!r}; expected one of {sorted(_ACTIVATIONS)}")

        self._dimension = int(output_dim)
        self._layers = int(layers)
        self._activation_name = activation
        self._activation = _ACTIVATIONS[activation]
        self._device = torch.device("cpu")

        # Deterministic init from a PRIVATE generator: a given random_state always yields the same
        # W/bias (same-seed reproducibility, CON-R-001) WITHOUT advancing the global torch RNG.
        generator = torch.Generator(device=self._device)
        generator.manual_seed(int(random_state) & 0xFFFFFFFF)

        # float64 throughout: the reused graph/correlation pipeline is double precision, and the fixed
        # encoder's state is float64 — matching dtype keeps parity comparisons (TASK-004) clean.
        weight = torch.empty(
            (_INPUT_FEATURE_WIDTH, self._dimension), dtype=torch.float64, device=self._device
        )
        # Xavier/Glorot-uniform bound, sampled from the private generator (deterministic, off-global).
        bound = float(np.sqrt(6.0 / (_INPUT_FEATURE_WIDTH + self._dimension)))
        weight.uniform_(-bound, bound, generator=generator)
        self.W = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.zeros(self._dimension, dtype=torch.float64, device=self._device))

    @property
    def dimension(self) -> int:
        """Output state width (= ``gcn_hidden_dim``); fixed across steps and subset sizes
        (CON-004)."""
        return self._dimension

    def _convolve(self, graph: CorrelationGraph, context: "SelectionContext") -> torch.Tensor:
        """Step 4 (learned) — ``H' = activation(Â · H · W)`` over ``graph``, returning ``(k, dimension)``.

        ``H`` is the standardized per-node summary matrix (train-only) reused from the fixed encoder's
        helpers; ``Â`` is the normalized augmented adjacency (constant). The result is autograd-connected
        to ``W``/``bias``.
        """
        node_features = _node_features(graph.nodes, context)  # (k, L) train-only summary stats
        standardized, _mean, _safe = _standardize(node_features)  # (k, L)
        h = torch.from_numpy(np.ascontiguousarray(standardized, dtype=np.float64))  # (k, L), no grad
        a_hat = _normalized_adjacency(graph, context.config.neighbor_global_mix)  # (k, k), no grad
        # Â · H · W  +  bias  ->  activation
        pre = a_hat @ h @ self.W + self.bias  # (k, dimension), grad flows through W/bias
        return self._activation(pre)

    def _pool(
        self, updated: torch.Tensor, graph: CorrelationGraph, context: "SelectionContext"
    ) -> torch.Tensor:
        """Step 5 — pool ``H'`` ``(k, dimension)`` into one length-``dimension`` vector per
        ``state_pooling``.

        ``"dt_importance"`` weights each node by its Decision-Tree importance (Method 1, default);
        ``"average"`` takes the node mean (Method 2). Importances come from the shared train-fit
        probe (validation partition only reuses the memo — never the test data). Differentiable
        through ``H'``.
        """
        mode = context.config.state_pooling
        if mode == "dt_importance":
            importances = context.probe.probe(graph.nodes, context.split.validation).feature_importances
            weights = np.array([importances[node] for node in graph.nodes], dtype=np.float64)  # (k,)
            w = torch.from_numpy(np.ascontiguousarray(weights))
            return w @ updated  # (dimension,)
        if mode == "average":
            return updated.mean(dim=0)  # (dimension,)
        raise ValueError(f"unknown state_pooling mode {mode!r}; expected 'dt_importance' or 'average'")

    def encode(self, selected: Sequence[int], context: "SelectionContext") -> torch.Tensor:
        """Encode ``selected`` into one pooled state vector of length :attr:`dimension`
        (differentiable).

        Builds the augmented graph over ``selected``, applies the learned convolution, and pools. Mirrors
        :meth:`state.encoder.TreeStructuredStateEncoder.encode` but with learnable ``W``. Raises
        ``ValueError`` for an unsupported ``per_node_features`` mode.
        """
        if context.config.per_node_features != "summary_statistics":
            raise ValueError(
                f"unsupported per_node_features mode {context.config.per_node_features!r}; "
                "only 'summary_statistics' is implemented"
            )
        graph = build_augmented_graph(selected, context)
        updated = self._convolve(graph, context)
        return self._pool(updated, graph, context)

    def encode_all(self, selected: Sequence[int], context: "SelectionContext") -> torch.Tensor:
        """Per-agent state matrix ``(n_features, dimension)`` — mirrors the fixed encoder's ``encode_all``.

        One learned build over the selected set ``S`` produces every *selected* agent's pooled-equivalent
        row (each selected node reads its convolved row ``H'``); each *deselected* agent ``i`` attaches to
        the fixed ``S`` graph through the **learned global term only** — the correlation-only attachment
        shape the fixed encoder documents (REQ-019-B), kept differentiable through ``W``::

            row_i = activation( (1 − λ) · ( corr(i, S) · H_S  +  1.0 · h_i ) · W  +  bias )

        where ``H_S`` is ``S``'s standardized node-feature matrix, ``h_i`` is ``i``'s summary features
        mapped into ``S``'s standardized space, and ``corr(i, i) = 1.0`` is the explicit self-term (the
        neighbour term is zero — ``i`` contributes no tree edges). Width stays :attr:`dimension` for every
        row (CON-004). Train-only (REQ-013); no RNG consumed at forward time. The whole matrix is
        autograd-connected to ``W``/``bias`` so the joint optimizer (TASK-005) can train through it.
        """
        if context.config.per_node_features != "summary_statistics":
            raise ValueError(
                f"unsupported per_node_features mode {context.config.per_node_features!r}; "
                "only 'summary_statistics' is implemented"
            )
        chosen = sorted({int(s) for s in selected})
        chosen_set = set(chosen)
        n = context.n_features

        graph = build_augmented_graph(chosen, context)
        # Standardize once over S; reuse the params to map deselected features into the same space.
        node_features = _node_features(graph.nodes, context)  # (k, L)
        standardized, mean, safe = _standardize(node_features)  # (k, L), (1, L), (1, L)
        h_s = torch.from_numpy(np.ascontiguousarray(standardized, dtype=np.float64))  # (k, L)
        a_hat = _normalized_adjacency(graph, context.config.neighbor_global_mix)  # (k, k)

        # Selected rows: full learned convolution H' = activation(Â · H_S · W + bias).
        selected_rows = self._activation(a_hat @ h_s @ self.W + self.bias)  # (k, dimension)

        position = {node: i for i, node in enumerate(graph.nodes)}
        rows = [torch.zeros(self._dimension, dtype=torch.float64) for _ in range(n)]
        for feature in chosen:
            rows[feature] = selected_rows[position[feature]]

        # Deselected agents: vectorized learned attachment to the fixed S graph (no per-i build).
        deselected = [i for i in range(n) if i not in chosen_set]
        if deselected:
            mix = context.config.neighbor_global_mix
            full_corr = train_feature_correlation(context)  # (n, n), memoized per run
            corr_des_s = full_corr[np.ix_(deselected, graph.nodes)]  # (d, k) = corr(i, S)
            raw = _node_features(tuple(deselected), context)  # (d, L)
            h_des = (raw - mean) / safe  # map into S's standardized space, (d, L)
            corr_t = torch.from_numpy(np.ascontiguousarray(corr_des_s, dtype=np.float64))  # (d, k)
            h_des_t = torch.from_numpy(np.ascontiguousarray(h_des, dtype=np.float64))  # (d, L)
            # global term: Σ_{w∈S} corr(i,w)·h_w  +  corr(i,i)=1.0 · h_i ; neighbour term is 0.
            global_pre = (1.0 - mix) * (corr_t @ h_s + h_des_t)  # (d, L)
            des_rows = self._activation(global_pre @ self.W + self.bias)  # (d, dimension)
            for idx, feature in enumerate(deselected):
                rows[feature] = des_rows[idx]

        return torch.stack(rows, dim=0)  # (n_features, dimension)
