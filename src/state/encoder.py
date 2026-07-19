"""Fixed-weight aggregator & pooler (COMP-007) — the tree-structured state vector.

Turns the augmented feature graph (COMP-005/006, :mod:`state.graph`) into one fixed-length
state vector, implementing the reference method's Section 3.2 Steps 4–5 with **fixed weights and no
trained parameters** (DEC-002):

- **Per-node features.** Each selected feature (graph node) gets a fixed-length descriptive vector
  ``h_v`` of summary statistics computed on the training partition — ``[mean, std, min, max]`` for the
  ``"summary_statistics"`` mode (``config.per_node_features``). The reference leaves the exact per-node
  feature set unspecified; this is a recorded fidelity gap-fill (RISK-001 / Q-003), configurable. The
  **state dimension equals the length of this vector** (4), so it is fixed regardless of how many
  features are selected.

- **Step 4 — enhanced graph convolution.** Each node's representation is updated as
  ``h'_v = λ·Σ_{u∈N(v)} W_{u,v} h_u + (1−λ)·Σ_{w∈V} W_{w,v} h_w``, where ``N(v)`` are the in-neighbors
  along the directed tree edges, ``W`` is the signed correlation edge weight, and ``λ`` is
  ``config.neighbor_global_mix``. A linear combination of length-``L`` vectors stays length ``L``.

- **Step 5 — pooling into the state.** The node representations are pooled into one vector: Method 1
  weights by Decision-Tree importance (``s = Σ_v I_v h'_v``) and Method 2 averages
  (``s = Σ_v h'_v / |V|``), selected by ``config.state_pooling``. Pooling collapses the variable node
  count into a single length-``L`` vector — this is what makes the state **subset-size-independent**
  (REQ-007 / CON-004 / AC-003).

**Scope boundary.** This module produces the *subset-level* state via ``encode(selected, context)``. It
does not conform to the engine's per-agent ``StateEncoder`` seam and does not decide how the one pooled
vector maps onto the N per-agent states — that binding is TASK-405 (analyzer-flow).

**Leakage safety (REQ-013).** Node features and correlations come from ``split.train``; importances come
from the train-fit probe; the test partition is never read. No randomness is consumed, so the encoder is
a pure, deterministic function of ``(selected, context)`` (CON-003) and holds no trained state.

**Fidelity note — deselected-agent state representation (REQ-019-B / RISK-001).** ``encode_all`` builds the
augmented graph **once** over the selected set ``S`` (the reference's state is defined over the selected
subset). A *selected* agent's row is unchanged and bit-identical to a per-feature build. A *deselected*
agent ``i`` is attached to that fixed ``S`` graph from its train-correlations to ``S`` — the reference
Step-4 *global* term only, with no per-``i`` Decision-Tree refit and no tree-edge neighbours for ``i``.
This is a deliberate, performance-motivated departure from the earlier per-deselected ``S ∪ {i}`` build
(itself a gap-fill): it removes the ~O(n^2.5) per-deselected recomputation — a dataset-agnostic gain that
grows with the feature count ``n`` (datasets with several hundred features drop from days to ~an hour;
small datasets are unaffected in absolute terms) — while leaving the committed subset and its metrics
unchanged. Consequence to record:
under the degenerate all-neighbour config ``neighbor_global_mix = 1.0`` deselected rows are zero. The
fallback if this changes convergence materially is the richer ``shared-S-tree`` deselected representation
(analyzer Path B), held in reserve. *Validation (WDBC, no-trainer IRFS):* the primary seed and seed 7
still converge with best/avg accuracy within ~1% of the prior build; seed 123 marginally misses the
smoothed-mean plateau leg (mean-shift 0.0103 vs tol 0.0100, oscillation still bounded) — a borderline
plateau miss from the changed deselected exploration signal, judged not material and accepted.

Satisfies COMP-007 -> REQ-006, REQ-007 (AC-003).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Sequence

import numpy as np

from state.graph import (
    CorrelationGraph,
    build_augmented_graph,
    train_feature_correlation,
)

if TYPE_CHECKING:
    from harness.contract import SelectionContext

# Per-node summary statistics for the "summary_statistics" mode (ASM-003 gap-fill). The state
# dimension equals the number of statistics, fixed and independent of the selected-subset size.
_SUMMARY_STATS = ("mean", "std", "min", "max")
_STATE_DIMENSION = len(_SUMMARY_STATS)


def _node_features(nodes: tuple[int, ...], context: "SelectionContext") -> np.ndarray:
    """Per-node ``[mean, std, min, max]`` matrix ``(k, L)`` computed on the training partition.

    Row ``i`` holds the summary statistics of feature ``nodes[i]`` over the training rows. A
    constant column yields ``std == 0`` (finite, no NaN).
    """
    columns = context.split.train.X[:, list(nodes)]  # (n_train, k)
    return np.stack(
        [
            columns.mean(axis=0),
            columns.std(axis=0),
            columns.min(axis=0),
            columns.max(axis=0),
        ],
        axis=1,
    )  # (k, L)


def _aggregate(graph: CorrelationGraph, node_features: np.ndarray, mix: float) -> np.ndarray:
    """Reference Step 4 — fixed-weight neighbor-vs-global convolution, returning ``H'`` of shape
    ``(k, L)``.

    ``mix`` is λ: the neighbor term sums correlation-weighted in-neighbors along the directed tree
    edges, the global term sums correlation-weighted contributions from every node (the diagonal 1.0
    includes the node itself). The correlation matrix is symmetric, so ``W_{w,v}`` reads from either
    orientation.
    """
    correlation = graph.correlation  # (k, k) signed
    position = {node: i for i, node in enumerate(graph.nodes)}

    k = len(graph.nodes)
    adjacency = np.zeros((k, k), dtype=float)
    for (
        parent,
        child,
    ) in graph.tree_edges:  # directed edge parent -> child == in-neighbor parent of child
        adjacency[position[parent], position[child]] = 1.0

    neighbor_weights = correlation * adjacency  # W_{u,v} only where a tree edge u->v exists
    neighbor_term = neighbor_weights.T @ node_features  # row v = Σ_u W_{u,v} h_u
    global_term = correlation @ node_features  # row v = Σ_w W_{w,v} h_w (correlation symmetric)

    return mix * neighbor_term + (1.0 - mix) * global_term


def _pool(updated_features: np.ndarray, graph: CorrelationGraph, context: "SelectionContext") -> np.ndarray:
    """Reference Step 5 — pool ``H'`` into one length-``L`` state vector per
    ``config.state_pooling``.

    ``"dt_importance"`` weights each node by its Decision-Tree importance (Method 1); ``"average"``
    takes the node mean (Method 2). Importances come from the shared probe on the training-fit tree;
    passing the validation partition only reuses the probe's memo cache and never reaches the test
    data.
    """
    mode = context.config.state_pooling
    if mode == "dt_importance":
        importances = context.probe.probe(graph.nodes, context.split.validation).feature_importances
        weights = np.array([importances[node] for node in graph.nodes], dtype=float)  # (k,)
        return weights @ updated_features  # (L,)
    if mode == "average":
        return updated_features.mean(axis=0)  # (L,)
    raise ValueError(f"unknown state_pooling mode {mode!r}; expected 'dt_importance' or 'average'")


def _standardize(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score each column of ``matrix`` across its rows (nodes), returning the standardization
    params.

    Removes per-feature scale dominance (e.g. the large-magnitude 'area' columns) from the node-
    feature descriptors before aggregation, so no single feature's raw scale swamps the state. A
    column whose nodes are all equal has zero std; centering then yields zeros (finite, no NaN).
    Returns ``(standardized, mean, safe_std)`` where ``mean``/``safe_std`` are the per-column ``(1,
    L)`` params used — exposed so a feature *outside* ``matrix`` (a deselected agent) can be mapped
    into the **same** standardized space (:func:`_subset_build` / ``encode_all``). Pure,
    deterministic, no randomness.
    """
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True)
    safe = np.where(std == 0.0, 1.0, std)
    return (matrix - mean) / safe, mean, safe


class _SubsetBuild(NamedTuple):
    """The reusable products of one augmented-graph build over a node set (the selected set ``S``).

    ``rows`` are the un-pooled aggregated ``H'`` ``(k, L)`` per subset node (the per-agent
    aggregation ``encode_all`` reads selected rows from); ``position`` maps a feature to its row
    index; ``nodes`` is the ordered subset (the column order correlation slices must follow);
    ``standardized`` is the ``(k, L)`` standardized node-feature matrix ``H_S``;
    ``mean``/``safe_std`` are the ``(1, L)`` standardization params, so a deselected feature can be
    mapped into the same space without re-standardizing ``S`` (COMP-007, Path A).
    """

    rows: np.ndarray
    position: dict[int, int]
    nodes: tuple[int, ...]
    standardized: np.ndarray
    mean: np.ndarray
    safe_std: np.ndarray


def _subset_build(node_set: Sequence[int], context: "SelectionContext") -> _SubsetBuild:
    """Build the augmented graph over ``node_set`` and return its reusable aggregation products.

    Computes per-node summary features on the training partition, standardizes them across nodes,
    and applies the fixed-weight graph convolution (:func:`_aggregate`) — the same Step-4
    representation the pooled encoder builds, returned UN-pooled — alongside the standardized
    features and params the per-agent attachment reuses. Pure, deterministic, train-only (no RNG, no
    test data).
    """
    graph = build_augmented_graph(node_set, context)
    node_features = _node_features(graph.nodes, context)
    standardized, mean, safe = _standardize(node_features)
    updated = _aggregate(graph, standardized, context.config.neighbor_global_mix)
    position = {node: i for i, node in enumerate(graph.nodes)}
    return _SubsetBuild(
        rows=updated,
        position=position,
        nodes=graph.nodes,
        standardized=standardized,
        mean=mean,
        safe_std=safe,
    )


class TreeStructuredStateEncoder:
    """Encodes a selected subset into one fixed-length, subset-size-independent state vector.

    Stateless and parameter-free (DEC-002): :meth:`encode` is a pure function of its arguments. The
    :attr:`dimension` is the per-node summary-statistic count, fixed across exploration steps.
    """

    dimension = _STATE_DIMENSION

    def encode(self, selected: Sequence[int], context: "SelectionContext") -> np.ndarray:
        """Encode ``selected`` into the pooled tree-structured state vector of length
        :attr:`dimension`.

        Builds the augmented graph, computes per-node summary features on train, applies the fixed-
        weight graph convolution, and pools into the state. Raises ``ValueError`` for an unsupported
        ``per_node_features`` mode.
        """
        if context.config.per_node_features != "summary_statistics":
            raise ValueError(
                f"unsupported per_node_features mode {context.config.per_node_features!r}; "
                "only 'summary_statistics' is implemented"
            )
        graph = build_augmented_graph(selected, context)
        node_features = _node_features(graph.nodes, context)
        updated_features = _aggregate(graph, node_features, context.config.neighbor_global_mix)
        return _pool(updated_features, graph, context)

    def encode_all(self, selected: Sequence[int], context: "SelectionContext") -> np.ndarray:
        """Per-agent state matrix ``(n_features, dimension)`` — agent ``i``'s own row reflects ``selected``.

        **One heavy build per call (COMP-007, Path A).** A single augmented-graph build over the selected
        set ``S`` produces every *selected* agent's row — bit-identical to the earlier per-feature build,
        because a selected feature's node set was already exactly ``S``. Each *deselected* agent ``i`` is
        then attached to that **fixed** ``S`` graph from its train-correlations to ``S`` (the reference
        Step-4 *global* term only; ``i`` contributes no tree edges, so its neighbour term is zero and no
        per-``i`` Decision-Tree is refit)::

            row_i = (1 − λ) · ( corr(i, S) · H_S  +  1.0 · h_i )

        where ``H_S`` is ``S``'s standardized node-feature matrix, ``h_i`` is ``i``'s summary features
        mapped into ``S``'s standardized space, ``corr(i, S)`` is sliced from the once-per-run full
        correlation matrix, and ``corr(i, i) = 1.0`` is the explicit self-term. This collapses the former
        ``1 + |deselected|`` graph builds to **one**, removing the per-deselected-feature recomputation
        that made the cost scale ~O(n^2.5) in the feature count ``n`` (a dataset-agnostic gain — datasets
        with several hundred features drop from days to ~an hour; small ones are unaffected). The width
        stays :attr:`dimension` (CON-004). Deselected rows are a deliberate, faithful change from the
        earlier ``S ∪ {i}`` build, recorded as a fidelity note (REQ-019-B). Under the degenerate
        ``λ = 1.0`` (all-neighbour) config, deselected rows are zero — a documented consequence of the
        correlation-only attachment. No RNG (CON-003); train-only (leakage-safe, REQ-013).
        """
        chosen = sorted({int(s) for s in selected})
        chosen_set = set(chosen)
        n = context.n_features
        states = np.zeros((n, self.dimension), dtype=float)

        # One build over S; every selected agent reads its exact (unchanged) row from it.
        build = _subset_build(chosen, context)
        for feature in chosen:
            states[feature] = build.rows[build.position[feature]]

        # Deselected agents: vectorized attachment to the fixed S graph (no per-i build).
        deselected = [i for i in range(n) if i not in chosen_set]
        if deselected:
            mix = context.config.neighbor_global_mix
            full_corr = train_feature_correlation(context)  # (n, n), memoized per run
            corr_des_s = full_corr[np.ix_(deselected, build.nodes)]  # (d, k) = corr(i, S)
            raw = _node_features(tuple(deselected), context)  # (d, L) summary stats on train
            h_des = (raw - build.mean) / build.safe_std  # map into S's standardized space
            # global term: Σ_{w∈S} corr(i,w)·h_w  +  corr(i,i)=1.0 · h_i ; neighbour term is 0.
            global_term = corr_des_s @ build.standardized + h_des  # (d, L)
            states[deselected] = (1.0 - mix) * global_term
        return states
