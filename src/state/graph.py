"""Augmented feature graph (COMP-005 + COMP-006) — the raw material of the tree-structured state.

Builds the structure the paper's state representation (Section 3.2) is computed over, in two
faithful steps:

- **COMP-005 — correlation graph.** The currently-selected features become the nodes of a fully
  connected graph; the weight on edge ``(u, v)`` is the **signed** Pearson correlation
  ``r_{f_u, f_v}`` between the two features (reference Step 1). Signed — not absolute — because the
  enhanced graph convolution (reference Step 4, built in COMP-007/TASK-402) sums ``W_{u,v} · h_u``,
  where the sign of the correlation carries information. A constant (zero-variance) column yields a
  correlation of ``0.0`` rather than NaN, matching the engine's existing correlation helpers.

- **COMP-006 — tree-edge augmentation.** The shared Decision-Tree probe is fit on the same selected
  subset; its structure is simplified to the feature-dependency relation ``T'`` and each directed
  edge ``f_parent → f_child`` is added to the graph (reference Step 3). Concretely: for every
  internal tree node splitting on feature ``f_parent`` and every internal child node splitting on
  feature ``f_child``, the directed edge ``(f_parent, f_child)`` is recorded — deduplicated, with
  self-loops dropped.

**Index space (correctness-critical).** ``sklearn``'s ``tree_.feature`` is indexed into the columns
the tree was *fit* on — i.e. positions within the selected subset, not original feature ids. The
probe fits on ``train.X[:, nodes]`` (see ``probe``), so tree feature ``j`` is original feature
``nodes[j]``. Every tree edge is remapped through ``nodes`` before being recorded, so the graph speaks
one consistent (original) feature space.

**Leakage safety (REQ-013 / DEC-005).** Correlations are computed only on ``split.train``; the tree
is fit by the probe on ``train`` and only its *structure* is read here (structure is independent of
the partition the probe scores accuracy on). The test partition is never reached — it has no public
attribute on ``Split``. No randomness is consumed: the build is a pure, deterministic function of
``(selected, context)`` (CON-003), and it holds no trained parameters (DEC-002).

**Reuse boundary (DEC-003).** :func:`average_pairwise_abs_correlation` exposes the mean absolute
intra-subset correlation directly from this graph, so the Personalized Reward block (COMP-008 /
TASK-403) reads its correlation penalty from here rather than recomputing it. Note the source
partition is ``train`` (this block's mandate, COMP-005); the provisional PHASE-002 reward computed
its penalty on ``validation`` — reconciling that is TASK-403's concern, not this module's.

Satisfies COMP-005 -> REQ-005, REQ-013; COMP-006 -> REQ-005.
"""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, NamedTuple, Sequence

import numpy as np

if TYPE_CHECKING:
    from sklearn.tree import DecisionTreeClassifier

    from harness.contract import SelectionContext


class CorrelationGraph(NamedTuple):
    """The augmented feature graph consumed by the aggregator (COMP-007) and reward (COMP-008).

    ``nodes`` are the selected feature indices in the **original** feature space, canonicalized to a
    sorted, de-duplicated tuple — so the same subset always yields the same graph regardless of
    selection order (underpinning same-seed reproducibility). ``correlation`` is the symmetric ``(k,
    k)`` matrix of signed Pearson correlations indexed by node *position* (``correlation[i, j]`` is
    the correlation between ``nodes[i]`` and ``nodes[j]``), with ``1.0`` on the diagonal.
    ``tree_edges`` are directed ``(parent_feature, child_feature)`` pairs in the original feature
    space, de-duplicated and self-loop-free; it is empty until tree-edge augmentation runs.
    """

    nodes: tuple[int, ...]
    correlation: np.ndarray
    tree_edges: tuple[tuple[int, int], ...]


def _signed_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Signed Pearson correlation between two 1-D series, ``0.0`` if either is constant.

    A zero-variance column makes the Pearson coefficient undefined (NaN); treating that as ``0.0``
    keeps every edge weight finite, consistent with the engine's existing correlation helpers.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.std() == 0.0 or b.std() == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _canonical_nodes(selected: Sequence[int]) -> tuple[int, ...]:
    """Sorted, de-duplicated tuple of original feature indices; rejects an empty subset."""
    nodes = tuple(sorted({int(s) for s in selected}))
    if not nodes:
        raise ValueError("selected subset must contain at least one feature index")
    return nodes


def build_correlation_graph(selected: Sequence[int], context: "SelectionContext") -> CorrelationGraph:
    """Build the COMP-005 correlation graph over ``selected`` from the training partition.

    Returns a :class:`CorrelationGraph` with the fully connected signed-Pearson correlation matrix
    and an empty ``tree_edges`` (added by :func:`augment_with_tree_edges`). Correlations are drawn
    only from ``context.split.train`` (REQ-013).

    The ``(k, k)`` matrix is **sliced** from the once-per-run full pairwise correlation matrix
    (:func:`subset_pairwise_signed_correlation`) rather than recomputed pairwise on every call. The
    slice ``full[nodes, nodes]`` reproduces the former per-pair build entry-for-entry (each entry is
    the same :func:`_signed_corr` value), and the diagonal is the matrix's own ``1.0`` self-term —
    so the graph is bit-identical to the previous build while the per-step O(k²) correlation
    recompute becomes a memoized lookup (the heavy cost in the per-step tree-structured state).
    """
    nodes = _canonical_nodes(selected)
    full = subset_pairwise_signed_correlation(context)
    # np.ix_ fancy-indexing returns a fresh array (no alias into the cached matrix); the diagonal it
    # carries is the cached 1.0 self-correlation, matching the former ``np.eye`` initialization.
    correlation = full[np.ix_(nodes, nodes)]
    return CorrelationGraph(nodes=nodes, correlation=correlation, tree_edges=())


# Module-level memo for the full train-feature correlation matrix. The training partition is fixed
# for an entire run, so this signed (n, n) matrix is computed once and reused across every step and
# every ``encode_all`` call. Keyed on the train array's object identity, with a weakref guard so a
# recycled ``id`` after garbage collection can never return a stale matrix for a different dataset
# (Reviewer constraint). Pure, train-only, deterministic — no RNG, never touches test data.
_FULL_CORR_CACHE: "dict[int, tuple[weakref.ref, np.ndarray]]" = {}


def train_feature_correlation(context: "SelectionContext") -> np.ndarray:
    """Signed Pearson correlation between every feature pair over the training partition.

    Returns the full ``(n_features, n_features)`` matrix whose ``[a, b]`` entry equals
    :func:`_signed_corr` of feature columns ``a`` and ``b`` on ``context.split.train`` — including the
    ``0.0`` convention when either column is constant (a constant column centers to all-zeros, so its
    row and column of the matmul are exactly ``0.0``). Computed once per training partition via a
    single standardized-column matmul (population z-scores) and memoized on the train array's identity,
    so the per-step ``encode_all`` attachment reads cheap ``corr(i, S)`` slices instead of rebuilding
    correlations. **Off-diagonal entries are the contract**; the diagonal is not consumed by callers
    (``encode_all`` uses an explicit ``corr(i, i) = 1.0`` self-term). Train-only, no randomness
    (REQ-013, CON-003).
    """
    train_x = context.split.train.X
    key = id(train_x)
    cached = _FULL_CORR_CACHE.get(key)
    if cached is not None and cached[0]() is train_x:
        return cached[1]

    x = np.asarray(train_x, dtype=float)
    n_rows = x.shape[0]
    centered = x - x.mean(axis=0, keepdims=True)  # constant columns -> exact zeros
    std = x.std(axis=0, keepdims=True)  # population std; the ratio matches np.corrcoef's r
    z = centered / np.where(std == 0.0, 1.0, std)  # constant columns stay zero (correlate 0.0)
    correlation = (z.T @ z) / n_rows
    _FULL_CORR_CACHE[key] = (weakref.ref(train_x), correlation)
    return correlation


# Module-level memo for the full train-feature correlation matrix built with the **pairwise**
# :func:`_signed_corr` estimator that the per-subset correlation graph historically used. This is a
# deliberate, separate matrix from :func:`train_feature_correlation` (the vectorized matmul above):
# the two estimators are mathematically equal but can differ at the last floating-point bit, and the
# selected-set graph feeds the state's graph convolution → ε-greedy action selection, so reproducing
# the former graph **bit-for-bit** (not just to ~1 ULP) keeps the recorded per-step series and the
# borderline convergence seeds unchanged. The matmul matrix remains the source for the deselected
# attachment in ``encode_all`` (its established behavior), so neither path's numbers move. Same
# weakref-guarded per-run caching as above.
_SUBSET_PAIRWISE_CORR_CACHE: "dict[int, tuple[weakref.ref, np.ndarray]]" = {}


def subset_pairwise_signed_correlation(context: "SelectionContext") -> np.ndarray:
    """Full ``(n, n)`` signed Pearson correlation, built once with the per-pair
    :func:`_signed_corr`.

    Entry ``[a, b]`` is exactly ``_signed_corr(train.X[:, a], train.X[:, b])`` — the same value (and
    the same ``0.0`` constant-column convention) the per-subset graph computed pointwise — with an
    explicit ``1.0`` diagonal self-correlation. Computed once per training partition (the train rows
    are fixed for a run) and memoized on the train array's identity, so
    :func:`build_correlation_graph` slices ``full[nodes, nodes]`` each step instead of recomputing
    ``O(k²)`` correlations. The one-time build is ``O(n²)`` pairwise correlations, amortized across
    every step and every method sharing the run's context. Train-only, no randomness (REQ-013,
    CON-003).
    """
    train_x = context.split.train.X
    key = id(train_x)
    cached = _SUBSET_PAIRWISE_CORR_CACHE.get(key)
    if cached is not None and cached[0]() is train_x:
        return cached[1]

    n = train_x.shape[1]
    correlation = np.eye(n, dtype=float)  # self-correlation 1.0 on the diagonal (matches former build)
    for i in range(n):
        column_i = train_x[:, i]
        for j in range(i + 1, n):
            c = _signed_corr(column_i, train_x[:, j])
            correlation[i, j] = c
            correlation[j, i] = c

    _SUBSET_PAIRWISE_CORR_CACHE[key] = (weakref.ref(train_x), correlation)
    return correlation


def _extract_tree_edges(
    tree: "DecisionTreeClassifier", nodes: tuple[int, ...]
) -> tuple[tuple[int, int], ...]:
    """Directed feature-dependency edges from a tree fit on the columns ``nodes`` (reference Step
    3).

    For each internal node (feature index ``>= 0``) and each of its non-leaf children, records the
    directed edge ``(parent_feature, child_feature)``. Tree feature indices are subset-local, so
    each is remapped to the original space via ``nodes``. Self-loops (a feature splitting both a
    parent and its child) are dropped and duplicate edges collapsed, preserving first-seen order.
    """
    inner = tree.tree_
    feature = inner.feature  # subset-local feature id per node; negative for leaves
    children_left = inner.children_left
    children_right = inner.children_right

    edges: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for node in range(inner.node_count):
        parent_local = feature[node]
        if parent_local < 0:  # leaf — no split feature
            continue
        parent_feature = nodes[parent_local]
        for child in (children_left[node], children_right[node]):
            if child < 0:  # absent child (TREE_LEAF sentinel)
                continue
            child_local = feature[child]
            if child_local < 0:  # child is a leaf — no dependency feature
                continue
            child_feature = nodes[child_local]
            if child_feature == parent_feature:  # drop self-loop
                continue
            edge = (parent_feature, child_feature)
            if edge not in seen:
                seen.add(edge)
                edges.append(edge)

    return tuple(edges)


def augment_with_tree_edges(graph: CorrelationGraph, tree: "DecisionTreeClassifier") -> CorrelationGraph:
    """Return ``graph`` with COMP-006 directed tree edges added.

    ``tree`` must have been fit on the columns ``graph.nodes`` in that order (the invariant
    :func:`build_augmented_graph` upholds), so its subset-local feature indices remap cleanly
    through ``graph.nodes`` to the original feature space.
    """
    return graph._replace(tree_edges=_extract_tree_edges(tree, graph.nodes))


def build_augmented_graph(selected: Sequence[int], context: "SelectionContext") -> CorrelationGraph:
    """Build the full augmented graph (COMP-005 correlation + COMP-006 tree edges) for ``selected``.

    The shared probe supplies the tree: it is fit on ``train`` restricted to the (canonical) nodes,
    so passing ``validation`` as the scoring partition only reuses the probe's memo cache and never
    influences the tree *structure* read here — the build stays leakage-safe (REQ-013).
    """
    graph = build_correlation_graph(selected, context)
    tree = context.probe.probe(graph.nodes, context.split.validation).tree
    return augment_with_tree_edges(graph, tree)


def average_pairwise_abs_correlation(graph: CorrelationGraph) -> float:
    """Mean absolute correlation over every unordered node pair (DEC-003 reuse for COMP-008).

    Returns ``0.0`` for fewer than two nodes (no pairs), matching the provisional overall reward's
    penalty semantics so the reward can read it from here instead of recomputing.
    """
    k = len(graph.nodes)
    if k < 2:
        return 0.0
    upper = np.triu_indices(k, k=1)
    return float(np.mean(np.abs(graph.correlation[upper])))
