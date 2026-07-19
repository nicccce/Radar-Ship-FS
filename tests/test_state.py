"""State domain (D2): the tree-structured state substrate — ``state/graph.py`` +
``state/encoder.py``.

Covers the two halves of the reference Section 3.2 state:

- **Augmented feature graph** (COMP-005/006): signed-Pearson correlation edges on the *training*
  partition (symmetric, unit-diagonal, NaN-safe), plus directed tree edges extracted from the probe
  tree, remapped subset-local→original, de-duplicated, and self-loop-free.
- **Fixed-weight aggregator & pooler** (COMP-007, DEC-002): a fixed-length state vector independent of
  the selected-subset size (the design-bearing invariant, AC-003/REQ-007), with λ (neighbor/global mix)
  and pooling mode read from config, and no trained parameters.

The graph's ``average_pairwise_abs_correlation`` is the accessor the overall-reward penalty reuses
(DEC-003). Build/encode *determinism* is proven once in ``test_invariants.py`` (D8); here we assert
statelessness (no fitted params) but not run-level reproduction.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from config import load_config
from data.loader import load
from data.splitter import make_split
from harness.contract import SelectionContext
from probe import DecisionTreeProbe
from rng import SeededRng
from state.encoder import TreeStructuredStateEncoder, _aggregate, _pool
from state.graph import (
    CorrelationGraph,
    augment_with_tree_edges,
    average_pairwise_abs_correlation,
    build_augmented_graph,
    build_correlation_graph,
)


@pytest.fixture(scope="module")
def context() -> SelectionContext:
    """A real WDBC selection context: shared split + probe under the single seeded RNG."""
    config = load_config()
    rng = SeededRng.from_seed(config.seeds[0])
    dataset = load(config)
    split = make_split(dataset, config, rng)
    probe = DecisionTreeProbe(split.train, config, rng)
    return SelectionContext(split=split, probe=probe, config=config, rng=rng)


def _with_config(context: SelectionContext, **overrides) -> SelectionContext:
    """Return a copy of ``context`` with config overrides applied (probe/split unaffected)."""
    return context._replace(config=load_config(overrides))


# === Correlation graph (COMP-005) =================================================================


def test_correlation_is_signed_pearson_on_train(context: SelectionContext) -> None:
    """Edge weights equal signed Pearson on the training partition — symmetric, unit-diagonal, and
    genuinely signed (WDBC has negatively-correlated pairs, so it is not absolute)."""
    selected = [0, 3, 7, 10]
    graph = build_correlation_graph(selected, context)

    assert graph.nodes == (0, 3, 7, 10)
    assert graph.correlation.shape == (4, 4)
    assert np.allclose(np.diag(graph.correlation), 1.0)
    assert np.allclose(graph.correlation, graph.correlation.T)

    train_x = context.split.train.X
    for i, fi in enumerate(graph.nodes):
        for j, fj in enumerate(graph.nodes):
            if i != j:
                expected = np.corrcoef(train_x[:, fi], train_x[:, fj])[0, 1]
                assert graph.correlation[i, j] == pytest.approx(expected)

    # Sign is preserved (an absolute-value matrix could not go negative).
    full = build_correlation_graph(range(context.n_features), context)
    assert full.correlation.min() < 0.0


def test_correlation_source_is_train_not_validation(context: SelectionContext) -> None:
    """Correlations are drawn from train, never validation (REQ-013 leakage safety); the partitions
    genuinely differ, so matching train is a real discriminator."""
    graph = build_correlation_graph([1, 2, 5], context)

    train_x, val_x = context.split.train.X, context.split.validation.X
    expected_train = np.corrcoef(train_x[:, 1], train_x[:, 2])[0, 1]
    expected_val = np.corrcoef(val_x[:, 1], val_x[:, 2])[0, 1]

    assert graph.correlation[0, 1] == pytest.approx(expected_train)
    assert expected_train != pytest.approx(expected_val)


def test_constant_column_yields_zero_correlation() -> None:
    """A zero-variance feature correlates 0.0 with everything (finite, not NaN)."""
    x = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [4.0, 5.0]])  # col 1 is constant
    ctx = SimpleNamespace(split=SimpleNamespace(train=SimpleNamespace(X=x)))

    graph = build_correlation_graph([0, 1], ctx)  # type: ignore[arg-type]
    assert graph.correlation[0, 1] == 0.0
    assert np.isfinite(graph.correlation).all()


def test_nodes_are_canonical_and_empty_rejected(context: SelectionContext) -> None:
    """Selection order/duplicates collapse to a sorted unique node tuple; empty subset is
    rejected."""
    graph = build_correlation_graph([10, 3, 3, 0], context)
    assert graph.nodes == (0, 3, 10)
    with pytest.raises(ValueError):
        build_correlation_graph([], context)


# === Tree-edge augmentation (COMP-006) ============================================================


def _stub_tree(feature, children_left, children_right) -> SimpleNamespace:
    """Minimal stand-in for a fitted sklearn tree exposing only the ``tree_`` arrays read here."""
    inner = SimpleNamespace(
        feature=np.asarray(feature),
        children_left=np.asarray(children_left),
        children_right=np.asarray(children_right),
        node_count=len(feature),
    )
    return SimpleNamespace(tree_=inner)


def test_tree_edges_remap_dedupe_and_live_in_original_space(context: SelectionContext) -> None:
    """Directed edges remap subset-local→original, collapse duplicates, drop self-loops, and leave
    the correlation block untouched — proven exactly on a hand-stubbed tree and structurally on a
    real one."""
    # Hand-stubbed tree over nodes (3,7,10,15): 0->1 and 0->2 both yield (3,10) [deduped];
    # 1->3 yields (10,3); 1->4 is (10,10) [self-loop drop]; leaf children contribute nothing.
    nodes = (3, 7, 10, 15)
    tree = _stub_tree(
        feature=[0, 2, 2, 0, 2, -2, -2, -2, -2],
        children_left=[1, 3, 5, 7, -1, -1, -1, -1, -1],
        children_right=[2, 4, 6, 8, -1, -1, -1, -1, -1],
    )
    base = CorrelationGraph(nodes=nodes, correlation=np.eye(4), tree_edges=())
    augmented = augment_with_tree_edges(base, tree)
    assert augmented.tree_edges == ((3, 10), (10, 3))
    assert np.array_equal(augmented.correlation, base.correlation)

    # End-to-end on a real probe tree: edges are directed node pairs in original space, no self-loops.
    graph = build_augmented_graph([0, 5, 10, 15, 20, 25], context)
    node_set = set(graph.nodes)
    assert len(graph.tree_edges) == len(set(graph.tree_edges))  # de-duplicated
    for parent, child in graph.tree_edges:
        assert parent in node_set and child in node_set and parent != child


def test_average_pairwise_abs_correlation_matches_manual(context: SelectionContext) -> None:
    """The reward-reuse accessor (DEC-003) equals the mean absolute upper-triangle correlation, and is
    0.0 for a singleton (no pairs)."""
    graph = build_correlation_graph([2, 4, 8, 16], context)
    k = len(graph.nodes)
    manual = np.mean([abs(graph.correlation[i, j]) for i in range(k) for j in range(i + 1, k)])
    assert average_pairwise_abs_correlation(graph) == pytest.approx(float(manual))

    singleton = build_correlation_graph([7], context)
    assert average_pairwise_abs_correlation(singleton) == 0.0


# === Fixed-weight aggregator & pooler (COMP-007) ==================================================


def test_state_dimension_is_fixed_and_subset_size_independent(context: SelectionContext) -> None:
    """The design-bearing invariant (AC-003/REQ-007): a size-2 and a size-9 subset both encode to a
    length-``dimension`` vector, and the encoder carries no fitted state (DEC-002 — no trained
    params)."""
    encoder = TreeStructuredStateEncoder()

    small = encoder.encode([0, 1], context)
    large = encoder.encode([0, 1, 2, 3, 4, 5, 6, 7, 8], context)

    assert small.shape == (encoder.dimension,)
    assert large.shape == (encoder.dimension,)
    assert encoder.dimension == 4  # [mean, std, min, max]
    assert vars(encoder) == {}  # stateless: instance carries no fitted state


def test_config_knobs_are_consumed(context: SelectionContext) -> None:
    """Λ (neighbor/global mix) and the pooling mode are read from config — changing either changes
    the encoded state (REQ-006) — and an unknown pooling mode fails loudly rather than silently
    defaulting."""
    selected = [0, 5, 10, 15, 20]

    all_neighbor = TreeStructuredStateEncoder().encode(
        selected, _with_config(context, neighbor_global_mix=1.0)
    )
    all_global = TreeStructuredStateEncoder().encode(selected, _with_config(context, neighbor_global_mix=0.0))
    assert not np.allclose(all_neighbor, all_global)

    importance_pooled = TreeStructuredStateEncoder().encode(
        selected, _with_config(context, state_pooling="dt_importance")
    )
    average_pooled = TreeStructuredStateEncoder().encode(
        selected, _with_config(context, state_pooling="average")
    )
    assert not np.allclose(importance_pooled, average_pooled)

    with pytest.raises(ValueError, match="state_pooling"):
        TreeStructuredStateEncoder().encode([1, 2, 3], _with_config(context, state_pooling="nonsense"))


def test_aggregate_matches_hand_computation() -> None:
    """``_aggregate`` reproduces the Step-4 fixed-weight convolution on a small graph with one directed
    tree edge: mix·neighbor + (1-mix)·global, global[v] = Σ_w C[w,v] H[w] (DEC-002, no trained params)."""
    correlation = np.array([[1.0, 0.5, -0.2], [0.5, 1.0, 0.3], [-0.2, 0.3, 1.0]])
    graph = CorrelationGraph(nodes=(0, 1, 2), correlation=correlation, tree_edges=((0, 1),))
    H = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 1.0]])  # (k=3, L=2)
    mix = 0.5

    global_term = correlation @ H  # symmetric, so C[w,v]·H[w] summed over w
    neighbor_term = np.zeros_like(H)
    neighbor_term[1] = correlation[0, 1] * H[0]  # nonzero only for v=1 (in-neighbor node 0)
    expected = mix * neighbor_term + (1.0 - mix) * global_term

    assert np.allclose(_aggregate(graph, H, mix), expected)


def test_pool_average_is_node_mean() -> None:
    """``average`` pooling is the per-node mean of ``H'`` (Method 2)."""
    updated = np.array([[2.0, 4.0], [4.0, 8.0]])
    graph = CorrelationGraph(nodes=(0, 1), correlation=np.eye(2), tree_edges=())
    ctx = SimpleNamespace(config=SimpleNamespace(state_pooling="average"))

    pooled = _pool(updated, graph, ctx)  # type: ignore[arg-type]
    assert np.allclose(pooled, [3.0, 6.0])
