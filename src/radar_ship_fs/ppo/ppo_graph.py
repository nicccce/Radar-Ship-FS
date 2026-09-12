from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier

from .feature_ids import normalized_feature_ids, random_feature_ids

DOMAIN_NAMES = ["Feature"]


@dataclass(frozen=True)
class FeatureGraph:
    signed_correlation: np.ndarray
    absolute_correlation: np.ndarray
    correlation_adjacency: np.ndarray
    dependency_adjacency: np.ndarray
    static_node_features: np.ndarray
    relevance: np.ndarray
    mutual_information: np.ndarray
    feature_ids: np.ndarray
    threshold: float
    edge_count: int
    dependency_edge_count: int

    @property
    def n_features(self) -> int:
        return int(self.signed_correlation.shape[0])

    def redundancy(self, mask: np.ndarray) -> float:
        indices = np.flatnonzero(mask)
        if indices.size < 2:
            return 0.0
        block = self.absolute_correlation[np.ix_(indices, indices)]
        return float(block[np.triu_indices(indices.size, k=1)].mean())


def _safe_abs_correlation(a: np.ndarray, b: np.ndarray) -> float:
    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return 0.0
    return abs(float(np.corrcoef(a, b)[0, 1]))


def _max_scale(values: np.ndarray) -> np.ndarray:
    maximum = float(np.max(values))
    if maximum <= 0.0:
        return np.zeros_like(values, dtype=np.float32)
    return (values / maximum).astype(np.float32)


def _normalize(adjacency: np.ndarray, *, signed: bool) -> np.ndarray:
    degree_source = np.abs(adjacency) if signed else adjacency
    inverse_sqrt = np.power(np.maximum(degree_source.sum(axis=1), 1e-12), -0.5)
    return inverse_sqrt[:, None] * adjacency * inverse_sqrt[None, :]


def _tree_dependency_graph(
    X_development: np.ndarray,
    y_development: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, int]:
    """Undirected parent-child split dependencies from a train-only Decision Tree."""
    tree = DecisionTreeClassifier(random_state=seed)
    tree.fit(X_development, y_development)
    structure = tree.tree_
    adjacency = np.zeros((X_development.shape[1], X_development.shape[1]), dtype=np.float64)
    for node, parent_feature in enumerate(structure.feature):
        if parent_feature < 0:
            continue
        for child in (structure.children_left[node], structure.children_right[node]):
            if child < 0:
                continue
            child_feature = int(structure.feature[child])
            if child_feature >= 0 and child_feature != parent_feature:
                adjacency[parent_feature, child_feature] = 1.0
                adjacency[child_feature, parent_feature] = 1.0
    edge_count = int(np.count_nonzero(np.triu(adjacency, k=1)))
    np.fill_diagonal(adjacency, 1.0)
    return _normalize(adjacency, signed=False).astype(np.float32), edge_count


def build_feature_graph(
    X_development: np.ndarray,
    y_development: np.ndarray,
    domains: np.ndarray,
    *,
    threshold: float,
    seed: int,
    tree_seed: int | None = None,
    feature_id_seed: int = 0,
    include_feature_id_node_feature: bool = True,
) -> FeatureGraph:
    """Build train-only signed-correlation and Decision-Tree dependency channels."""
    signed = np.corrcoef(np.asarray(X_development, dtype=np.float64), rowvar=False)
    signed = np.nan_to_num(signed, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(signed, 1.0)
    absolute = np.abs(signed)

    correlation = np.where(absolute >= threshold, signed, 0.0)
    np.fill_diagonal(correlation, 1.0)
    normalized_correlation = _normalize(correlation, signed=True).astype(np.float32)
    dependency, dependency_edge_count = _tree_dependency_graph(
        X_development,
        y_development,
        seed if tree_seed is None else tree_seed,
    )

    relevance = np.asarray(
        [
            _safe_abs_correlation(X_development[:, index], y_development)
            for index in range(X_development.shape[1])
        ],
        dtype=np.float32,
    )
    mi = mutual_info_classif(X_development, y_development, random_state=seed).astype(np.float32)
    correlation_degree = (correlation != 0.0).sum(axis=1).astype(np.float32) - 1.0
    dependency_degree = (dependency > 0.0).sum(axis=1).astype(np.float32) - 1.0
    denominator = max(1.0, float(X_development.shape[1] - 1))
    domain_one_hot = np.eye(len(DOMAIN_NAMES), dtype=np.float32)[domains]
    feature_ids = random_feature_ids(X_development.shape[1], feature_id_seed)
    static_parts = [
        _max_scale(relevance),
        _max_scale(mi),
        correlation_degree / denominator,
        dependency_degree / denominator,
        domain_one_hot,
    ]
    if include_feature_id_node_feature:
        static_parts.append(normalized_feature_ids(feature_ids))
    static = np.column_stack(static_parts)

    return FeatureGraph(
        signed_correlation=signed.astype(np.float32),
        absolute_correlation=absolute.astype(np.float32),
        correlation_adjacency=normalized_correlation,
        dependency_adjacency=dependency,
        static_node_features=static.astype(np.float32),
        relevance=relevance,
        mutual_information=mi,
        feature_ids=feature_ids,
        threshold=float(threshold),
        edge_count=int(np.count_nonzero(np.triu(correlation, k=1))),
        dependency_edge_count=dependency_edge_count,
    )
