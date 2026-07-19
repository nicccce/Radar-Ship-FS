"""Tree-structured state representation (BLOCK-002).

Owns the fixed-length state vector built from a feature-feature correlation graph augmented with
directed Decision-Tree edges, aggregated with fixed weights and pooled into a vector whose length is
independent of the selected-subset size, with no trained parameters (DEC-002). This package is
consumed by the engine's state seam (``engine.seam.StateEncoder``) once the aggregator/pooler
(COMP-007) is bound.

The augmented graph (COMP-005 + COMP-006) lives in :mod:`state.graph`; it is the single source of
intra-subset correlation structure, reused by the Personalized Reward block (BLOCK-003) rather than
recomputed (DEC-003).
"""
