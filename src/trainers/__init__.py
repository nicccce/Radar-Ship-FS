"""Interactive trainer guidance (BLOCK-001) — the advice capability that defines IRFS over MARLFS.

Owns classifying the feature-agents by their per-step actions and steering the hesitant ones toward
comparatively strong features: by training-data relevance (COMP-002), by Decision-Tree importance
versus the assertive median (COMP-003), or under a hybrid schedule that sequences the two trainers
and then withdraws (COMP-004). The classification that every trainer's advice is defined relative to
lives in :mod:`trainers.classify` (COMP-001) and comes first.

These components produce *advised actions for hesitant agents*; applying that advice onto the
engine's actions through the pluggable advice contract is BLOCK-004's job (``methods``), not this
package's.
"""
