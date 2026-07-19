"""Fixed-size baseline sizing rule (REQ-006) — the shared half-feature-count helper.

The fixed-size classical baselines select half the input features. This module is the single home of
that rule so the DT-importance recursive eliminator (COMP-006) and the mRMR selector (COMP-007) size
their subsets identically rather than each re-deriving ``n // 2``. Keeping one helper means the
half-count semantics — including the floor behavior on an odd feature count — are defined in exactly
one place (REQ-006, AC-003 sizing portion).

The relevance top-k baseline (COMP-005, PHASE-002) predates this helper and inlines the same
``n_features // 2``; it is left unchanged here (COMPAT-001 — consume PHASE-002 as-is).
"""

from __future__ import annotations


def target_size(n_features: int) -> int:
    """Return the fixed-size subset size for ``n_features``: the floor of half.

    ``n_features // 2`` — e.g. 30 features select 15, and an odd count of 7 selects 3. This is the
    half-feature-count sizing rule (REQ-006) shared by the fixed-size classical baselines.
    """
    return n_features // 2
