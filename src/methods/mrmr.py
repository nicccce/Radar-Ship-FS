"""MRMR classical selector (COMP-007) — pinned external implementation.

A fixed-size classical baseline that selects the top-half features by minimum redundancy / maximum
relevance, delegating the mRMR computation to a specific, pinned external library (``mrmr-
selection``) rather than reimplementing it. Returned through the common subset contract
(``Selector``), so the orchestrator invokes it with the identical ``select(context) ->
SubsetSelection`` signature used for every method.

Pinned implementation (RISK-004 / ROLLBACK-003): mRMR results vary by library, so the implementation
is fixed to ``mrmr-selection`` and pinned in ``requirements.lock``. Its identity (name + version) is
exposed by :func:`implementation_identity` for the run artifact to record (the artifact wiring
itself is PHASE-005 / COMP-023; this module only exposes the record). If the pinned library cannot
be resolved or produces environment-dependent subsets, ROLLBACK-003 applies (pin an alternate, or
drop mRMR for the run and record the omission).

Determinism (REQ-021 / AC-005): the library scores relevance by the ANOVA F-statistic and redundancy
by Pearson correlation — both deterministic functions of the training data with no RNG — so no draw
from the shared RNG is needed (unlike the stochastic mutual-information baseline). ``n_jobs=1``
keeps the result independent of the host CPU count, so the same pinned library reproduces the same
subset across environments.

Leakage-safe (REQ-010 / DEC-005): relevance and redundancy are computed only on
``context.split.train``; validation and test are never read here. Subset scoring is the
orchestrator's job through the external probe, not this selector's.

Sizing (REQ-006): K is the shared half-feature-count target, reused from TASK-301 rather than re-
derived, so this baseline sizes identically to the other fixed-size baselines.

Satisfies COMP-007 -> REQ-005, REQ-006; AC-005, AC-003 (sizing portion).
"""

from __future__ import annotations

from importlib.metadata import version

import pandas as pd
from mrmr import mrmr_classif

from harness.contract import SelectionContext, SubsetSelection, make_selection
from methods.sizing import target_size

# The pinned mRMR implementation (RISK-004). The distribution name is the pip/lock package
# name; ``mrmr_classif`` is imported from its ``mrmr`` import package.
IMPLEMENTATION_NAME = "mrmr-selection"


def implementation_identity() -> dict[str, str]:
    """Return the pinned mRMR implementation's identity for artifact recording.

    A readable ``{"name", "version"}`` record (RISK-004 / output (c)); the version is read from the
    installed distribution metadata so it always reflects the resolved pin. PHASE-005 (COMP-023)
    writes this into the emitted artifact; this module only exposes it.
    """
    return {"name": IMPLEMENTATION_NAME, "version": version(IMPLEMENTATION_NAME)}


class MRMRSelector:
    """Select the top-half features by mRMR using the pinned external implementation.

    Conforms structurally to :class:`harness.contract.Selector`: a single ``select`` taking the
    shared :class:`SelectionContext` and returning a :class:`SubsetSelection` with no per-step
    series (classical single-shot selection).
    """

    def select(self, context: SelectionContext) -> SubsetSelection:
        """Run mRMR on the training partition; keep the top ``target_size`` features.

        The training features are wrapped in a DataFrame whose column labels are the integer feature
        indices (0..n-1), so the labels the library returns are already the column indices the
        contract expects — no name round-trip. ``mrmr_classif`` selects K =
        ``target_size(n_features)`` features by maximizing relevance and minimizing redundancy; the
        result is returned through the common contract. The selection is deterministic for fixed
        training data (no RNG draw).
        """
        train = context.split.train
        features = pd.DataFrame(train.X, columns=range(context.n_features))
        target = pd.Series(train.y)

        selected = mrmr_classif(
            X=features,
            y=target,
            K=target_size(context.n_features),
            show_progress=False,
            n_jobs=1,
        )
        return make_selection(selected)
