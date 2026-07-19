"""DT-importance recursive eliminator (COMP-006).

A fixed-size classical baseline: starting from the full feature set, repeatedly fit the shared
Decision-Tree probe on the surviving features, drop the single least-important one, and continue
until half the original features remain. This is recursive feature elimination driven by the probe's
per-iteration Decision-Tree importances rather than by a standalone model, so it scores on the same
shared probe every other method composes through (DEC-002, COMPAT-001).

Returned through the common subset contract (``Selector``), so the orchestrator invokes it with the
identical ``select(context) -> SubsetSelection`` signature used for every method.

Leakage-safe (REQ-010 / DEC-005): the per-iteration importances come from the probe's tree, which is
fit on ``context.split.train``; elimination decisions never read validation or test. The probe call
passes ``context.split.validation`` as the (unused) scoring partition to stay on the contract's
leakage-safe scoring partition and to warm the probe cache for the orchestrator's final scoring of
the same subset — the test partition is never released here.

Determinism (REQ-021 / AC-005): the probe's ``random_state`` is fixed for the run, so its
importances are deterministic; ties at the minimum importance are broken by lowest feature index, so
the same seed yields an identical subset.

Elimination granularity is one feature per iteration (canonical RFE). The reference and config leave
no step-size knob (COMP-025 has no such field), and a step of one is the most faithful, fully
deterministic choice; a configurable step size is deferred (see Notes in the TASK-301 work item)
rather than introducing a config-schema change.

Satisfies COMP-006 -> REQ-005, REQ-006; AC-005, AC-003 (sizing portion).
"""

from __future__ import annotations

import numpy as np

from harness.contract import SelectionContext, SubsetSelection, make_selection
from methods.sizing import target_size


class DTImportanceEliminator:
    """Recursively eliminate the least DT-important feature until half remain.

    Conforms structurally to :class:`harness.contract.Selector`: a single ``select`` taking the
    shared :class:`SelectionContext` and returning a :class:`SubsetSelection` with no per-step
    series (classical single-shot selection).
    """

    def select(self, context: SelectionContext) -> SubsetSelection:
        """Drop the lowest-importance surviving feature until ``target_size`` remain.

        Each iteration fits the shared probe on the surviving features and reads their Decision-Tree
        importances; the surviving feature with the smallest importance is removed, ties broken by
        ascending feature index. The loop stops once exactly ``target_size(n_features)`` features
        remain (half, floored), so on a 30-feature dataset 15 features are returned.
        """
        target = target_size(context.n_features)
        # Surviving feature indices, kept in ascending order so that argmin over their
        # importances breaks ties by lowest feature index (deterministic under the seed).
        surviving = list(range(context.n_features))

        while len(surviving) > target:
            result = context.probe.probe(surviving, context.split.validation)
            surviving_importances = result.feature_importances[surviving]
            drop_position = int(np.argmin(surviving_importances))
            del surviving[drop_position]

        return make_selection(surviving)
