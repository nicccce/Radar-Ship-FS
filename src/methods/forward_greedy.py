"""True forward selection: enumerate every remaining feature at every step."""

from __future__ import annotations

from harness.contract import SelectionContext, SubsetSelection, make_selection
from methods.sizing import configured_target_size


class ForwardGreedySelector:
    """Add the best remaining feature while the shared inner-CV score improves."""

    def select(self, context: SelectionContext) -> SubsetSelection:
        selected: list[int] = []
        remaining = set(range(context.n_features))
        best_accuracy = -1.0
        budget = configured_target_size(context)

        while remaining and len(selected) < budget:
            candidates: list[tuple[float, int]] = []
            for feature in sorted(remaining):
                subset = tuple(sorted((*selected, feature)))
                accuracy = context.probe.probe(
                    subset,
                    context.split.validation,
                ).accuracy
                candidates.append((float(accuracy), feature))

            candidate_accuracy, candidate_feature = max(
                candidates,
                key=lambda item: (item[0], -item[1]),
            )
            if candidate_accuracy <= best_accuracy + 1e-12:
                break
            selected.append(candidate_feature)
            remaining.remove(candidate_feature)
            best_accuracy = candidate_accuracy

        return make_selection(selected)
