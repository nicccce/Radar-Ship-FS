"""L1 / LASSO classical selector (COMP-008).

The only variable-size classical baseline: fit an L1-penalized model on the training partition and
keep exactly the features whose coefficients are driven to non-zero. L1 regularization zeroes out
uninformative coefficients, so the surviving set is the selected subset — its size depends on the
penalty strength, not on the half-feature-count rule the fixed-size baselines use. Returned through
the common subset contract (``Selector``).

Model: ``LogisticRegression`` with ``penalty="l1"`` (the classification-task analog of
LASSO) and the ``liblinear`` solver, whose coordinate descent gives an exact-sparse, deterministic
primal-L1 fit, wrapped in ``OneVsRestClassifier`` so the baseline supports multiclass labels
(REQ-001): liblinear refuses ``n_classes >= 3`` directly, so each class gets its own binary L1 fit
and a feature survives if it is non-zero for any class. For a binary label this is a single fit
identical to the bare estimator (WDBC is unchanged). The inverse- regularization strength ``C`` is
read from config (COMP-025, ``l1_C``); a smaller ``C`` is a stronger penalty and yields fewer
surviving features.

Feature standardization: features are standardized (``StandardScaler``) before the fit. L1 selection
is scale-sensitive — on WDBC, raw feature magnitudes span orders of magnitude (e.g. "area" ~10^3 vs
"smoothness" ~10^-2), which both biases which coefficients survive and prevents the solver from
converging. The scaler is fit on the training partition only (leakage-safe), and standardization is
monotonic per feature, so the surviving column indices map directly back to original feature
indices.

Determinism (REQ-021 / AC-005): liblinear's primal-L1 coordinate descent is a deterministic function
of the (standardized) training data and penalty; a ``random_state`` is still drawn from the single
shared RNG (CON-003) and passed for consistency with the other selectors.

Leakage-safe (REQ-010 / DEC-005): both the scaler and the model are fit only on
``context.split.train``; validation and test are never read here. Subset scoring is the
orchestrator's job through the external probe, not this selector's.

Satisfies COMP-008 -> REQ-005; AC-005.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from harness.contract import SelectionContext, SubsetSelection, make_selection

# High enough that the standardized primal-L1 fit converges without a ConvergenceWarning
# across the configured penalty range; not a result-affecting knob once converged, so it is
# held here rather than added to the config surface.
_MAX_ITER = 5000


class L1Selector:
    """Select the features with non-zero coefficients under an L1-penalized fit.

    Conforms structurally to :class:`harness.contract.Selector`: a single ``select`` taking the
    shared :class:`SelectionContext` and returning a :class:`SubsetSelection` with no per-step
    series (classical single-shot selection).
    """

    def select(self, context: SelectionContext) -> SubsetSelection:
        """Fit a standardized L1-penalized model on train; keep non-zero coefficients.

        Standardize the training features, fit pure-L1 logistic regression with the configured
        penalty strength (``config.l1_C``), and drop features whose coefficients L1 drives to zero.
        The surviving (non-zero-coefficient) features are returned as a variable-size subset.
        ``coef_`` is ``(n_classes_or_1, n_features)``; a feature is kept if it is non-zero for any
        class.
        """
        train = context.split.train
        random_state = int(context.rng.numpy.integers(0, 2**32))
        model = make_pipeline(
            StandardScaler(),
            OneVsRestClassifier(
                LogisticRegression(
                    penalty="l1",
                    solver="liblinear",
                    C=context.config.l1_C,
                    max_iter=_MAX_ITER,
                    random_state=random_state,
                ),
            ),
        )
        model.fit(train.X, train.y)

        # OneVsRestClassifier exposes no aggregate ``coef_``; stack the per-class binary L1 fits into
        # ``(n_classes_or_1, n_features)``. For a binary label this is one fit identical to the bare
        # estimator (so WDBC is unchanged); for multiclass it is one L1 fit per class. A feature is kept
        # if it is non-zero for any class.
        coef = np.vstack([est.coef_ for est in model[-1].estimators_])
        nonzero = np.flatnonzero(np.any(coef != 0, axis=0))
        return make_selection(nonzero)
