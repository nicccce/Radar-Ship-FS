"""Unified final Logistic-Regression scorer for frozen feature subsets.

The scorer has no feature-selection logic. It fits StandardScaler and LogisticRegression on the
full development training data supplied by the caller, then evaluates the already-frozen subset on
the held-out test data exactly once.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Optional, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class LogisticRegressionMetrics(NamedTuple):
    train_accuracy: float
    test_accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    f1_macro: float
    roc_auc: float
    confusion_matrix: list[list[int]]
    positive_label: int


def score_frozen_subset_with_lr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    subset: Sequence[int],
    *,
    C: float = 1.0,
    solver: str = "liblinear",
    max_iter: int = 5000,
    class_weight: Optional[str] = "balanced",
    random_state: int = 42,
) -> LogisticRegressionMetrics:
    """Fit the shared LR pipeline on training data and score one frozen subset on test."""
    subset_idx = np.asarray(sorted({int(index) for index in subset}), dtype=int)
    if subset_idx.size == 0:
        raise ValueError("subset must contain at least one feature")
    if subset_idx.min() < 0 or subset_idx.max() >= X_train.shape[1]:
        raise ValueError("subset contains an out-of-range feature")
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("train and test must have the same feature count")

    classes = np.unique(y_train)
    if classes.size != 2:
        raise ValueError("the current final scorer expects binary classification")
    positive_label = int(classes[-1])

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logistic_regression",
                LogisticRegression(
                    C=C,
                    solver=solver,
                    max_iter=max_iter,
                    class_weight=class_weight,
                    random_state=random_state,
                ),
            ),
        ]
    )
    pipeline.fit(X_train[:, subset_idx], y_train)
    train_prediction = pipeline.predict(X_train[:, subset_idx])
    test_prediction = pipeline.predict(X_test[:, subset_idx])
    probabilities = pipeline.predict_proba(X_test[:, subset_idx])
    lr = pipeline.named_steps["logistic_regression"]
    positive_column = int(np.flatnonzero(lr.classes_ == positive_label)[0])
    positive_probability = probabilities[:, positive_column]
    binary_test_target = (np.asarray(y_test) == positive_label).astype(int)

    return LogisticRegressionMetrics(
        train_accuracy=float(accuracy_score(y_train, train_prediction)),
        test_accuracy=float(accuracy_score(y_test, test_prediction)),
        balanced_accuracy=float(balanced_accuracy_score(y_test, test_prediction)),
        precision=float(precision_score(y_test, test_prediction, pos_label=positive_label, zero_division=0)),
        recall=float(recall_score(y_test, test_prediction, pos_label=positive_label, zero_division=0)),
        f1=float(f1_score(y_test, test_prediction, pos_label=positive_label, zero_division=0)),
        f1_macro=float(f1_score(y_test, test_prediction, average="macro", zero_division=0)),
        roc_auc=float(roc_auc_score(binary_test_target, positive_probability)),
        confusion_matrix=confusion_matrix(y_test, test_prediction, labels=classes).astype(int).tolist(),
        positive_label=positive_label,
    )


def lr_metrics_to_dict(metrics: LogisticRegressionMetrics) -> dict[str, Any]:
    return {field: value for field, value in zip(LogisticRegressionMetrics._fields, metrics)}
