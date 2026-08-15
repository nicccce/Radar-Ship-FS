from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier


@dataclass(frozen=True)
class CVResult:
    mean_accuracy: float
    fold_accuracies: tuple[float, ...]


class SubsetEvaluator:
    """Fixed-fold, development-only DT scorer with subset memoization."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        seed: int,
        folds: int,
        n_jobs: int,
        repeats: int = 1,
        row_indices: np.ndarray | None = None,
    ) -> None:
        self.X = X
        self.y = y
        self.seed = int(seed)
        self.n_jobs = int(n_jobs)
        self.row_indices = (
            np.arange(X.shape[0], dtype=int)
            if row_indices is None
            else np.asarray(row_indices, dtype=int)
        )
        splitter = RepeatedStratifiedKFold(
            n_splits=folds,
            n_repeats=repeats,
            random_state=seed,
        )
        self.folds = tuple(
            (fit.astype(int), held.astype(int)) for fit, held in splitter.split(X, y)
        )
        self._cache: dict[bytes, CVResult] = {}

    @property
    def evaluated_subsets(self) -> int:
        return len(self._cache)

    def score(self, mask: np.ndarray) -> CVResult:
        mask = np.asarray(mask, dtype=bool)
        indices = np.flatnonzero(mask)
        if indices.size == 0:
            raise ValueError("cannot evaluate an empty feature subset")
        key = np.packbits(mask).tobytes()
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        def fit_fold(fit_rows: np.ndarray, held_rows: np.ndarray) -> float:
            model = DecisionTreeClassifier(random_state=self.seed)
            model.fit(self.X[fit_rows][:, indices], self.y[fit_rows])
            return float(model.score(self.X[held_rows][:, indices], self.y[held_rows]))

        if self.n_jobs == 1:
            scores = [fit_fold(fit, held) for fit, held in self.folds]
        else:
            scores = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(fit_fold)(fit, held) for fit, held in self.folds
            )
        result = CVResult(float(np.mean(scores)), tuple(float(value) for value in scores))
        self._cache[key] = result
        return result

    def fold_indices(self) -> list[dict[str, list[int]]]:
        return [
            {
                "fit": self.row_indices[fit].tolist(),
                "held_out": self.row_indices[held].tolist(),
            }
            for fit, held in self.folds
        ]


def evaluate_tree_on_test(
    X_development: np.ndarray,
    y_development: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    mask: np.ndarray,
    *,
    seed: int,
) -> dict[str, float | int]:
    indices = np.flatnonzero(mask)
    model = DecisionTreeClassifier(random_state=seed)
    model.fit(X_development[:, indices], y_development)
    prediction = model.predict(X_test[:, indices])
    positive_index = int(np.flatnonzero(model.classes_ == 1)[0])
    probability = model.predict_proba(X_test[:, indices])[:, positive_index]
    return {
        "selected_count": int(indices.size),
        "development_accuracy": float(model.score(X_development[:, indices], y_development)),
        "test_accuracy": float(accuracy_score(y_test, prediction)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_test, prediction)),
        "test_precision": float(precision_score(y_test, prediction, pos_label=1, zero_division=0)),
        "test_recall": float(recall_score(y_test, prediction, pos_label=1, zero_division=0)),
        "test_f1": float(f1_score(y_test, prediction, pos_label=1, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_test, probability)),
    }
