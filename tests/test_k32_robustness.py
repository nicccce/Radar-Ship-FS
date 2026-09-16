import numpy as np

from run_k32_robustness import (
    EPS,
    best_improvement_swaps,
    clean_on_outer_train,
    exact_k_forward,
    inner_repeats,
)


class ToyScorer:
    def __init__(self, values):
        self.values = values
        self.requests = 0

    def score_many(self, subsets):
        self.requests += len(subsets)
        return [self.values.get(tuple(subset), float(sum(subset))) for subset in subsets]

    def costs(self):
        return {"candidate_requests": self.requests}


def test_outer_train_cleaning_tracks_constant_and_duplicate_coordinates() -> None:
    values = np.asarray([[0, 1, 1, 7], [1, 2, 2, 7], [2, 3, 3, 7]], dtype=float)
    ids, metadata = clean_on_outer_train(values)
    assert ids.tolist() == [1, 2]
    assert metadata["constant_feature_ids_1based"] == [4]
    assert metadata["duplicate_feature_mapping_1based"] == {"3": 2}
    assert metadata["fit_scope"] == "outer_train_only"


def test_inner_repeats_cover_outer_train_once_per_repeat() -> None:
    rows = np.arange(100)
    labels = np.tile([0, 1], 50)
    repeats = inner_repeats(rows, labels, outer_fold=0)
    assert len(repeats) == 3
    for repeat in repeats:
        held = [row for fold in repeat for row in fold["held_out"]]
        assert sorted(held) == rows.tolist()
        assert all(not set(fold["fit"]) & set(fold["held_out"]) for fold in repeat)


def test_forward_enumerates_all_features_and_ties_keep_first() -> None:
    scorer = ToyScorer({})
    subset, _, path = exact_k_forward(scorer, n_features=4, k=2, label="toy")
    assert subset == (2, 3)
    assert [row["candidate_count"] for row in path] == [4, 3]
    tie = ToyScorer({(0,): 1.0, (1,): 1.0, (2,): 0.0})
    subset, _, _ = exact_k_forward(tie, n_features=3, k=1, label="toy")
    assert subset == (0,)


def test_swap_requires_strict_improvement_and_checks_full_neighborhood(monkeypatch) -> None:
    monkeypatch.setattr("run_k32_robustness.N_FEATURES", 4)
    start = (0, 1)
    scorer = ToyScorer({(0, 2): 1.0, (0, 3): 1.0, (1, 2): 1.0, (1, 3): 1.0})
    endpoint, score, path, archive, termination = best_improvement_swaps(
        scorer, start, 1.0, max_rounds=2, label="toy"
    )
    assert endpoint == start and score == 1.0
    assert path[0]["candidate_count"] == 4
    assert not path[0]["accepted"]
    assert len(archive) == 1
    assert termination == "no_strict_single_swap_improvement"
    assert path[0]["chosen_objective"] <= 1.0 + EPS

