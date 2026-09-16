"""Small invariant tests for exhaustive coverage, bridges, and actual fit accounting."""

import numpy as np

from radar_ship_fs.ppo.ppo_graph import FeatureGraph
from run_search_diagnosis import Ledger, add_double, covered, double_covered, swaps
from run_unified_baselines import build_seed_context


def graph(d):
    identity = np.eye(d, dtype=np.float32)
    return FeatureGraph(
        identity,
        identity,
        identity,
        identity,
        np.column_stack([np.arange(d), np.arange(d)]).astype(np.float32),
        np.arange(d),
        np.arange(d),
        np.arange(d),
        0.8,
        0,
        0,
    )


def test_exhaustive_neighbors_are_exact():
    for k in (8, 16, 32):
        s = tuple(range(k))
        neighbors = swaps(s, 65)
        assert len(neighbors) == len(set(neighbors)) == k * (65 - k)
        assert all(len(set(s) - set(t)) == len(set(t) - set(s)) == 1 for t in neighbors)


def test_mask_covers_16_pairs_and_can_exclude_additions():
    g, s = graph(12), tuple(range(6))
    neighbors = swaps(s, 12)
    assert sum(all(covered(g, s, t)) for t in neighbors) == 16
    assert covered(g, s, (1, 2, 3, 4, 5, 6)) == (True, False)
    assert double_covered(g, s, (2, 3, 4, 5, 10, 11))
    assert not double_covered(g, s, (0, 1, 2, 3, 6, 7))


def test_cache_accounts_for_actual_fits_and_gain():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(60, 8)).astype(np.float32)
    y = np.tile([0, 1], 30)
    context = build_seed_context(X, y, seed=42, validation_fraction=0.25, n_splits=5)
    ledger = Ledger(context, np.arange(1, 9), graph(8), jobs=1)
    ledger.score([(1, 0), (0, 1)], source="start")
    ledger.score([(0, 2)], source="neighbor", anchor=(0, 1))
    assert ledger.fits == 10
    assert [r["actual_new_fit_count"] for r in ledger.events] == [5, 0, 5]
    assert ledger.events[-1]["gain_accuracy"] == (
        ledger.cache[(0, 2)]["search_accuracy"] - ledger.cache[(0, 1)]["search_accuracy"]
    )


def test_double_budget_and_two_out_two_in_without_validation():
    class FakeLedger:
        def __init__(self):
            self.ids, self.graph, self.cache = np.arange(12), graph(12), {}

        def score(self, subsets, **kwargs):
            for s in subsets:
                if s not in self.cache:
                    self.cache[s] = {"id": len(self.cache), "search_accuracy": 0.5}
            return [0.5] * len(subsets)

    ledger, groups, s = FakeLedger(), [], tuple(range(6))
    neighbors = swaps(s, 12)
    ledger.score([s] + neighbors)
    chosen = add_double(ledger, groups, s, neighbors, [0.5] * len(neighbors), 6, 42)
    assert chosen == s
    assert len(groups[0]["members"]) <= 512
    by_id = {v["id"]: key for key, v in ledger.cache.items()}
    for member in groups[0]["members"]:
        target = by_id[member["candidate_id"]]
        assert len(set(s) - set(target)) == len(set(target) - set(s)) == 2
        assert member["bridge_gain"] == 0
