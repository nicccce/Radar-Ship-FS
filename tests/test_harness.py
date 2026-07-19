"""Harness domain (D7): the measurement spine and cross-seed aggregation — ``harness/*`` +
probe/metrics.

Three concerns:

- **Common subset contract** (``harness/contract.py``): an arbitrary selector satisfies the ``Selector``
  seam with zero inheritance; ``make_selection`` yields a canonical non-empty subset; the result reserves
  an (empty-by-default) per-step slot.
- **Measurement spine**: a contract-produced subset is directly scoreable by the live Decision-Tree probe
  on the shared validation partition (well-formed accuracy + normalized importances), windowed Best/Average
  summarize a per-step series, and the whole config→load→split→probe→metrics chain runs end to end.
- **Cross-seed aggregate** (``harness/aggregate.py``): the paired comparison anchors ``full_irfs`` against
  both a classical baseline and the reinforced MARLFS baseline, skips an absent reinforced baseline, and
  carries a windowed row for reinforced methods only. Fixtures are written to a tmp dir (no repo artifact).

The test partition's structural sealing is proven in ``test_data.py`` (D1); run-level determinism/leakage in
``test_invariants.py`` (D8).
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from config import load_config
from data.loader import load
from data.splitter import make_split
from harness.aggregate import aggregate
from harness.contract import (
    SelectionContext,
    Selector,
    StepRecord,
    SubsetSelection,
    make_selection,
)
from metrics import compute_windowed_metrics
from probe import DecisionTreeProbe, ProbeResult
from rng import init_rng

# A fixed handful of the 30 WDBC features, so probe scoring is stable.
_SUBSET = [0, 7, 20, 27]


def wire_context() -> SelectionContext:
    """Build a real WDBC SelectionContext via the PHASE-001 wiring chain (fixed seed)."""
    config = load_config()
    rng = init_rng(config.seeds[0])
    data = load(config)
    split = make_split(data, config, rng)
    probe = DecisionTreeProbe(split.train, config, rng)
    return SelectionContext(split=split, probe=probe, config=config, rng=rng)


@pytest.fixture()
def context() -> SelectionContext:
    return wire_context()


# === Common subset contract =======================================================================


class _FixedSelector:
    """Returns a fixed arbitrary subset, ignoring the context (classical-style: no series)."""

    def select(self, context: SelectionContext) -> SubsetSelection:
        return make_selection([2, 0, 1])


class _HalfSelector:
    """Selects the first half of the dataset's features, sized from the context."""

    def select(self, context: SelectionContext) -> SubsetSelection:
        return make_selection(range(max(1, context.n_features // 2)))


class _NotASelector:
    """Has no ``select`` method — must not satisfy the contract."""


def test_arbitrary_selector_satisfies_contract() -> None:
    """Structural conformance: an arbitrary selector is a ``Selector`` by structure alone (no
    inheritance), callable through a ``Selector``-typed parameter; an object lacking ``select`` is
    not a Selector."""
    selector = _FixedSelector()

    assert isinstance(selector, Selector)
    assert Selector not in type(selector).__mro__
    assert isinstance(selector.select(context=None), SubsetSelection)  # type: ignore[arg-type]
    assert not isinstance(_NotASelector(), Selector)


def test_make_selection_canonicalizes_and_rejects_empty() -> None:
    """Indices become a sorted, de-duplicated tuple of ints regardless of input order; an empty
    subset is rejected (mirroring the probe's empty-subset rejection)."""
    selection = make_selection([3, 1, 1, 2])
    assert selection.selected == (1, 2, 3)
    assert all(isinstance(i, int) for i in selection.selected)
    assert make_selection([0, 5, 9]).selected == (0, 5, 9)  # already canonical, unchanged

    with pytest.raises(ValueError):
        make_selection([])


def test_subset_selection_per_step_shape() -> None:
    """``per_step`` defaults empty for a classical-style selection and carries StepRecords when
    set."""
    assert make_selection([0, 1]).per_step == ()

    populated = SubsetSelection(selected=(0, 1), per_step=(StepRecord(subset=(0,), accuracy=0.9),))
    assert len(populated.per_step) == 1
    assert populated.per_step[0].subset == (0,) and populated.per_step[0].accuracy == pytest.approx(0.9)


# === Measurement spine (probe + metrics) ==========================================================


def test_probe_scores_a_contract_subset_well_formed(context: SelectionContext) -> None:
    """End-to-end composability: n_features is derived from the data, a contract subset is
    canonical, and the live probe scores it with a valid accuracy and normalized importances (zero
    off-subset, sum 1)."""
    assert context.n_features == context.split.train.X.shape[1] == 30  # derived, not hardcoded

    half = _HalfSelector().select(context)
    assert half.selected == tuple(sorted(set(half.selected)))  # canonical

    result = context.probe.probe(make_selection(_SUBSET).selected, context.split.validation)
    assert isinstance(result, ProbeResult)
    assert 0.0 <= result.accuracy <= 1.0

    importances = result.feature_importances
    assert importances.shape == (context.n_features,)
    unselected = [i for i in range(context.n_features) if i not in _SUBSET]
    assert np.all(importances[unselected] == 0.0)  # every unselected feature is zero
    assert importances[_SUBSET].sum() == pytest.approx(1.0)  # normalized over the selected set
    assert result.tree is not None and result.tree.tree_ is not None


def test_windowed_metrics_over_a_series() -> None:
    """Best/average over the full series and over a restricted half-open window."""
    series = [0.80, 0.90, 0.85, 0.70]

    best_full, avg_full = compute_windowed_metrics(series)
    assert best_full == pytest.approx(0.90)
    assert avg_full == pytest.approx(sum(series) / len(series))

    best_win, avg_win = compute_windowed_metrics(series, window=(1, 3))  # -> [0.90, 0.85]
    assert best_win == pytest.approx(0.90)
    assert avg_win == pytest.approx((0.90 + 0.85) / 2)


def test_full_chain_runs_without_error(context: SelectionContext) -> None:
    """The measurement spine runs top to bottom: config→load→split→probe→metrics without raising."""
    accuracy = context.probe.probe(make_selection(_SUBSET).selected, context.split.validation).accuracy
    best, average = compute_windowed_metrics([accuracy])
    assert best == pytest.approx(accuracy)
    assert average == pytest.approx(accuracy)


# === Cross-seed aggregate =========================================================================


def _write_seed(base_dir, dataset, seed, *, test_n_samples, methods, windowed=None) -> None:
    """Write a seed's ``test.json`` (+ optional ``selection.json``) under
    ``<base>/<dataset>/seed-<n>/``.

    ``methods`` maps method name → held-out test accuracy; ``windowed`` (optional) maps a reinforced
    method name → ``{"best", "average"}`` and is written into ``selection.json``'s comparison block.
    All fixtures are fabricated here in a tmp dir — no pre-existing repo artifact is read.
    """
    seed_dir = os.path.join(base_dir, dataset, f"seed-{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    with open(os.path.join(seed_dir, "test.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {
                "test_n_samples": test_n_samples,
                "methods": [
                    {"name": name, "size": 5, "validation": 0.9, "test": test}
                    for name, test in methods.items()
                ],
            },
            fh,
        )
    if windowed is not None:
        with open(os.path.join(seed_dir, "selection.json"), "w", encoding="utf-8") as fh:
            json.dump({"comparison": {"windowed": windowed}}, fh)


def test_paired_comparison_anchors_full_irfs_against_marlfs_and_classicals(tmp_path) -> None:
    """full_irfs is paired against both a classical baseline (dt_rfe) and the reinforced MARLFS
    baseline, and never against itself; the paired deltas aggregate correctly across two seeds."""
    base = str(tmp_path)
    _write_seed(
        base,
        "wdbc",
        1,
        test_n_samples=100,
        methods={"dt_rfe": 0.83, "marlfs": 0.80, "full_irfs": 0.85},
    )
    _write_seed(
        base,
        "wdbc",
        2,
        test_n_samples=100,
        methods={"dt_rfe": 0.88, "marlfs": 0.78, "full_irfs": 0.90},
    )

    agg = aggregate("wdbc", base_dir=base, seeds=[1, 2])
    pairs = {p["baseline"]: p for p in agg["paired_vs_full_irfs"]}

    assert "dt_rfe" in pairs and "marlfs" in pairs
    assert "full_irfs" not in pairs  # the headline is never paired against itself

    # full_irfs − marlfs: deltas (0.05, 0.12) → mean 0.085, two wins; samples (5.0, 12.0) → 8.5.
    marlfs = pairs["marlfs"]
    assert (marlfs["win"], marlfs["tie"], marlfs["loss"]) == (2, 0, 0)
    assert abs(marlfs["mean_delta_acc"] - 0.085) < 1e-9
    assert abs(marlfs["mean_delta_samples"] - 8.5) < 1e-9


def test_absent_reinforced_baseline_is_skipped(tmp_path) -> None:
    """A run without marlfs yields no marlfs pairing — reinforced baselines absent from a run are
    skipped."""
    base = str(tmp_path)
    _write_seed(base, "wdbc", 1, test_n_samples=100, methods={"dt_rfe": 0.83, "full_irfs": 0.85})

    agg = aggregate("wdbc", base_dir=base, seeds=[1])
    assert {p["baseline"] for p in agg["paired_vs_full_irfs"]} == {"dt_rfe"}


def test_marlfs_row_carries_windowed_metrics(tmp_path) -> None:
    """Marlfs is reinforced → it gets a windowed Best/Average row; a classical baseline does not."""
    base = str(tmp_path)
    _write_seed(
        base,
        "wdbc",
        1,
        test_n_samples=100,
        methods={"dt_rfe": 0.83, "marlfs": 0.80, "full_irfs": 0.85},
        windowed={
            "marlfs": {"best": 0.82, "average": 0.79},
            "full_irfs": {"best": 0.86, "average": 0.84},
        },
    )

    agg = aggregate("wdbc", base_dir=base, seeds=[1])
    rows = {m["name"]: m for m in agg["methods"]}

    assert rows["marlfs"]["windowed"] == {"best_mean": 0.82, "average_mean": 0.79}
    assert rows["dt_rfe"]["windowed"] is None
