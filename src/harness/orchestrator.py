"""Single-run method orchestrator (COMP-001) — classical portion.

Builds the one shared leakage-safe run context (configuration/seed → dataset → split → probe) and
drives a selection method through the common subset contract on that context: invoke
``selector.select(context)``, score the returned subset through the external Decision-Tree probe,
and collect the method's subset, size, and accuracy. One context is built per run and shared across
every method, which is what puts all methods on equal footing — one split, one probe (CON-004 single
seeded RNG, epic COMPAT-001).

This task wires the classical path only (the relevance top-k baseline). The structure is
deliberately forward-compatible: methods are run from an ordered ``(name, selector)`` list and each
result carries a ``per_step`` slot, so PHASE-003 can add the no-trainer reinforced method (TASK-213)
and drive the windowed Best/Average metrics over its per-step series, and the artifact emitter
(TASK-204) can serialize these results — without reshaping the classical path.

Leakage (REQ-010 / RISK-002): subsets are scored on ``context.split.validation``; the test partition
is reserved for final reported metrics and is not used on this path. The validation-vs-
test scoring deviation is recorded in RISK-002; surfacing held-out test accuracy is deferred to
PHASE-005.

Satisfies COMP-001 -> REQ-001, REQ-002 (classical portion); AC-001 (classical portion).
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Sequence

from config import IrfsConfig, load_config
from data.loader import load
from data.splitter import make_split
from harness.contract import SelectionContext, Selector, StepRecord
from probe import DecisionTreeProbe
from rng import init_rng


def build_run_context(config: Optional[IrfsConfig] = None, *, seed: Optional[int] = None) -> SelectionContext:
    """Wire the one shared run context every method composes through.

    Performs the canonical PHASE-001 wiring in a fixed order so the run is reproducible
    under the active seed: seed the single shared RNG, load the dataset, carve the
    leakage-safe split, and bind the Decision-Tree probe to the training partition. The
    probe and split are shared by every method invoked on the returned context (equal
    footing). ``config`` defaults to the effective configuration (``load_config()``); the
    active ``seed`` defaults to ``config.seeds[0]`` (the primary seed) — a multi-seed sweep
    calls this once per seed.
    """
    config = config or load_config()
    active_seed = seed if seed is not None else config.seeds[0]
    rng = init_rng(active_seed)
    data = load(config)
    split = make_split(data, config, rng)
    probe = DecisionTreeProbe(split.train, config, rng)
    return SelectionContext(split=split, probe=probe, config=config, rng=rng)


class MethodRun(NamedTuple):
    """One method's outcome from a single run.

    ``accuracy`` is the subset's Decision-Tree accuracy on the validation partition (the leakage-
    safe in-run score, Q1). ``per_step`` is the method's per-step accuracy series; it is empty for a
    single-shot classical method and is the forward-compat slot the reinforced engine fills
    (TASK-213).
    """

    name: str
    selected: tuple[int, ...]
    size: int
    accuracy: float
    per_step: tuple[StepRecord, ...] = ()


class MethodOrchestrator:
    """Run selection methods on one shared leakage-safe context and score each subset.

    Builds the shared :class:`SelectionContext` once at construction so every method runs against
    the same split and probe. Methods are supplied to :meth:`run` as an ordered ``(name, selector)``
    list and invoked solely through the common subset contract.
    """

    def __init__(self, config: Optional[IrfsConfig] = None, *, seed: Optional[int] = None) -> None:
        self._config = config or load_config()
        self._context = build_run_context(self._config, seed=seed)

    @property
    def context(self) -> SelectionContext:
        """The shared run context (one split + one probe) all methods are scored on."""
        return self._context

    def run(self, selectors: Sequence[tuple[str, Selector]], *, on_method=None) -> list[MethodRun]:
        """Invoke each selector through the contract and score its subset on validation.

        For each ``(name, selector)``: call ``selector.select(context)`` to produce a subset through
        the common contract, score that subset with the shared probe on the validation partition,
        and collect a :class:`MethodRun`. Test data is not used on this path.

        ``on_method``, when supplied, is an optional observational progress hook called
        ``on_method(name, "start")`` just before a selector runs and ``on_method(name, "done")``
        after — a "sign it's working" for the slow selectors (the reinforced engine, dt_rfe). It
        touches nothing in the run, so results are unchanged; default ``None`` stays silent (callers
        and tests unaffected).
        """
        runs: list[MethodRun] = []
        for name, selector in selectors:
            if on_method is not None:
                on_method(name, "start")
            selection = selector.select(self._context)
            result = self._context.probe.probe(selection.selected, self._context.split.validation)
            runs.append(
                MethodRun(
                    name=name,
                    selected=selection.selected,
                    size=len(selection.selected),
                    accuracy=result.accuracy,
                    per_step=selection.per_step,
                )
            )
            if on_method is not None:
                on_method(name, "done")
        return runs
