"""Common subset-production contract — the equal-footing seam (COMPAT-001).

Every selection method in this feature satisfies one interface: given a :class:`SelectionContext`
(the one shared leakage-safe split, the one shared Decision-Tree probe, the effective configuration,
and the single seeded RNG), it produces a :class:`SubsetSelection` (the selected feature subset,
plus an optional per-step accuracy series the reinforced engine fills and the classical baseline
leaves empty). A single orchestrator (TASK-203) invokes either method through this identical
``select(context) -> SubsetSelection`` signature, so the relevance top-k baseline (TASK-202) and the
no-trainer reinforced engine (TASK-210) run on equal footing and the PHASE-003 comparison needs no
contract change.

Leakage invariant (REQ-010 / DEC-005): a selector scores candidate subsets only on
``context.split.validation``; test is used later for final metrics.

This contract is distinct from the engine's internal state/reward seam (TASK-205): the engine
*satisfies* this subset contract at its boundary (TASK-210) while consuming its state and reward
through that separate, replaceable seam.

Satisfies REQ-001 (enabling — fixes the shape REQ-001 is delivered against). Stable across this
feature's inner PHASE-001/002/003 (epic COMPAT-001 / CON-003).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Protocol, Sequence, runtime_checkable

if TYPE_CHECKING:
    # Annotation-only imports keep this load-bearing seam import-light — referencing
    # the probe's concrete type must not pull sklearn into every contract consumer.
    from config import IrfsConfig
    from data.splitter import Split
    from probe import DecisionTreeProbe
    from rng import SeededRng


class SelectionContext(NamedTuple):
    """Everything a selection method needs to produce a subset, on the shared footing.

    Each method receives the same context: the classical baseline reads
    ``split.train`` to rank features and ignores ``probe``; the reinforced engine scores
    candidate subsets through ``probe`` on ``split.validation`` and draws its randomness
    from ``rng``. Method-internal machinery (agents, state encoder, reward) is never
    carried here — only the shared services every method composes through.
    """

    split: "Split"
    probe: "DecisionTreeProbe"
    config: "IrfsConfig"
    rng: "SeededRng"

    @property
    def n_features(self) -> int:
        """Total feature count of the dataset (column count of the training partition)."""
        return int(self.split.train.X.shape[1])


class StepRecord(NamedTuple):
    """One step of the reinforced engine's exploration: the subset and its accuracy.

    The per-step series of these records is what PHASE-003 drives the windowed Best/Average metrics
    over; the classical baseline produces none.
    """

    subset: tuple[int, ...]
    accuracy: float


class SubsetSelection(NamedTuple):
    """The result every method returns through the contract.

    ``selected`` is the chosen feature subset as a canonical (sorted, de-duplicated) tuple of column
    indices, directly scoreable by ``DecisionTreeProbe.probe``. ``per_step`` is the reinforced
    engine's per-step accuracy series; it defaults to empty, which is exactly what a classical
    single-shot selector returns.
    """

    selected: tuple[int, ...]
    per_step: tuple[StepRecord, ...] = ()


def make_selection(
    indices: Sequence[int],
    per_step: Sequence[StepRecord] = (),
) -> SubsetSelection:
    """Build a :class:`SubsetSelection` with a canonical, non-empty selected subset.

    ``indices`` are canonicalized to a sorted tuple of unique ``int`` column indices, so two runs
    that select the same features compare equal regardless of selection order (underpinning the
    same-seed reproducibility proof, TASK-215). An empty subset raises ``ValueError``, consistent
    with the probe rejecting an empty subset.
    """
    canonical = tuple(sorted({int(i) for i in indices}))
    if not canonical:
        raise ValueError("selected subset must contain at least one feature index")
    return SubsetSelection(selected=canonical, per_step=tuple(per_step))


@runtime_checkable
class Selector(Protocol):
    """The one interface every selection method satisfies (REQ-001 / COMPAT-001).

    A selector exposes a single ``select`` method taking the shared
    :class:`SelectionContext` and returning a :class:`SubsetSelection`. Conformance is
    structural: any object (function-backed class or engine runner) with a matching
    ``select`` satisfies the contract with no inheritance and no change to the
    orchestrator's calling signature.
    """

    def select(self, context: SelectionContext) -> SubsetSelection:
        """Produce a feature subset from the shared context."""
        ...
