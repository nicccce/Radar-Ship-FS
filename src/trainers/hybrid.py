"""Hybrid teaching scheduler (COMP-004) — sequence the two trainers, then withdraw guidance.

The hybrid teaching method the headline IRFS comparison turns on (REQ-004 / FLOW-002). It runs one
trainer for an initial stretch of steps, switches to the second for a following stretch, then withdraws
guidance entirely — with the switch points set by configuration (COMP-025 / :mod:`config`). It does
not produce advice itself; it only selects *which* trainer is active at a given step, delegating the
advice to that trainer's ``advise`` (COMP-002 / COMP-003). The classifier (COMP-001) and the two trainers
are settled upstream services this component merely sequences.

Boundary semantics — two configurable switch points ``switch`` and ``withdraw`` (``hybrid_switch_step``,
``hybrid_withdraw_step``), with ``0 <= switch <= withdraw``:

- steps ``[0, switch)`` — the **relevance** trainer (COMP-002), the initial stretch;
- steps ``[switch, withdraw)`` — the **DT-importance** trainer (COMP-003), the following stretch;
- steps ``[withdraw, ...)`` — **no trainer**: guidance is withdrawn and no advice is given.

Degenerate boundaries are allowed and well-defined: ``switch == 0`` skips the relevance stretch,
``switch == withdraw`` skips the DT-importance stretch, and ``withdraw == 0`` withdraws from the first
step (the no-trainer configuration TASK-405 already runs is the natural ``withdraw == 0`` case).

The scheduler adds no data access and no randomness of its own — leakage safety (REQ-013) and determinism
(CON-003) are exactly those of whichever trainer it delegates to. Applying the selected trainer's advice
onto the engine's actions is BLOCK-004's job (TASK-411), not this component's.

Satisfies COMP-004 -> REQ-004.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Protocol

from trainers.dt_importance import DTImportanceTrainer
from trainers.relevance import RelevanceTrainer

if TYPE_CHECKING:
    from config import IrfsConfig
    from harness.contract import SelectionContext
    from trainers.classify import AgentClassification


class Trainer(Protocol):
    """The advice surface every trainer exposes (COMP-002 / COMP-003)."""

    def advise(
        self, classification: "AgentClassification", context: "SelectionContext"
    ) -> Dict[int, int]: ...


class HybridTeachingScheduler:
    """Select the active trainer per step across two configurable boundaries, then withdraw
    (COMP-004).

    Sequences the ``relevance`` trainer over ``[0, switch_step)``, the ``dt_importance`` trainer
    over ``[switch_step, withdraw_step)``, and no trainer thereafter. The trainers are injected so
    the schedule can be exercised over fakes; :func:`build_hybrid_scheduler` wires the real ones
    from configuration.
    """

    def __init__(
        self,
        relevance: Trainer,
        dt_importance: Trainer,
        switch_step: int,
        withdraw_step: int,
    ) -> None:
        if not 0 <= switch_step <= withdraw_step:
            raise ValueError(
                f"hybrid boundaries must satisfy 0 <= switch ({switch_step}) <= withdraw ({withdraw_step})"
            )
        self._relevance = relevance
        self._dt_importance = dt_importance
        self._switch_step = switch_step
        self._withdraw_step = withdraw_step

    def active_trainer(self, step: int) -> Optional[Trainer]:
        """Return the trainer active at ``step``, or ``None`` once guidance is withdrawn.

        Relevance over ``[0, switch_step)``, DT-importance over ``[switch_step, withdraw_step)``,
        then ``None`` for every ``step >= withdraw_step``.
        """
        if step < self._switch_step:
            return self._relevance
        if step < self._withdraw_step:
            return self._dt_importance
        return None

    def advise(
        self,
        step: int,
        classification: "AgentClassification",
        context: "SelectionContext",
    ) -> Dict[int, int]:
        """Advise via the trainer active at ``step``; empty advice once guidance is withdrawn."""
        trainer = self.active_trainer(step)
        if trainer is None:
            return {}
        return trainer.advise(classification, context)


def build_hybrid_scheduler(config: "IrfsConfig") -> HybridTeachingScheduler:
    """Wire the hybrid scheduler from configuration — the real trainers and the configured
    boundaries."""
    return HybridTeachingScheduler(
        relevance=RelevanceTrainer(),
        dt_importance=DTImportanceTrainer(),
        switch_step=config.hybrid_switch_step,
        withdraw_step=config.hybrid_withdraw_step,
    )
