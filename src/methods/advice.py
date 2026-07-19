"""Trainer–engine advice adapter (COMP-010) — apply trainer advice to hesitant agents in the engine.

The bridge between the settled trainer block (``trainers``) and the engine's interactive-advice seam
(:class:`engine.seam.ActionAdvisor`, TASK-411). The engine consults the seam once per step, after
every agent has voted, handing over the previous step's action vector and the current votes; this
adapter classifies the agents (COMP-001 / :func:`trainers.classify.classify_agents`) and asks the
active advice source — a single trainer (COMP-002/003) or the hybrid scheduler (COMP-004) — for the
hesitant-agent overrides, returning them as the seam's ``{feature: action}`` map (DEC-001 pluggable
contract, DEC-005).

The engine owns *where* advice lands (it applies the map to the votes before the SELECT-union); this
adapter owns *what* the advice is. The two responsibilities meet only at the seam's narrow map, so
the no-trainer / single-trainer / hybrid configurations differ solely by which advisor (or ``None``)
is injected (REQ-010). The classification is computed here, not in the engine, keeping the engine
ignorant of trainers.

**Step handling.** The seam carries the exploration ``step`` because the hybrid scheduler sequences
trainers by step (COMP-004); a single-trainer advice source ignores it. Both are wrapped behind one
``advice_fn(step, classification, context)`` callable so the adapter itself is schedule-agnostic.

**Leakage safety (AC-007).** This adapter passes only the shared ``context`` through to the
trainers, which read relevance and importances from the training partition alone; no test (or even
validation) decision data reaches the advice. Determinism (CON-003) is inherited: a trainer instance
is created once per advisor and reused across steps, so the relevance trainer's single seeded draw
is cached for the run.

Satisfies COMP-010 -> REQ-010.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Mapping, Sequence

from trainers.classify import classify_agents
from trainers.dt_importance import DTImportanceTrainer
from trainers.hybrid import build_hybrid_scheduler
from trainers.relevance import RelevanceTrainer

if TYPE_CHECKING:
    from config import IrfsConfig
    from harness.contract import SelectionContext
    from trainers.classify import AgentClassification

# A per-step advice source: given the step, the agent partition, and the context, return the
# hesitant-agent overrides. Single trainers ignore ``step``; the hybrid scheduler uses it.
AdviceFn = Callable[[int, "AgentClassification", "SelectionContext"], Mapping[int, int]]


class TrainerAdvisor:
    """Engine ``ActionAdvisor`` seam over a per-step trainer advice source (COMP-010).

    Implements :meth:`advise` by classifying the agents from the prior/current action vectors and
    delegating to the wrapped ``advice_fn``. Holds no state of its own; any caching (e.g. the
    relevance trainer's per-run relevance vector) lives in the advice source captured by
    ``advice_fn``.
    """

    def __init__(self, advice_fn: AdviceFn) -> None:
        self._advice_fn = advice_fn

    def advise(
        self,
        step: int,
        prior_actions: Sequence[int],
        current_actions: Sequence[int],
        context: "SelectionContext",
    ) -> Mapping[int, int]:
        """Classify the agents from ``(prior, current)`` actions and return the active source's
        overrides."""
        classification = classify_agents(prior_actions, current_actions)
        return self._advice_fn(step, classification, context)


def build_relevance_advisor() -> TrainerAdvisor:
    """Advisor driven solely by the relevance trainer (COMP-002); ``step`` is ignored."""
    trainer = RelevanceTrainer()
    return TrainerAdvisor(lambda step, classification, context: trainer.advise(classification, context))


def build_dt_importance_advisor() -> TrainerAdvisor:
    """Advisor driven solely by the DT-importance trainer (COMP-003); ``step`` is ignored."""
    trainer = DTImportanceTrainer()
    return TrainerAdvisor(lambda step, classification, context: trainer.advise(classification, context))


def build_hybrid_advisor(config: "IrfsConfig") -> TrainerAdvisor:
    """Advisor driven by the hybrid scheduler (COMP-004): relevance, then DT-importance, then
    withdraw."""
    scheduler = build_hybrid_scheduler(config)
    return TrainerAdvisor(scheduler.advise)
