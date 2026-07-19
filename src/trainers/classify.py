"""Agent classifier (COMP-001) — partition agents into participated / assertive / hesitant.

At each exploration step every feature-agent has just produced a select/deselect action, and carries
the action it produced the previous step. This component partitions the agents from that ``(prior,
current)`` action pair (REQ-001 / AC-001), exposing the three sets the trainers' advice is defined
against (FLOW-001). It is the first piece of the trainer block: relevance (COMP-002) and DT-importance
(COMP-003) both advise the *hesitant* set relative to the *assertive* set, over the *participated*
features, so the partition has to exist before any advice can.

Definitions follow the reference algorithms directly. An agent **participated** if its feature was
selected in the prior step. Among the participants:

- **assertive** — selected in *both* steps: a stable, confident keep. These features are the comparison
  baseline the trainers measure a hesitant feature against (e.g. the assertive importance median, COMP-003).
- **hesitant** — selected in the prior step but deselected now: the agent is about to drop a previously
  participated feature.
  These are the agents the trainers steer toward selection when their feature looks comparatively strong.

Trainer advice is **one-directional by construction**: it can only flip a hesitant agent's vote from
deselect back to select (re-keeping a previously participated feature that still looks competitive); it
never flips an assertive agent's vote the other way. The partition therefore only ever gates resistance
to *premature removal* — assertive keeps and newly selected features (``D -> S``, not participated) are
left to normal RL exploration and are outside the trainer comparison entirely.

This yields a clean split of the agents that took part: ``participated == assertive ⊔ hesitant``
(disjoint), while agents deselected in both steps simply did not participate and appear in none of the
sets. Each set is returned as a sorted tuple of feature indices (the agent's feature is its index, the
same convention :func:`engine.agents.build_agents` uses).

Pure and deterministic: the partition is a function of the two action sequences alone — it reads no
dataset partition (so leakage is not even in scope here, REQ-013) and consumes no randomness (CON-003).

Satisfies COMP-001 -> REQ-001 (AC-001).
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

from engine.policy import ACTION_DESELECT, ACTION_SELECT


class AgentClassification(NamedTuple):
    """The per-step partition of feature-agents, each set a sorted tuple of feature indices.

    ``participated`` is the disjoint union of ``assertive`` and ``hesitant``; agents not selected in
    the prior step appear in none of the three, even if their current vote is ``SELECT``.
    """

    participated: tuple[int, ...]
    assertive: tuple[int, ...]
    hesitant: tuple[int, ...]


def classify_agents(
    prior_actions: Sequence[int],
    current_actions: Sequence[int],
) -> AgentClassification:
    """Partition agents into participated / assertive / hesitant from their prior and current
    actions.

    ``prior_actions`` and ``current_actions`` are equal-length sequences of ``ACTION_SELECT`` /
    ``ACTION_DESELECT`` indexed by feature (position ``i`` is agent/feature ``i``). Returns the
    three sets per the module's definitions. Raises ``ValueError`` if the two sequences differ in
    length or contain a value that is neither ``ACTION_SELECT`` nor ``ACTION_DESELECT``.
    """
    if len(prior_actions) != len(current_actions):
        raise ValueError(
            f"prior_actions ({len(prior_actions)}) and current_actions "
            f"({len(current_actions)}) must have one entry per feature-agent"
        )

    participated: list[int] = []
    assertive: list[int] = []
    hesitant: list[int] = []

    for feature, (prior, current) in enumerate(zip(prior_actions, current_actions)):
        prior_selected = _as_selected(prior, feature, "prior_actions")
        current_selected = _as_selected(current, feature, "current_actions")

        if prior_selected:
            participated.append(feature)
            if current_selected:
                assertive.append(feature)
            else:  # prior SELECT -> current DESELECT: the paper's hesitant case
                hesitant.append(feature)

    return AgentClassification(
        participated=tuple(participated),
        assertive=tuple(assertive),
        hesitant=tuple(hesitant),
    )


def _as_selected(action: int, feature: int, which: str) -> bool:
    """Return whether ``action`` is a SELECT, rejecting any value outside the two-action space."""
    if action == ACTION_SELECT:
        return True
    if action == ACTION_DESELECT:
        return False
    raise ValueError(
        f"{which}[{feature}] = {action!r} is not a valid action "
        f"(expected {ACTION_SELECT} SELECT or {ACTION_DESELECT} DESELECT)"
    )
