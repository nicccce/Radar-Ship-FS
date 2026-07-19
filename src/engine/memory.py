"""Experience memory + temporal-difference updates (COMP-006 + COMP-005 updater portion).

This is the *learning* half of the no-trainer engine. :class:`ExperienceMemory` stores each agent's
transitions — ``(state, action, reward, next_state)`` — and supplies sampled mini-batches; the
:func:`td_update` function replays a mini-batch to move an agent's value policy toward the temporal-
difference (Bellman) target. The agents from TASK-208 act; this module is what lets them improve.
The exploration loop that *triggers* storage and updates each step is TASK-210.

Transitions are kept per agent (keyed by feature id), because the agents are independent — each owns
a separate value network (TASK-208) and learns only from its own feature's experience. Sampling and
the update draw their randomness from the single shared RNG (CON-003), so a given seed reproduces
the same mini-batches and the same learning.

No separate target network is used (ASM-005 / DEC-004): the bootstrap value of the next state is
read from the online policy itself. The separate target network is held in reserve as the primary
recovery lever for non-convergence (RISK-001 / Q-003); :func:`td_update` computes the bootstrap in
one place so that lever can be attached later without reshaping the update.

This module is import-light (primitives, not the config/rng modules), mirroring
:mod:`engine.policy`, so it stays a thin learning utility over whatever policy the engine holds.

Satisfies COMP-006 -> REQ-005 and the COMP-005 Bellman-updater portion -> REQ-005.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, NamedTuple, Tuple

import numpy as np

if TYPE_CHECKING:
    from engine.policy import ValuePolicy
    from rng import SeededRng


class Transition(NamedTuple):
    """One stored step of an agent's experience.

    ``state`` and ``next_state`` are the agent's seam-supplied state vectors before and after the
    step; ``action`` is the chosen action (``ACTION_DESELECT`` / ``ACTION_SELECT``); ``reward`` is
    the overall reward assigned to the step (TASK-207, stored here as a value).

    ``agent``, ``subset`` and ``next_subset`` are the joint-learner (TASK-005) re-encoding fields:
    the acting agent's feature id and the committed subsets *before* and *after* the step. The fixed
    encoder's per-head path (:func:`td_update`) ignores them and reads only the stale numpy
    ``state``/``next_state`` (byte-identical to today); the trained-encoder joint learner ignores
    the stale vectors and **re-encodes** ``subset``/``next_subset`` through the current encoder
    under autograd, because once the encoder's weights move the stored vectors are stale. They
    default to ``None``/empty so any caller that does not need re-encoding (the fixed path) is
    unaffected.
    """

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    agent: "int | None" = None
    subset: Tuple[int, ...] = ()
    next_subset: Tuple[int, ...] = ()


class ExperienceMemory:
    """Per-agent transition store that supplies uniformly-sampled mini-batches.

    Transitions are partitioned by agent (feature id); :meth:`sample` draws only from the named
    agent's own experience, so independent agents never learn from each other's transitions.
    """

    def __init__(self) -> None:
        self._by_agent: Dict[int, List[Transition]] = {}

    def add(self, agent: int, transition: Transition) -> None:
        """Append ``transition`` to ``agent``'s store (unbounded; the step budget bounds growth)."""
        self._by_agent.setdefault(int(agent), []).append(transition)

    def count(self, agent: int) -> int:
        """Number of transitions stored for ``agent``."""
        return len(self._by_agent.get(int(agent), ()))

    def sample(self, agent: int, batch_size: int, rng: "SeededRng") -> List[Transition]:
        """Uniformly sample up to ``batch_size`` of ``agent``'s transitions, without replacement.

        Draws ``min(batch_size, available)`` transitions so a warmup phase with fewer stored
        transitions than the configured batch size still yields a usable batch. Sampling indices
        come from the shared ``rng`` (CON-003), so the mini-batch is reproducible under the seed.
        """
        stored = self._by_agent.get(int(agent), [])
        if not stored:
            return []
        take = min(int(batch_size), len(stored))
        idx = rng.numpy.choice(len(stored), size=take, replace=False)
        return [stored[i] for i in idx]


def _bootstrap_value(policy: "ValuePolicy", next_state: np.ndarray) -> float:
    """Estimated value of ``next_state`` used in the TD target — the next state's greedy value.

    Read from the *online* policy (no separate target network, ASM-005 / DEC-004). This is the
    single place the bootstrap is computed, so the reserved target-network recovery lever (RISK-001
    / Q-003) can be attached here later — by reading a frozen target policy instead — without
    touching the rest of the update.
    """
    return float(np.max(policy.values(next_state)))


def td_update(policy: "ValuePolicy", batch: List[Transition], discount: float) -> None:
    """Apply one temporal-difference (Bellman) update to ``policy`` from ``batch``.

    For each transition the target for the *taken* action is ``reward + discount ·
    value(next_state)`` (Q-learning, with the next-state value bootstrapped from the online policy);
    the untaken action's target is left at the policy's current estimate so only the taken action
    carries error. The batch is applied with a single :meth:`ValuePolicy.update` (one Adam step over
    the MSE of the 2-vector), moving each ``Q(state, action)`` toward its target.

    ``discount`` is passed as a primitive (the engine reads ``config.discount``); a ``None``-empty
    ``batch`` is a no-op. Deterministic given the policy's weights and the batch.
    """
    if not batch:
        return

    states = np.array([np.asarray(t.state, dtype=float) for t in batch])
    targets = np.array([policy.values(t.state) for t in batch], dtype=float)
    for row, transition in enumerate(batch):
        bootstrap = _bootstrap_value(policy, transition.next_state)
        targets[row, transition.action] = transition.reward + discount * bootstrap

    policy.update(states, targets)
