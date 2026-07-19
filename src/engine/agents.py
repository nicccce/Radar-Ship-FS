"""Feature agents (COMP-004) — one independent select/deselect agent per feature.

The engine represents each feature with its own :class:`FeatureAgent` (REQ-004): the number of
agents follows the dataset's feature count, and each agent decides — from its seam-supplied state
vector — whether to select or deselect *its own* feature, by consulting its private
:class:`~engine.policy.ValuePolicy` ε-greedily. The agents are independent: each owns a separate
value network and (in TASK-209) learns only from its own feature's transitions.

:func:`build_agents` is the wiring point. It sizes the agent population to ``context.n_features``
and gives every agent a policy whose input dimensionality is read from the state encoder's
``dimension`` (the seam, not a hardcoded width — CON-005) and whose weight initialization is seeded
by an integer drawn from the single shared RNG (CON-003), drawn once per agent in feature order so
the whole population is reproducible under the seed.

This is the acting population; the exploration loop that steps it each step is TASK-210, and the
temporal-difference updates that train each policy are TASK-209.

Satisfies COMP-004 -> REQ-004.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import numpy as np

from engine.policy import ValuePolicy

if TYPE_CHECKING:
    from engine.seam import StateEncoder
    from harness.contract import SelectionContext


class FeatureAgent:
    """One independent agent owning a single feature and the value policy it acts through."""

    def __init__(self, feature: int, policy: ValuePolicy) -> None:
        self.feature = int(feature)
        self.policy = policy

    def act(self, state: np.ndarray, rng, exploitation_probability: float) -> int:
        """Ε-greedily choose ``ACTION_SELECT`` or ``ACTION_DESELECT`` for this agent's feature."""
        return self.policy.select_action(state, rng, exploitation_probability)


def build_agents(context: "SelectionContext", encoder: "StateEncoder") -> List[FeatureAgent]:
    """Create one :class:`FeatureAgent` per dataset feature, sized and seeded from ``context``.

    The population size equals ``context.n_features`` (REQ-004). Each agent's policy is sized to
    ``encoder.dimension`` (the state seam) and seeded with one ``random_state`` integer drawn from
    the shared RNG (CON-003) — drawn in feature order, so the agents are reproducible under the
    seed.
    """
    config = context.config
    agents: List[FeatureAgent] = []
    for feature in range(context.n_features):
        random_state = int(context.rng.numpy.integers(0, 2**32))
        policy = ValuePolicy(
            state_dim=encoder.dimension,
            hidden_layer_sizes=config.hidden_layer_sizes,
            activation=config.activation,
            learning_rate=config.learning_rate,
            random_state=random_state,
        )
        agents.append(FeatureAgent(feature=feature, policy=policy))
    return agents
