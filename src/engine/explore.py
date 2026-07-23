"""Exploration loop / standalone reinforced engine (COMP-007).

The no-trainer reinforced feature selector, assembled from the engine's parts: it satisfies the
common subset contract (``select(context) -> SubsetSelection``, the same interface the classical
baseline uses), so the orchestrator runs it on equal footing with no contract change.

The loop is a *synchronous population vote* (the Path A direction): starting from a deterministic
random half of the features, each step every feature-agent reads its seam-supplied state (computed
against the currently-committed subset) and votes select/deselect ε-greedily; the union of SELECT
votes becomes the new subset, which is scored once on the validation partition by the shared
Decision-Tree probe. A non-degeneracy guard (AC-004) keeps the committed subset strictly between
none and all — a sweep that votes empty or all-features keeps the previous subset, which also avoids
the probe's empty-subset rejection. The per-step ``(subset, accuracy)`` series is recorded, and the
engine returns the best-accuracy subset it saw together with that full series.

The agents learn as the loop runs: each step the reward for the new subset is read per agent through
the seam's ``agent`` parameter and stored to that agent's transition, and each agent's value policy
is updated by replaying a mini-batch from its experience (temporal-difference, TASK-209). An overall
reward ignores ``agent`` and so returns one value for all (the provisional behavior); a personalized
reward (TASK-405) returns each agent its own signal — both through the one unchanged seam signature.

Leakage invariant (REQ-010 / AC-007): every per-step decision is scored on
``context.split.validation`` only; the test partition has no public attribute and
``release_test_for_final_metrics`` is never called, so no in-run decision can read held-out data.
Determinism (CON-003): the initial subset, the agents' weight initialization, ε-greedy action
selection, and experience sampling all draw from the single shared RNG, so a seed reproduces the
subset and the series.

The state encoder and reward are injectable (defaulting to the minimal state and overall reward),
keeping the engine seam-agnostic: an alternate-shaped substrate runs through it unchanged (REQ-014,
verified at TASK-212), and PHASE-004's richer state/reward replace the defaults without touching
this loop.

Satisfies COMP-007 -> REQ-006, REQ-007, REQ-010, REQ-012, AC-003, AC-004, AC-007; REQ-001 (engine-
side satisfaction of the common subset contract).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, Optional

from engine.agents import build_agents
from engine.learner import JointTDLearner
from engine.memory import ExperienceMemory, Transition, td_update
from engine.policy import ACTION_DESELECT, ACTION_SELECT
from engine.reward_overall import OverallReward
from engine.state_minimal import MinimalStateEncoder
from harness.contract import StepRecord, make_selection

if TYPE_CHECKING:
    import numpy as np

    from engine.seam import ActionAdvisor, RewardFunction, StateEncoder
    from harness.contract import SelectionContext, SubsetSelection


class ReinforcedEngine:
    """Standalone no-trainer reinforced selector satisfying the common subset contract.

    Conforms structurally to :class:`harness.contract.Selector`: a single :meth:`select` taking the
    shared context and returning a :class:`~harness.contract.SubsetSelection` whose ``per_step``
    carries the reinforced engine's per-step accuracy series. The state encoder and reward are
    injectable to keep the engine seam-agnostic (REQ-014).
    """

    def __init__(
        self,
        encoder: "Optional[StateEncoder]" = None,
        reward: "Optional[RewardFunction]" = None,
        advisor: "Optional[ActionAdvisor]" = None,
    ) -> None:
        self._encoder: "StateEncoder" = encoder if encoder is not None else MinimalStateEncoder()
        self._reward: "RewardFunction" = reward if reward is not None else OverallReward()
        # Interactive-advice seam (COMP-010, TASK-411). Left as None for the no-trainer
        # configuration: the loop then skips advice entirely, so the proven no-trainer behavior
        # (TASK-406) is preserved bit-for-bit. A trainer/hybrid advisor overrides hesitant agents'
        # votes each step through the pluggable contract.
        self._advisor: "Optional[ActionAdvisor]" = advisor

    def _initial_subset(self, context: "SelectionContext") -> tuple[int, ...]:
        """A deterministic random half, capped by the configured feature budget when present."""
        n = context.n_features
        k = max(1, n // 2)
        feature_budget = getattr(context.config, "feature_budget", None)
        if feature_budget is not None:
            k = min(k, int(feature_budget))
        chosen = context.rng.numpy.choice(n, size=k, replace=False)
        return tuple(sorted(int(i) for i in chosen))

    def _encode_all(self, agents, subset: tuple[int, ...], context: "SelectionContext") -> "List[np.ndarray]":
        """Each agent's state vector against ``subset`` (re-encoded as the subset changes)."""
        return [self._encoder.encode(agent.feature, subset, context) for agent in agents]

    def _score(self, context: "SelectionContext", subset: tuple[int, ...]) -> float:
        """Validation accuracy of ``subset`` from the shared probe — the leakage-safe scoring path.

        Leakage tripwire (REQ-010 / AC-007): the only partition passed to the probe is
        ``context.split.validation``; the test partition is never referenced and
        ``release_test_for_final_metrics`` is never called.
        """
        validation = context.split.validation
        return float(context.probe.probe(subset, validation).accuracy)

    def _make_joint_learner(
        self, agents, learning_rate: float, discount: float
    ) -> "Optional[JointTDLearner]":
        """The joint learner for a trained-encoder run, or ``None`` for the fixed-encoder parity
        path.

        Joint-training branch (TASK-005 / DEC-005): a trained-encoder run replaces the per-head
        ``td_update`` loop with ONE joint learner that owns the encoder's real parameter tensors
        together with every head and trains them from a single aggregated TD backward per step. The
        caller's first ``_encode_all`` has already built/seeded the encoder behind the seam, so the
        registration handoff (the adapter's live ``TrainableGCNEncoder``) is materialized — never
        the inert accessor. The fixed encoder exposes no ``trainable_encoder`` handoff, so this
        returns ``None`` and the per-head parity path runs UNCHANGED (byte-identical to today).
        """
        if hasattr(self._encoder, "trainable_encoder"):
            return JointTDLearner(
                self._encoder.trainable_encoder(),
                agents,
                learning_rate=learning_rate,
                discount=discount,
            )
        return None

    @staticmethod
    def _is_better_candidate(
        accuracy: float,
        subset: tuple[int, ...],
        best_accuracy: float,
        best_subset: tuple[int, ...],
        feature_budget: Optional[int] = None,
    ) -> bool:
        """Reject over-budget candidates, then prefer accuracy and finally fewer features."""
        if feature_budget is not None and len(subset) > feature_budget:
            return False
        if accuracy > best_accuracy and not math.isclose(accuracy, best_accuracy, rel_tol=0.0, abs_tol=1e-12):
            return True
        return math.isclose(accuracy, best_accuracy, rel_tol=0.0, abs_tol=1e-12) and len(subset) < len(
            best_subset
        )

    def _vote(
        self,
        agents,
        states: "List[np.ndarray]",
        committed: tuple[int, ...],
        context: "SelectionContext",
        rng,
        exploitation_probability: float,
        step: int,
        n: int,
    ) -> "tuple[list, tuple[int, ...]]":
        """One synchronous population vote: return ``(actions, new_subset)`` for this step.

        Each agent votes select/deselect ε-greedily from its state (the only RNG draws here, in
        agent order). Interactive advice (COMP-010, TASK-411) is then applied: with no advisor the
        block is skipped entirely — no extra work and no RNG draw — so the no-trainer run is
        unchanged (TASK-406); with an advisor, the current votes are compared against the previously
        committed subset (the agent's prior "selected" reality) to find hesitant agents, and the
        advisor's ``{feature: action}`` overrides are applied to ``actions`` *before* the union — so
        an override flows into both the committed subset and the per-agent learning signal
        (``actions[i]``). Advice reads only non-test data via ``context`` (AC-007) and draws only
        from the shared RNG (CON-003). The non-degeneracy guard (AC-004) keeps the previous subset
        when a sweep votes empty or all features, so every scored and returned subset is strictly
        between none and all (and the probe is never handed an empty one).
        """
        actions = [agent.act(states[i], rng, exploitation_probability) for i, agent in enumerate(agents)]
        if self._advisor is not None:
            committed_set = set(committed)
            prior_actions = [
                ACTION_SELECT if agent.feature in committed_set else ACTION_DESELECT for agent in agents
            ]
            overrides = self._advisor.advise(step, prior_actions, actions, context)
            for feature, action in overrides.items():
                actions[feature] = action
        voted = tuple(
            sorted(agent.feature for agent, action in zip(agents, actions) if action == ACTION_SELECT)
        )
        new_subset = voted if 0 < len(voted) < n else committed
        return actions, new_subset

    def _learn_step(
        self,
        agents,
        memory: ExperienceMemory,
        states: "List[np.ndarray]",
        actions: list,
        next_states: "List[np.ndarray]",
        committed: tuple[int, ...],
        new_subset: tuple[int, ...],
        context: "SelectionContext",
        rng,
        learner: "Optional[JointTDLearner]",
        mini_batch_size: int,
        discount: float,
    ) -> None:
        """Store each agent's transition and update value policies for this step.

        The reward for ``new_subset`` is read per agent through the seam's ``agent`` parameter (TASK-405),
        so a personalized reward returns each agent its own signal while an overall reward — which ignores
        ``agent`` — yields the same value for all, leaving the provisional broadcast behavior unchanged.
        Each transition stores the subsets (committed before/after) and the agent id alongside the stale
        numpy state vectors: the fixed-encoder per-head path reads the vectors; the joint learner ignores
        them and RE-ENCODES the subsets under autograd (the vectors go stale the moment the encoder's
        weights move). Storing both keeps each path self-contained.

        Fixed-encoder parity path (DEC-005): per-agent ``td_update`` against the static encoder, UNCHANGED
        (byte-identical to today). Trained-encoder joint update (TASK-005): exactly ONE optimizer step over
        encoder ∪ heads per exploration step — per-agent batches from the shared RNG, autograd re-encode,
        per-agent TD MSE summed into one scalar, one backward.
        """
        for i, agent in enumerate(agents):
            reward = self._reward.reward(new_subset, context, agent=agent.feature)
            memory.add(
                agent.feature,
                Transition(
                    states[i],
                    actions[i],
                    reward,
                    next_states[i],
                    agent=agent.feature,
                    subset=committed,
                    next_subset=new_subset,
                ),
            )
            if learner is None:
                td_update(
                    agent.policy,
                    memory.sample(agent.feature, mini_batch_size, rng),
                    discount,
                )
        if learner is not None:
            learner.step(memory, context, rng, batch_size=mini_batch_size)

    def select(self, context: "SelectionContext", *, on_step=None, on_initial=None) -> "SubsetSelection":
        """Run the synchronous population-vote loop and return the best subset + per-step series.

        Each step: every agent votes (:meth:`_vote`), the new subset is scored once on validation
        (:meth:`_score`), the best-so-far is tracked, and the agents learn from the transition
        (:meth:`_learn_step`). ``on_step``, when supplied, is an optional progress hook called once per
        step as ``on_step(step, budget, accuracy, best_accuracy)`` (0-based ``step``). It is purely
        observational. ``on_initial(subset, accuracy)`` exposes the actual initial candidate; its
        accuracy is ``None`` when budget-aware candidate scoring is disabled. Neither hook touches the
        RNG, subset, or learning, so a run with a hook
        is bit-identical to one without. Default ``None`` keeps the engine silent (tests unaffected).
        """
        rng = context.rng
        n = context.n_features
        config = context.config
        exploitation_probability = config.exploitation_probability
        budget = config.exploration_step_budget
        mini_batch_size = config.mini_batch_size
        discount = config.discount
        learning_rate = config.learning_rate
        feature_budget = getattr(config, "feature_budget", None)

        agents = build_agents(context, self._encoder)
        memory = ExperienceMemory()

        initial_subset = self._initial_subset(context)
        committed = initial_subset
        states = self._encode_all(agents, committed, context)
        learner = self._make_joint_learner(agents, learning_rate, discount)

        per_step: List[StepRecord] = []
        best_subset = initial_subset
        # Budget-aware experiments include the initial feasible subset in the candidate pool. With
        # no budget configured, preserve the historical per-step-only selection rule.
        initial_accuracy = self._score(context, initial_subset) if feature_budget is not None else None
        best_accuracy = initial_accuracy if initial_accuracy is not None else -1.0
        if on_initial is not None:
            on_initial(initial_subset, initial_accuracy)

        for step in range(budget):
            actions, new_subset = self._vote(
                agents, states, committed, context, rng, exploitation_probability, step, n
            )

            accuracy = self._score(context, new_subset)
            per_step.append(StepRecord(subset=new_subset, accuracy=accuracy))
            if self._is_better_candidate(
                accuracy,
                new_subset,
                best_accuracy,
                best_subset,
                feature_budget,
            ):
                best_accuracy, best_subset = accuracy, new_subset

            next_states = self._encode_all(agents, new_subset, context)
            self._learn_step(
                agents,
                memory,
                states,
                actions,
                next_states,
                committed,
                new_subset,
                context,
                rng,
                learner,
                mini_batch_size,
                discount,
            )

            committed = new_subset
            states = next_states

            if on_step is not None:
                on_step(step, budget, accuracy, best_accuracy)

        return make_selection(best_subset, per_step=per_step)
