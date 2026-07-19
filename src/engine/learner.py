"""Joint TD learner & optimizer (TASK-005, PHASE-002) — switch the learning on.

This is the module that makes the shared trained encoder actually *learn*. Where PHASE-001's
:func:`engine.memory.td_update` trains each agent's value head in isolation against a static
encoder, :class:`JointTDLearner` owns **one** :class:`torch.optim.Adam` over the union of the
trained encoder's parameters and every agent head's parameters, and moves them together from a
**single aggregated temporal-difference signal in one backward pass per step** (DEC-002, AC-002).
The state representation therefore adapts over a run — the feature's core claim.

**Why re-encode under autograd.** A stored transition keeps the agent's ``subset`` and ``agent`` id,
not a frozen state vector: once the encoder's weights move, any state computed from the old weights
is stale, and a stale numpy vector carries no gradient back into the encoder. So
:meth:`JointTDLearner.step` **re-encodes each sampled transition's subset through the *current*
encoder under autograd** (``encoder.encode_all(subset, context)`` — the torch-tensor path, not the
detached seam read the acting loop uses), indexes the acting agent's row, and computes the head's
value on that live tensor. The gradient then flows head → encoder, so one ``loss.backward()`` trains
both.

**Loss aggregation (one backward, one step).** For every sampled transition the TD target for the
*taken* action is ``reward + discount · max(values(next_subset))`` (Q-learning, next-state value
bootstrapped from the online head, mirroring :func:`engine.memory._bootstrap_value`); the untaken
action's target is the head's own current estimate, so only the taken action carries error. Each
transition contributes one per-transition MSE over its 2-vector; **all** per-transition losses
across **all** agents are summed into one scalar, then a single ``optimizer.zero_grad()`` /
``loss.backward()`` / ``optimizer.step()`` updates ``encoder ∪ heads`` exactly once (AC-002). The
aggregated scalar loss is returned for observability.

**Determinism (CON-R-001 / DEC-004).** The optimizer is constructed once with a seeded learning
rate; mini-batch sampling draws from the single shared RNG via :meth:`ExperienceMemory.sample`, so a
given seed reproduces the same batches and the same joint update. No global torch RNG is touched.

**Bootstrap value (no-grad).** The next-state bootstrap (``max`` over the next subset's encoded
value) is read **without** building it into the autograd graph — it is the regression *target*, not
a parameter to differentiate (standard Q-learning semi-gradient, matching the per-head path).

**Fixed-encoder mode does not use this module.** The call site (``explore.py``) keeps the per-head
:func:`td_update` loop byte-identical when the fixed encoder is selected (DEC-005 parity path). This
learner is constructed and stepped only on a trained-encoder run.

Satisfies COMP-002/COMP-001 joint-training portion -> REQ-002, REQ-004, REQ-005 (verified TASK-006).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Sequence

import numpy as np
import torch

from engine.policy import N_ACTIONS

if TYPE_CHECKING:
    from engine.agents import FeatureAgent
    from engine.memory import ExperienceMemory
    from harness.contract import SelectionContext
    from rng import SeededRng
    from state.gcn_encoder import TrainableGCNEncoder

# CPU reference platform (matches the encoder/heads): no GPU/MPS nondeterminism (CON-R-001).
_DEVICE = torch.device("cpu")


class JointTDLearner:
    """One optimizer over ``encoder ∪ heads``; one aggregated TD backward per step (DEC-002).

    Constructed once per trained-encoder run with the live encoder and the agent population, so the
    single :class:`torch.optim.Adam` owns the encoder's real parameter tensors together with every
    head's parameters. :meth:`step` performs exactly one optimizer update per exploration step.
    """

    def __init__(
        self,
        encoder: "TrainableGCNEncoder",
        agents: "Sequence[FeatureAgent]",
        *,
        learning_rate: float,
        discount: float,
    ) -> None:
        self._encoder = encoder
        self._agents = list(agents)
        self._discount = float(discount)

        # One optimizer over the UNION of encoder params and every head's params (DEC-002). The
        # encoder tensors come first, then each head in feature order — a fixed, seed-stable order.
        params: List[torch.nn.Parameter] = list(encoder.parameters())
        for agent in self._agents:
            params.extend(agent.policy.parameters())
        self._optimizer = torch.optim.Adam(params, lr=float(learning_rate))

    @property
    def optimizer(self) -> "torch.optim.Optimizer":
        """The single joint optimizer (exposed for the one-optimizer/one-step assertion, AC-002)."""
        return self._optimizer

    def _head_dtype(self, agent: "FeatureAgent") -> torch.dtype:
        """The dtype of an agent head's first parameter (heads are float32; the encoder is float64).

        The encoder emits float64 state rows; the heads were built float32 (policy.py). The re-
        encoded row is cast to the head's dtype before the forward so the matmul dtypes agree, while
        the gradient still flows back into the float64 encoder parameters through the cast.
        """
        return next(agent.policy.parameters()).dtype

    def step(
        self,
        memory: "ExperienceMemory",
        context: "SelectionContext",
        rng: "SeededRng",
        *,
        batch_size: int,
    ) -> float:
        """One joint TD update over ``encoder ∪ heads``; returns the aggregated scalar loss.

        Samples up to ``batch_size`` of each agent's stored transitions from the shared ``rng``, re-
        encodes every sampled transition's subset through the current encoder **under autograd**,
        computes the per-transition TD MSE for the taken action, sums all per-transition losses
        across all agents into one scalar, and applies exactly one backward + one optimizer step. A
        step with no stored transitions for any agent is a no-op returning ``0.0``.
        """
        per_transition_losses: List[torch.Tensor] = []

        for agent in self._agents:
            batch = memory.sample(agent.feature, batch_size, rng)
            if not batch:
                continue
            head = agent.policy.head
            dtype = self._head_dtype(agent)
            for transition in batch:
                # Re-encode the STORED subset through the CURRENT encoder, under autograd, and read
                # the acting agent's row — the live, weight-dependent state the gradient flows through.
                state_row = self._encode_row(transition.subset, transition.agent, context).to(dtype)
                values = head(state_row.reshape(1, -1)).reshape(N_ACTIONS)  # (2,) grad-connected

                # TD target for the TAKEN action: reward + discount * max(values(next_subset)). The
                # bootstrap is a regression target (no grad), matching the per-head semi-gradient path.
                target = values.detach().clone()
                bootstrap = self._bootstrap_value(transition.next_subset, transition.agent, context, dtype)
                target[transition.action] = float(transition.reward) + self._discount * bootstrap

                per_transition_losses.append(((values - target) ** 2).mean())

        if not per_transition_losses:
            return 0.0

        # One aggregated scalar, ONE backward, ONE optimizer step over encoder ∪ heads (AC-002).
        loss = torch.stack(per_transition_losses).sum()
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        return float(loss.detach().cpu())

    def _encode_row(self, subset: Sequence[int], agent: int, context: "SelectionContext") -> torch.Tensor:
        """Agent ``agent``'s autograd-connected state row for ``subset`` (the trainable torch path).

        Uses the encoder's torch-tensor :meth:`encode_all` (NOT the detached seam read), so the row
        carries gradient back into the encoder's ``W``/``bias``.
        """
        matrix = self._encoder.encode_all(subset, context)  # (n_features, dimension), grad-connected
        return matrix[int(agent)]

    def _bootstrap_value(
        self,
        next_subset: Sequence[int],
        agent: int,
        context: "SelectionContext",
        dtype: torch.dtype,
    ) -> float:
        """Greedy value of ``next_subset`` for ``agent`` — the TD bootstrap, computed WITHOUT grad.

        Read from the online head on the encoder's current (no-grad) encoding of the next subset, so
        it never enters the autograd graph (semi-gradient Q-learning, mirroring the per-head path's
        :func:`engine.memory._bootstrap_value`).
        """
        with torch.no_grad():
            row = self._encoder.encode_all(next_subset, context)[int(agent)].to(dtype)
            head = self._agents[int(agent)].policy.head
            out = head(row.reshape(1, -1)).reshape(N_ACTIONS)
            return float(np.max(out.cpu().numpy()))
