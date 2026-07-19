"""IRFS seam adapters (COMP-011) — bridge the feedback modules onto the engine's per-agent seam.

Two shape mismatches are reconciled here, behind the engine's *unchanged* interface and subset
contract (DEC-004, CON-002 / ``engine.seam``):

- :class:`TreeStateSeamAdapter` / :class:`TrainableGCNSeamAdapter` adapt a **subset-level** state
  encoder — one identity-bearing row per feature over the graph on ``selected ∪ {feature}`` — to the
  engine's **per-agent** ``StateEncoder`` seam ``encode(feature, selected, context)`` by returning the
  calling agent's own row. The full ``(N, dimension)`` matrix is memoized per subset so the heavy
  augmented-graph build runs once per exploration step, not once per agent.

- :class:`PersonalizedRewardSeamAdapter` adapts the per-agent reward **vector**
  (:func:`reward.personalize.per_agent_reward_vector`) to the engine's scalar ``RewardFunction`` seam
  ``reward(selected, context, *, agent)``, returning agent ``i``'s component. Each agent therefore learns
  from its own personalized signal ``I_i·(Acc−βR)`` (or the frequency share) when selected, exactly zero
  when deselected.

**Frequency-scheme history (gap-fill).** The ``frequency`` reward scheme needs historical selection
counts, which the seam signature does not carry. The reward adapter keeps its own per-feature tally,
incremented once per *committed subset* it is asked to reward (detected by subset change, so the N
per-agent calls within a step count once). Immediately-repeated subsets — common at convergence — are
counted once; a recorded interpretation of the reference's under-specified "historical selection"
(Q-001 / RISK-001). The headline default scheme is ``dt_importance`` (ASM-001), which needs no history and
leaves this adapter stateless.

**Leakage safety (REQ-013 / AC-007).** Every bound signal is computed only on non-test partitions: the
state on ``split.train``, the reward's accuracy on ``split.validation`` and its correlation on
``split.train``, importances from the train-fit probe. The engine never releases the test partition.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Sequence

import numpy as np

from reward.overall import overall_reward
from reward.personalize import per_agent_reward_vector
from rng import SeededRng
from state.encoder import TreeStructuredStateEncoder
from state.gcn_encoder import TrainableGCNEncoder

if TYPE_CHECKING:
    from config import IrfsConfig
    from harness.contract import SelectionContext


class TreeStateSeamAdapter:
    """Per-agent ``StateEncoder`` seam over the tree-structured encoder's per-agent state matrix.

    Presents the encoder's :meth:`~state.encoder.TreeStructuredStateEncoder.encode_all` —
    one identity-bearing row per feature over the graph on ``selected ∪ {feature}`` — through the
    engine's per-agent ``encode(feature, selected, context)`` signature by returning agent ``feature``'s
    own row. Each agent therefore observes a distinct, feature-specific, subset-reactive vector (restoring
    the per-agent identity the earlier pooled-and-shared mapping had collapsed). The full ``(N, dimension)``
    matrix is memoized per subset, so the heavy augmented-graph builds run once per exploration step rather
    than once per agent.
    """

    def __init__(self, encoder: Optional[TreeStructuredStateEncoder] = None) -> None:
        self._encoder = encoder if encoder is not None else TreeStructuredStateEncoder()
        self._cache_key: Optional[tuple[int, ...]] = None
        self._cache_mat: "Optional[np.ndarray]" = None

    @property
    def dimension(self) -> int:
        """Fixed state width — delegated to the wrapped encoder (CON-004)."""
        return self._encoder.dimension

    def encode(self, feature: int, selected: Sequence[int], context: "SelectionContext") -> "np.ndarray":
        """Return agent ``feature``'s own per-agent state row for ``selected``.

        The encoder's full per-agent matrix is memoized by subset so the N per-agent calls within a
        step reuse one computation; this method indexes out the calling agent's row.
        """
        key = tuple(int(s) for s in selected)
        if key != self._cache_key:
            self._cache_key = key
            self._cache_mat = self._encoder.encode_all(selected, context)
        return self._cache_mat[int(feature)].copy()


class TrainableGCNSeamAdapter:
    """Per-agent ``StateEncoder`` seam over the **trainable** GCN encoder's per-agent state matrix.

    The trained counterpart to :class:`TreeStateSeamAdapter` (DEC-005): it presents
    :meth:`~state.gcn_encoder.TrainableGCNEncoder.encode_all` — the ``(n_features, dimension)``
    learned/pooled matrix over the augmented graph — through the engine's per-agent
    ``encode(feature, selected, context)`` seam by returning agent ``feature``'s own row, exactly as the
    fixed adapter does. The full matrix is memoized per subset so the heavy graph build runs once per
    exploration step, not once per agent.

    **Acting path returns detached numpy (CON-R-002 / DEC-005).** ``encode_all`` here returns an
    autograd-connected ``torch`` tensor; this adapter ``.detach()``-es it to a plain numpy matrix so the
    engine consumes the seam's numpy contract **unchanged** — the engine never sees a tensor. The
    autograd re-encode that the joint learner trains through is TASK-005, not this adapter; PHASE-001
    keeps the weights static, so the acting path is purely a forward read.

    **Seeded lazy build (CON-R-001).** The wrapped :class:`TrainableGCNEncoder` needs a ``random_state``,
    which only exists once a context (and thus the single shared RNG) is in hand. The encoder is therefore
    built on the first :meth:`encode` from ``context.rng`` with one integer draw — the same one-time
    shared-RNG draw the Decision-Tree probe and relevance trainer use (CON-003) — and reused thereafter.
    :attr:`dimension` is known from config (``gcn_hidden_dim``) before that first build, so the engine can
    size its policy input without forcing a build. :meth:`parameters` exposes the live encoder's learnable
    parameters for the registration handoff (RISK-003), building the encoder if a context was supplied at
    construction (eager registration) or after the first encode otherwise.
    """

    def __init__(
        self,
        config: "IrfsConfig",
        *,
        encoder: Optional[TrainableGCNEncoder] = None,
        rng: Optional[SeededRng] = None,
    ) -> None:
        self._config = config
        self._encoder = encoder  # may be None until the first encode draws a seed from context.rng
        # Optional eager-build RNG: when the selector hands the shared RNG in, the encoder (and its
        # parameters) can be registered at selection time rather than waiting for the first encode.
        if self._encoder is None and rng is not None:
            self._encoder = self._build_encoder(rng)
        self._cache_key: Optional[tuple[int, ...]] = None
        self._cache_mat: "Optional[np.ndarray]" = None

    def _build_encoder(self, rng: SeededRng) -> TrainableGCNEncoder:
        """Construct the trainable encoder, drawing ``random_state`` once from the shared RNG
        (CON-003)."""
        random_state = int(rng.numpy.integers(0, 2**32))
        return TrainableGCNEncoder(
            output_dim=self._config.gcn_hidden_dim,
            layers=self._config.gcn_layers,
            activation=self._config.activation,
            random_state=random_state,
        )

    @property
    def dimension(self) -> int:
        """Fixed state width — ``gcn_hidden_dim``, known before the encoder is built (CON-004)."""
        if self._encoder is not None:
            return self._encoder.dimension
        return int(self._config.gcn_hidden_dim)

    def parameters(self):
        """The live encoder's learnable parameters (registration handoff for TASK-005; RISK-003).

        Raises ``RuntimeError`` if the encoder has not been built yet (no context/RNG ever
        supplied), making an inert registration loud rather than silent.
        """
        if self._encoder is None:
            raise RuntimeError(
                "TrainableGCNSeamAdapter has no encoder yet; build it via an eager rng at construction "
                "or run one encode() so a context.rng seeds it before reading parameters()."
            )
        return self._encoder.parameters()

    def trainable_encoder(self) -> "TrainableGCNEncoder":
        """The live wrapped :class:`TrainableGCNEncoder` (joint-learner re-encode handoff,
        TASK-005).

        The joint learner needs the encoder's autograd-connected ``encode_all`` (not this adapter's
        detached seam read) to re-encode stored subsets under autograd, and the encoder's real
        parameter tensors for its single optimizer. Both are the SAME object whose ``parameters()``
        the registration handoff exposes, so the optimizer owns live tensors — never the inert
        accessor. Raises ``RuntimeError`` if the encoder has not been built yet, exactly like
        :meth:`parameters`.
        """
        if self._encoder is None:
            raise RuntimeError(
                "TrainableGCNSeamAdapter has no encoder yet; run one encode() so a context.rng seeds "
                "it (or build eagerly) before reading the trainable encoder for the joint learner."
            )
        return self._encoder

    def encode(self, feature: int, selected: Sequence[int], context: "SelectionContext") -> "np.ndarray":
        """Return agent ``feature``'s detached-numpy state row for ``selected`` (engine seam
        contract).

        On the first call the encoder is seeded from ``context.rng`` (CON-003). The encoder's full
        per-agent ``torch`` matrix is detached to numpy and memoized by subset, so the N per-agent
        calls within a step reuse one forward build; this method indexes out the calling agent's
        row.
        """
        if self._encoder is None:
            self._encoder = self._build_encoder(context.rng)
        key = tuple(int(s) for s in selected)
        if key != self._cache_key:
            self._cache_key = key
            # encode_all returns an autograd-connected torch tensor; detach to the seam's numpy contract
            # so the engine (TASK-005 owns the autograd re-encode, not this acting path).
            self._cache_mat = self._encoder.encode_all(selected, context).detach().numpy()
        return self._cache_mat[int(feature)].copy()


class PersonalizedRewardSeamAdapter:
    """Scalar ``RewardFunction`` seam over the per-agent reward vector.

    Returns agent ``i``'s component of :func:`per_agent_reward_vector` (zero for a deselected
    agent). The vector is memoized per committed subset so the N per-agent calls within a step share
    one computation; for the ``frequency`` scheme the per-feature selection tally is advanced once
    per committed subset at that same point. ``agent=None`` falls back to the un-personalized
    overall reward for seam conformance.
    """

    def __init__(self) -> None:
        self._counts: Optional[List[float]] = None
        self._cache_key: Optional[tuple[int, ...]] = None
        self._cache_vec: "Optional[np.ndarray]" = None

    def reward(
        self,
        selected: Sequence[int],
        context: "SelectionContext",
        *,
        agent: Optional[int] = None,
    ) -> float:
        """Per-agent reward for ``selected``; agent ``i``'s share, or the overall reward when
        ``agent`` is None."""
        if agent is None:
            return float(overall_reward(selected, context))

        # Path B (config-gated, cause-C control): the symmetric mode hands every agent — selected and
        # deselected alike — the full overall reward, restoring the full-magnitude, action-symmetric
        # signal the minimal engine converged with. Default "reference" falls through to the faithful
        # importance-weighted, deselected-zero scheme below (fidelity note when non-reference).
        if context.config.per_agent_credit == "symmetric":
            return float(overall_reward(selected, context))

        key = tuple(int(s) for s in selected)
        if key != self._cache_key:
            selection_counts = self._advance_history(key, context) if _is_frequency(context) else None
            self._cache_vec = per_agent_reward_vector(selected, context, selection_counts=selection_counts)
            self._cache_key = key
        return float(self._cache_vec[int(agent)])

    def _advance_history(self, committed: tuple[int, ...], context: "SelectionContext") -> List[float]:
        """Increment the per-feature selection tally once for this newly committed subset, then
        return it."""
        if self._counts is None:
            self._counts = [0.0] * context.n_features
        for feature in committed:
            self._counts[feature] += 1.0
        return self._counts


def _is_frequency(context: "SelectionContext") -> bool:
    return context.config.reward_scheme == "frequency"
