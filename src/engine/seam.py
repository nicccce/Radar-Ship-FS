"""Engine swap seam — the replaceable state/reward interface the engine consumes (REQ-014).

The no-trainer reinforced engine reads each agent's state and receives its reward through
exactly two interfaces defined here, and embeds neither implementation:

- :class:`StateEncoder` — maps a feature and the currently-selected subset to a
  fixed-length vector. The length is a property of the *encoder* (:attr:`StateEncoder.dimension`),
  not of the subset, so it is constant across exploration steps (CON-005) and the engine's
  policy-input shape never shifts.
- :class:`RewardFunction` — returns the per-step reward scalar for the current subset,
  optionally specialized to one feature-agent (``agent``).

Why this shape is forward-compatible (RISK-004). The engine reads ``dimension`` rather than
assuming a constant, so PHASE-004's Decision-Tree-structured state — of a *different* length —
conforms with no engine change. The reward carries an optional ``agent`` identity from the
start: the provisional overall reward (TASK-207) ignores it and applies one value uniformly,
while PHASE-004's personalized reward reads it to return a per-agent signal — both through the
*same* signature. The concrete minimal state (TASK-206) and overall reward (TASK-207) are built
against these interfaces; an alternate-shaped stand-in is verified to satisfy them at TASK-212.

This seam is internal to the engine and is distinct from ``harness.contract``: the engine
satisfies that external subset contract at its boundary (TASK-210) while consuming its state and
reward through this replaceable interface.

Satisfies REQ-014 (enabling — establishes the seam shape REQ-014 is delivered against). Honors
CON-005 (fixed, subset-size-independent state dimension) and mitigates RISK-004 (seam-shape churn).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Optional, Protocol, Sequence, runtime_checkable

if TYPE_CHECKING:
    # Annotation-only import keeps this load-bearing seam import-light, mirroring
    # ``harness.contract``: naming the context type must not pull its dependencies
    # into every seam consumer.
    import numpy as np

    from harness.contract import SelectionContext


@runtime_checkable
class StateEncoder(Protocol):
    """Encodes (feature, currently-selected subset) into a fixed-length state vector.

    Conformance is structural (no inheritance): any object exposing ``dimension`` and a matching
    ``encode`` satisfies the seam. The engine reads :attr:`dimension` to size its policy input and
    re-invokes :meth:`encode` each step as the selected subset grows.
    """

    @property
    def dimension(self) -> int:
        """Length of every vector :meth:`encode` returns.

        Fixed and independent of the current subset size (CON-005), so the policy-input shape is
        stable across exploration steps. A later, richer encoder may declare a different
        ``dimension``; the engine adapts because it reads this value rather than assuming a
        constant.
        """
        ...

    def encode(
        self,
        feature: int,
        selected: Sequence[int],
        context: "SelectionContext",
    ) -> "np.ndarray":
        """Encode ``feature`` against the currently-``selected`` subset into a state vector.

        Returns a one-dimensional array of length :attr:`dimension`, computed from the shared
        ``context`` (split, probe, config, seed). The length does not depend on ``len(selected)``.
        """
        ...


@runtime_checkable
class RewardFunction(Protocol):
    """Returns the per-step reward scalar for the current subset (optionally per agent).

    Conformance is structural. ``agent`` carries the forward-compatibility hedge (RISK-004): an
    overall reward ignores it and applies one value uniformly; a later personalized reward reads it
    to return a per-agent signal — both through this one signature.
    """

    def reward(
        self,
        selected: Sequence[int],
        context: "SelectionContext",
        *,
        agent: Optional[int] = None,
    ) -> float:
        """Per-step reward for the current ``selected`` subset, scored through ``context``.

        ``agent`` optionally identifies the feature-agent the reward is computed for. When ``None``
        (or ignored), the reward is applied uniformly across all agents.
        """
        ...


@runtime_checkable
class ActionAdvisor(Protocol):
    """Overrides hesitant agents' votes once per step, after voting and before the subset is
    committed.

    The interactive-advice seam (COMP-010): unlike :class:`StateEncoder` / :class:`RewardFunction`,
    which the engine reads *per agent*, the advisor is consulted *once per step over the whole
    population* — advice is defined against the participated/assertive/hesitant partition
    (COMP-001), which only exists after every agent has voted. The engine supplies the previous
    step's action vector and the current votes; the advisor returns a sparse ``{feature: action}``
    override map the engine applies to the current votes before the SELECT-union. Conformance is
    structural.

    When no advisor is injected the engine skips this seam entirely, so the no-trainer configuration
    is unchanged (the override map is the only way advice enters action selection — DEC-001
    pluggable contract). The advisor reads only non-test partitions through ``context`` (AC-007) and
    consumes randomness solely from the shared RNG (CON-003).
    """

    def advise(
        self,
        step: int,
        prior_actions: Sequence[int],
        current_actions: Sequence[int],
        context: "SelectionContext",
    ) -> Mapping[int, int]:
        """Return a ``{feature: action}`` override map for the current step's votes.

        ``step`` is the zero-based exploration step (some advisors sequence trainers by step).
        ``prior_actions`` and ``current_actions`` are per-feature action vectors (indexed by
        feature); the returned mapping carries only the features whose action the advisor overrides,
        leaving all others untouched. An empty mapping means no advice this step.
        """
        ...
