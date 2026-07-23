"""Configuration surface (COMP-025).

Exposes every parameter the reference method leaves unspecified as typed configuration with recorded
defaults, and provides a single effective-configuration view (one immutable ``IrfsConfig``
instance). Defaults are anchored to the recorded assumptions ASM-001–ASM-008; the seed parameter
feeds the seeded RNG (COMP-026), which is intentionally decoupled from this module (both have
``Dependencies: none``).

Satisfies COMP-025 -> REQ-020.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from typing import Any, Optional

# Reference leaves the seed unspecified; recorded as a configurable default (REQ-020).
# A run sweeps this list of seeds; a single-seed run is just a one-element list.
DEFAULT_SEEDS = (5, 8)


@dataclass(frozen=True)
class IrfsConfig:
    """The single effective-configuration view.

    Frozen so the effective configuration cannot be mutated after loading; use :func:`load_config`
    to produce a view with overrides applied.
    """

    # --- Determinism (each seed feeds COMP-026; no ASM — reference-unspecified) ---
    # The run sweeps every seed in this list; a single-seed run is a one-element list. Each seed
    # drives one independent leakage-safe run with its own shared RNG (CON-003), written to its own
    # ``seed-<n>/`` artifact folder. ``seeds[0]`` is the primary seed used when a single run is wired
    # without an explicit seed override.
    seeds: tuple[int, ...] = DEFAULT_SEEDS

    # --- Dataset selection (COMP-001 consumes this; config names the dataset) ---
    dataset: str = "wdbc"

    # --- Local dataset root for file-based datasets (e.g. Parkinson's). sklearn-bunch datasets ignore it;
    # the file-based loader resolves its files under <data_dir>/. Kept out of version control
    # (``data/`` is gitignored), so large raw datasets live locally rather than in the repo.
    data_dir: str = "data"

    # --- Data split ratios (ASM-001): 80/20 train/test, validation carved from train ---
    test_fraction: float = 0.2
    validation_fraction: float = 0.2  # fraction of the post-test training pool

    # --- Classical baselines (reference-unspecified; recorded default) ---
    # L1/LASSO inverse-regularization strength C for the L1-penalized model (COMP-008,
    # sklearn LogisticRegression solver='liblinear'). Smaller C = stronger penalty = fewer
    # surviving (non-zero-coefficient) features; the subset size is variable, not half-count.
    l1_C: float = 1.0

    # --- RL policy hyperparameters (ASM-002) ---
    discount: float = 0.9
    # Probability of taking the greedy (exploit) action in ε-greedy selection; the agent acts
    # randomly with probability (1 − this). At 0.9 the population exploits 90% / explores 10%.
    exploitation_probability: float = 0.9
    mini_batch_size: int = 16
    learning_rate: float = 0.01
    hidden_layer_sizes: tuple[int, ...] = (128, 128)
    activation: str = "relu"

    # --- Exploration loop (ASM-004): step budget the reinforced engine runs for (REQ-007).
    # Provisional default; tuned for WDBC convergence in TASK-211 (Q-001). Disclosed in the
    # effective-configuration view (REQ-012 / AC-009).
    exploration_step_budget: int = 250

    # --- State representation (ASM-003 aggregation/per-node; ASM-005 pooling) ---
    neighbor_global_mix: float = 0.5  # even neighbor-vs-global mix (ASM-003)
    per_node_features: str = "summary_statistics"  # ASM-003
    state_pooling: str = "dt_importance"  # ASM-005

    # --- Trainable GCN encoder (COMP-001 / TASK-002): output state width and learned-layer depth.
    # Reference-unspecified; recorded config defaults (REQ-020 convention; Q-001 gap-fills, depth/width
    # tuning is a deferred Non-Goal). Disclosed in the effective-configuration view.
    gcn_hidden_dim: int = 4  # learned output state width; matches the fixed encoder's 4 for parity
    gcn_layers: int = 1  # single learned graph-convolution layer (depth tuning deferred, Q-001)

    # --- State-encoder selection (COMP-004 / TASK-003): which encoder the run drops in behind the
    # engine's unchanged StateEncoder seam. "fixed" (default) keeps PHASE-004's fixed-weight
    # TreeStructuredStateEncoder, byte-identical to the PHASE-004 baseline (AC-007); it is the
    # reproduction substrate, not a method the reference describes. "trained_gcn" selects the learnable
    # TrainableGCNEncoder (TASK-002) — the GCN-over-correlation-graph state the reference uses — and
    # registers its parameters for the joint learner (TASK-005). The choice sits BEHIND the seam
    # (DEC-005 / CON-R-002): the engine, subset contract, and orchestrator are untouched. Disclosed in
    # the effective-configuration view (REQ-020).
    state_encoder: str = "fixed"  # "fixed" (default) | "trained_gcn"

    # --- Reward (ASM-004): headline full-IRFS scheme + correlation-penalty weight beta ---
    reward_scheme: str = "dt_importance"
    correlation_penalty_weight: float = 1.0  # beta; configurable default
    # Optional soft cardinality constraint:
    #   lambda * max(0, (|S| - feature_budget) / feature_budget)
    # Disabled by default so historical no-budget selection behavior remains unchanged.
    feature_budget: Optional[int] = None
    over_budget_penalty_weight: float = 0.0  # lambda

    # --- Hybrid teaching schedule (COMP-004): the two switch points that sequence the
    # relevance trainer, then the DT-importance trainer, then withdraw guidance (REQ-004).
    # Reference fixes the hybrid shape but not the boundaries; recorded as a fidelity gap-fill
    # (thirds of the default 100-step budget). Require 0 <= switch <= withdraw.
    hybrid_switch_step: int = 33  # relevance runs over steps [0, switch); DT-importance begins here
    hybrid_withdraw_step: int = 66  # DT-importance runs over [switch, withdraw); guidance withdrawn after

    # --- Metrics window (ASM-006): None = full exploration run ---
    metric_window: Optional[tuple[int, int]] = None

    # --- Trainer stability (ASM-008): no separate target network by default ---
    use_target_network: bool = False

    # --- Per-agent credit mode (cause-C investigation; default = faithful reference) ---
    # Reshapes how the overall reward becomes each agent's learning signal. "reference" (default)
    # is the reference scheme r_i = weight_i·(Acc−βR) with deselected agents pinned to zero — the
    # honest reproduction. "symmetric" gives every agent (selected and deselected) the full overall
    # reward (the diagnostic control: the symmetric, full-magnitude signal the minimal engine
    # converged with). "marginal" is reserved for the counterfactual per-agent credit (Path C),
    # not yet implemented. Any non-"reference" value is an opt-in deviation, recorded as a fidelity
    # note (RISK-001 / COMP-024).
    per_agent_credit: str = "reference"


def load_config(overrides: Optional[Mapping[str, Any]] = None) -> IrfsConfig:
    """Return the single effective configuration with ``overrides`` applied.

    Starting from the recorded defaults, each provided key replaces exactly that field and leaves
    all others at their defaults. Unknown keys raise ``ValueError`` rather than being silently
    ignored.
    """
    base = IrfsConfig()
    if not overrides:
        return base
    known = {f.name for f in fields(IrfsConfig)}
    unknown = set(overrides) - known
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    resolved = dict(overrides)
    # Accept a list/iterable of seeds and store the immutable tuple the frozen view requires.
    if "seeds" in resolved:
        resolved["seeds"] = tuple(int(s) for s in resolved["seeds"])
        if not resolved["seeds"]:
            raise ValueError("seeds must contain at least one seed")
    if resolved.get("feature_budget") is not None:
        resolved["feature_budget"] = int(resolved["feature_budget"])
        if resolved["feature_budget"] <= 0:
            raise ValueError("feature_budget must be positive")
    if float(resolved.get("over_budget_penalty_weight", base.over_budget_penalty_weight)) < 0.0:
        raise ValueError("over_budget_penalty_weight must be non-negative")
    return replace(base, **resolved)
