from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ExperimentConfig:
    """The deliberately small set of knobs used by the reproduction."""

    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT.parent / "dataset")
    data_version: str = "v16n"
    output_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "outputs_optimized_topology_ppo"
    )
    seed: int = 42

    reference_validation_fraction: float = 0.25
    graph_threshold: float = 0.5
    inner_cv_folds: int = 5
    audit_cv_repeats: int = 5
    cv_jobs: int = 5
    feature_budget: int = 32
    min_features: int = 4
    search_mode: str = "swap"
    max_swaps: int = 2
    swap_candidate_pool: int = 4

    episodes: int = 64
    episodes_per_update: int = 16
    greedy_rollouts: int = 1
    hidden_dim: int = 64
    actor_prior_scale: float = 2.0
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    ppo_epochs: int = 4
    minibatch_size: int = 256
    entropy_coef: float = 0.003
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.03

    correlation_penalty: float = 0.02
    sparsity_bonus: float = 0.002
    shaping_scale: float = 0.1
    terminal_reward_scale: float = 10.0
    archive_min_cv_gain: float = 0.003
    device: str = "auto"

    def validate(self, n_features: int | None = None) -> None:
        if not 0.0 <= self.ppo_graph_threshold <= 1.0:
            raise ValueError("graph_threshold must be in [0, 1]")
        if self.inner_cv_folds < 2 or self.audit_cv_repeats < 1 or self.cv_jobs < 1:
            raise ValueError("inner_cv_folds >= 2 and cv_jobs >= 1 are required")
        if self.episodes < 1 or self.episodes_per_update < 1:
            raise ValueError("episodes and episodes_per_update must be positive")
        if not 1 <= self.min_features <= self.feature_budget:
            raise ValueError("require 1 <= min_features <= feature_budget")
        if self.search_mode not in {"scratch", "swap"}:
            raise ValueError("search_mode must be 'scratch' or 'swap'")
        if self.max_swaps < 1:
            raise ValueError("max_swaps must be positive")
        if self.swap_candidate_pool < 1:
            raise ValueError("swap_candidate_pool must be positive")
        if n_features is not None and self.feature_budget > n_features:
            raise ValueError("feature_budget cannot exceed the cleaned feature count")

    def as_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["data_dir"] = str(self.data_dir)
        values["output_dir"] = str(self.output_dir)
        return values
