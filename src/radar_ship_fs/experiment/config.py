"""Strict TOML configuration for reproducible feature-selection experiments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Sequence, TypeVar

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on the Python 3.10 experiment image
    import tomli as tomllib

from config import IrfsConfig, load_config

_T = TypeVar("_T")


def _strict_dataclass(cls: type[_T], values: Mapping[str, Any], section: str) -> _T:
    known = {field.name for field in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown fields in [{section}]: {sorted(unknown)}")
    return cls(**values)


@dataclass(frozen=True)
class DatasetSpec:
    """Dataset identity and deterministic stage-2 context settings."""

    name: str
    version: str
    data_dir: str = "../dataset"
    seeds: tuple[int, ...] = (42, 43, 44, 45, 46)
    validation_fraction: float = 0.25
    inner_cv_folds: int = 5
    expected_clean_features: int | None = None


@dataclass(frozen=True)
class TrainingSpec:
    """Complete stable-DQN and existing IRFS feedback configuration."""

    steps: int = 250
    discount: float = 0.9
    learning_rate: float = 3e-4
    batch_size: int = 32
    replay_capacity: int = 2048
    warmup_steps: int = 32
    target_sync_interval: int = 25
    gradient_clip_norm: float = 10.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_fraction: float = 0.7
    checkpoint_interval: int = 25
    hidden_layer_sizes: tuple[int, ...] = (128, 128)
    activation: str = "relu"
    correlation_penalty_weight: float = 1.0
    feature_budget: int | None = None
    over_budget_penalty_weight: float = 0.0
    reward_scheme: str = "dt_importance"
    per_agent_credit: str = "reference"
    neighbor_global_mix: float = 0.5
    per_node_features: str = "summary_statistics"
    state_pooling: str = "dt_importance"
    gcn_hidden_dim: int = 4
    gcn_layers: int = 1
    hybrid_switch_step: int = 83
    hybrid_withdraw_step: int = 166


@dataclass(frozen=True)
class MethodSpec:
    """One named method in the experiment matrix."""

    name: str
    encoder: str
    advisor: str | None = None
    reward: str | None = None
    enabled: bool = True


@dataclass(frozen=True)
class OutputSpec:
    """Artifact location and default resumption policy."""

    root: str
    resume: bool = True


@dataclass(frozen=True)
class ExperimentSpec:
    """Immutable, validated experiment specification."""

    schema_version: int
    algorithm_version: str
    dataset: DatasetSpec
    training: TrainingSpec
    methods: tuple[MethodSpec, ...]
    output: OutputSpec

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"unsupported schema_version {self.schema_version}; expected 1")
        if self.algorithm_version not in {"legacy_v1", "stable_v1"}:
            raise ValueError("algorithm_version must be 'legacy_v1' or 'stable_v1'")
        if not self.dataset.name or not self.dataset.version:
            raise ValueError("dataset.name and dataset.version must be non-empty")
        if not self.dataset.seeds:
            raise ValueError("dataset.seeds must contain at least one seed")
        if self.dataset.inner_cv_folds < 2:
            raise ValueError("dataset.inner_cv_folds must be at least 2")
        if not 0.0 < self.dataset.validation_fraction < 1.0:
            raise ValueError("dataset.validation_fraction must be in (0, 1)")

        cfg = self.training
        positive = {
            "steps": cfg.steps,
            "learning_rate": cfg.learning_rate,
            "batch_size": cfg.batch_size,
            "replay_capacity": cfg.replay_capacity,
            "warmup_steps": cfg.warmup_steps,
            "target_sync_interval": cfg.target_sync_interval,
            "gradient_clip_norm": cfg.gradient_clip_norm,
            "checkpoint_interval": cfg.checkpoint_interval,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"training values must be positive: {invalid}")
        if not 0.0 <= cfg.discount < 1.0:
            raise ValueError("training.discount must be in [0, 1)")
        if not 0.0 <= cfg.epsilon_end <= cfg.epsilon_start <= 1.0:
            raise ValueError("epsilon values must satisfy 0 <= end <= start <= 1")
        if not 0.0 < cfg.epsilon_decay_fraction <= 1.0:
            raise ValueError("epsilon_decay_fraction must be in (0, 1]")
        if cfg.batch_size > cfg.replay_capacity:
            raise ValueError("batch_size cannot exceed replay_capacity")
        if cfg.warmup_steps > cfg.replay_capacity:
            raise ValueError("warmup_steps cannot exceed replay_capacity")
        if cfg.warmup_steps < cfg.batch_size:
            raise ValueError("warmup_steps cannot be smaller than batch_size")
        if not cfg.hidden_layer_sizes or any(width <= 0 for width in cfg.hidden_layer_sizes):
            raise ValueError("hidden_layer_sizes must contain only positive widths")
        if cfg.correlation_penalty_weight < 0.0:
            raise ValueError("correlation_penalty_weight must be non-negative")
        if not 0.0 <= cfg.neighbor_global_mix <= 1.0:
            raise ValueError("neighbor_global_mix must be in [0, 1]")
        if cfg.per_node_features != "summary_statistics":
            raise ValueError("per_node_features must be summary_statistics")
        if cfg.state_pooling not in {"dt_importance", "average"}:
            raise ValueError("state_pooling must be dt_importance or average")
        if cfg.gcn_hidden_dim <= 0 or cfg.gcn_layers <= 0:
            raise ValueError("GCN dimensions and layer count must be positive")
        if not 0 <= cfg.hybrid_switch_step <= cfg.hybrid_withdraw_step <= cfg.steps:
            raise ValueError("hybrid boundaries must satisfy 0 <= switch <= withdraw <= steps")
        if cfg.activation not in {"relu", "tanh", "logistic"}:
            raise ValueError("activation must be relu, tanh, or logistic")
        if cfg.reward_scheme not in {"dt_importance", "frequency"}:
            raise ValueError("reward_scheme must be dt_importance or frequency")
        if cfg.per_agent_credit not in {"reference", "symmetric"}:
            if cfg.per_agent_credit == "marginal":
                raise ValueError("per_agent_credit='marginal' is reserved but not implemented")
            raise ValueError("per_agent_credit must be reference or symmetric")
        if cfg.feature_budget is not None and cfg.feature_budget <= 0:
            raise ValueError("feature_budget must be positive")
        if cfg.over_budget_penalty_weight < 0.0:
            raise ValueError("over_budget_penalty_weight must be non-negative")

        enabled = [method for method in self.methods if method.enabled]
        if not enabled:
            raise ValueError("at least one method must be enabled")
        names = [method.name for method in enabled]
        if len(names) != len(set(names)):
            raise ValueError("enabled method names must be unique")
        allowed_encoders = {"minimal", "fixed", "trained_gcn"}
        for method in enabled:
            if method.encoder not in allowed_encoders:
                raise ValueError(
                    f"method {method.name!r} has unknown encoder {method.encoder!r}; "
                    f"expected one of {sorted(allowed_encoders)}"
                )
            if method.advisor not in {None, "none", "hybrid", "relevance", "dt_importance"}:
                raise ValueError(f"method {method.name!r} has unsupported advisor {method.advisor!r}")
            if method.reward not in {None, "uniform", "personalized"}:
                raise ValueError(f"method {method.name!r} has unsupported reward {method.reward!r}")

        if not self.output.root:
            raise ValueError("output.root must be non-empty")

    @property
    def enabled_methods(self) -> tuple[MethodSpec, ...]:
        return tuple(method for method in self.methods if method.enabled)

    def canonical_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(asdict(self), sort_keys=True))

    def canonical_json(self) -> str:
        return json.dumps(self.canonical_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def irfs_config(self, method: MethodSpec) -> IrfsConfig:
        """Translate the stable experiment schema to the existing data/feedback context."""
        encoder = "fixed" if method.encoder == "minimal" else method.encoder
        cfg = self.training
        return load_config(
            {
                "dataset": self.dataset.name,
                "data_dir": self.dataset.data_dir,
                "radar_ship_version": self.dataset.version,
                "seeds": self.dataset.seeds,
                "validation_fraction": self.dataset.validation_fraction,
                "exploration_step_budget": cfg.steps,
                "discount": cfg.discount,
                "learning_rate": cfg.learning_rate,
                "mini_batch_size": cfg.batch_size,
                "hidden_layer_sizes": cfg.hidden_layer_sizes,
                "activation": cfg.activation,
                "correlation_penalty_weight": cfg.correlation_penalty_weight,
                "feature_budget": cfg.feature_budget,
                "over_budget_penalty_weight": cfg.over_budget_penalty_weight,
                "reward_scheme": cfg.reward_scheme,
                "per_agent_credit": cfg.per_agent_credit,
                "neighbor_global_mix": cfg.neighbor_global_mix,
                "per_node_features": cfg.per_node_features,
                "state_pooling": cfg.state_pooling,
                "gcn_hidden_dim": cfg.gcn_hidden_dim,
                "gcn_layers": cfg.gcn_layers,
                "hybrid_switch_step": cfg.hybrid_switch_step,
                "hybrid_withdraw_step": cfg.hybrid_withdraw_step,
                "state_encoder": encoder,
            }
        )


def _as_int_tuple(values: Sequence[Any], name: str) -> tuple[int, ...]:
    try:
        return tuple(int(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer array") from exc


def load_experiment_spec(path: str | Path) -> ExperimentSpec:
    """Load and strictly validate one TOML experiment specification."""
    source = Path(path)
    with source.open("rb") as handle:
        raw = tomllib.load(handle)
    top_known = {"schema_version", "algorithm_version", "dataset", "training", "methods", "output"}
    unknown = set(raw) - top_known
    if unknown:
        raise ValueError(f"unknown top-level experiment fields: {sorted(unknown)}")
    missing = top_known - set(raw)
    if missing:
        raise ValueError(f"missing top-level experiment fields: {sorted(missing)}")

    dataset_values = dict(raw["dataset"])
    if "seeds" in dataset_values:
        dataset_values["seeds"] = _as_int_tuple(dataset_values["seeds"], "dataset.seeds")
    training_values = dict(raw["training"])
    if "hidden_layer_sizes" in training_values:
        training_values["hidden_layer_sizes"] = _as_int_tuple(
            training_values["hidden_layer_sizes"], "training.hidden_layer_sizes"
        )
    methods_raw = raw["methods"]
    if not isinstance(methods_raw, list):
        raise ValueError("[[methods]] must be an array of tables")

    spec = ExperimentSpec(
        schema_version=int(raw["schema_version"]),
        algorithm_version=str(raw["algorithm_version"]),
        dataset=_strict_dataclass(DatasetSpec, dataset_values, "dataset"),
        training=_strict_dataclass(TrainingSpec, training_values, "training"),
        methods=tuple(_strict_dataclass(MethodSpec, item, "methods") for item in methods_raw),
        output=_strict_dataclass(OutputSpec, raw["output"], "output"),
    )
    spec.validate()
    return spec
