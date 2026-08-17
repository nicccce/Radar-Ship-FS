"""End-to-end topology-aware GNN-PPO experiment."""

from __future__ import annotations

import csv
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from .config import ExperimentConfig
from .data import DOMAIN_NAMES, RadarData, load_radar_data
from .evaluator import CVResult, SubsetEvaluator, ValidationEvaluator, evaluate_tree_on_test
from .ppo_agent import PPOAgent, RolloutBuffer
from .ppo_env import EpisodeSummary, FeatureSelectionEnv
from .ppo_graph import FeatureGraph, build_feature_graph
from .ppo_model import GraphActorCritic


@dataclass(frozen=True)
class Candidate:
    mask: np.ndarray
    cv_accuracy: float
    fold_accuracies: tuple[float, ...]
    redundancy: float
    objective: float
    origin: str

    @property
    def selected_count(self) -> int:
        return int(self.mask.sum())


def _write_json(path: Path, content: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(content, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _reference_random_states(seed: int) -> tuple[int, int, int]:
    """Match Radar-Ship-FS random draws for reordering, reward CV and final DT."""
    rng = np.random.default_rng(seed)
    validation_seed = int(rng.integers(0, 2**32))
    tree_seed = int(rng.integers(0, 2**32))
    final_tree_seed = int(rng.integers(0, 2**32))
    return validation_seed, tree_seed, final_tree_seed


def _audit_seed(seed: int) -> int:
    """Derive a deterministic CV seed never exposed through PPO rewards."""
    state = np.random.SeedSequence([seed, 0xA11D17]).generate_state(1)
    return int(state[0])


def _reference_development(
    data: RadarData,
    validation_seed: int,
    validation_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Recreate Radar-Ship-FS's train-then-validation development row order."""
    source_indices = np.arange(data.X_development.shape[0])
    fit_indices, validation_indices = train_test_split(
        source_indices,
        test_size=validation_fraction,
        random_state=validation_seed,
        stratify=data.y_development,
    )
    order = np.concatenate((fit_indices, validation_indices))
    return (
        data.X_development[fit_indices],
        data.y_development[fit_indices],
        data.X_development[validation_indices],
        data.y_development[validation_indices],
        data.X_development[order],
        data.y_development[order],
        order,
    )


def _mask_from_top_scores(scores: np.ndarray, k: int) -> np.ndarray:
    ranking = np.argsort(-np.nan_to_num(scores, nan=-np.inf), kind="stable")
    mask = np.zeros(scores.size, dtype=bool)
    mask[ranking[:k]] = True
    return mask


def _candidate(
    mask: np.ndarray,
    cv: CVResult,
    graph: FeatureGraph,
    config: ExperimentConfig,
    origin: str,
) -> Candidate:
    redundancy = graph.redundancy(mask)
    sparsity = 1.0 - float(mask.sum()) / config.feature_budget
    objective = cv.mean_accuracy - config.correlation_penalty * redundancy
    objective += config.sparsity_bonus * sparsity
    return Candidate(
        mask.copy(),
        cv.mean_accuracy,
        cv.fold_accuracies,
        redundancy,
        objective,
        origin,
    )


def _candidate_from_summary(summary: EpisodeSummary, origin: str) -> Candidate:
    return Candidate(
        summary.selected.copy(),
        summary.cv_accuracy,
        summary.fold_accuracies,
        summary.redundancy,
        summary.objective,
        origin,
    )


def _is_better(candidate: Candidate, incumbent: Candidate) -> bool:
    if candidate.objective > incumbent.objective + 1e-12:
        return True
    if abs(candidate.objective - incumbent.objective) <= 1e-12:
        return candidate.selected_count < incumbent.selected_count
    return False


def _is_archive_better(
    candidate: Candidate,
    incumbent: Candidate,
    minimum_cv_gain: float,
) -> bool:
    return candidate.cv_accuracy >= incumbent.cv_accuracy + minimum_cv_gain


def _metrics(candidate: Candidate) -> dict[str, Any]:
    return {
        "selected_count": candidate.selected_count,
        "mean_accuracy": candidate.cv_accuracy,
        "fold_accuracies": list(candidate.fold_accuracies),
        "mean_abs_correlation": candidate.redundancy,
        "objective": candidate.objective,
    }


def _selection_record(
    candidate: Candidate,
    data: RadarData,
    graph: FeatureGraph,
    config: ExperimentConfig,
    reward_evaluator: SubsetEvaluator,
    audit_evaluator: ValidationEvaluator,
    reward_baselines: dict[str, Candidate],
    audit_baselines: dict[str, Candidate],
    device: torch.device,
    elapsed_seconds: float,
    validation_seed: int,
    tree_seed: int,
    audit_seed: int,
    final_tree_seed: int,
    development_order: np.ndarray,
) -> dict[str, Any]:
    indices = np.flatnonzero(candidate.mask)
    reward_selected = _candidate(
        candidate.mask,
        reward_evaluator.score(candidate.mask),
        graph,
        config,
        candidate.origin,
    )
    return {
        "protocol": {
            "selection_rows": "all source-train rows",
            "development_order": ("reference train split followed by validation split"),
            "policy_reward_score": (f"fixed stratified {config.inner_cv_folds}-fold DecisionTree accuracy"),
            "archive_selection_score": (
                f"independent {config.audit_cv_repeats}x"
                f"{config.inner_cv_folds}-fold repeated stratified "
                "DecisionTree accuracy; never used as PPO reward"
            ),
            "source_test_role": ("sealed until this file and checkpoint are written"),
            "preprocessing_fit_scope": "source_train_only",
            "mi_warm_start": True,
            "action_space": (f"{graph.n_features} feature-node actions plus one STOP action"),
            "search_mode": config.search_mode,
            "maximum_local_swaps": (config.max_swaps if config.search_mode == "swap" else None),
            "actor_prior": "centered mutual-information rank",
            "archive_replacement_rule": (f"audit CV gain >= {config.archive_min_cv_gain:.4f}"),
        },
        "config": config.as_dict(),
        "runtime": {
            "device": str(device),
            "torch_version": torch.__version__,
            "selection_elapsed_seconds": elapsed_seconds,
        },
        "random_states": {
            "experiment_seed": config.seed,
            "reference_validation_split_seed": validation_seed,
            "reward_cv_and_decision_tree_seed": tree_seed,
            "audit_cv_and_decision_tree_seed": audit_seed,
            "final_decision_tree_seed": final_tree_seed,
        },
        "dataset": data.metadata,
        "graph": {
            "nodes": graph.n_features,
            "signed_correlation_edges_without_self_loops": graph.edge_count,
            "tree_dependency_edges_without_self_loops": (graph.dependency_edge_count),
            "absolute_pearson_threshold": graph.threshold,
            "physical_domains": list(DOMAIN_NAMES),
            "channels": [
                "signed Pearson correlation",
                "DecisionTree parent-child split dependency",
            ],
            "fit_scope": "source_train_only",
        },
        "baselines_reward_cv": {name: _metrics(value) for name, value in reward_baselines.items()},
        "baselines_audit_cv": {name: _metrics(value) for name, value in audit_baselines.items()},
        "best_candidate": {
            "origin": candidate.origin,
            "selected_clean_indices": indices.astype(int).tolist(),
            "selected_original_feature_ids": (data.original_feature_ids[indices].astype(int).tolist()),
            "selected_feature_names": [data.feature_names[index] for index in indices],
            "selected_domains": [DOMAIN_NAMES[int(data.domains[index])] for index in indices],
            "selected_count": candidate.selected_count,
            "compression_ratio": 1.0 - candidate.selected_count / graph.n_features,
            "reward_cv_dt_accuracy": reward_selected.cv_accuracy,
            "audit_cv_dt_accuracy": candidate.cv_accuracy,
            "inner_cv_dt_accuracy": candidate.cv_accuracy,
            "audit_fold_accuracies": list(candidate.fold_accuracies),
            "mean_abs_correlation": candidate.redundancy,
            "selection_objective": candidate.objective,
        },
        "cross_validation": {
            "development_row_order": development_order.astype(int).tolist(),
            "reward": {
                "evaluated_unique_subsets": reward_evaluator.evaluated_subsets,
                "fold_indices_within_source_train": reward_evaluator.fold_indices(),
            },
            "audit": {
                "evaluated_unique_subsets": audit_evaluator.evaluated_subsets,
                "fold_indices_within_source_train": audit_evaluator.fold_indices(),
            },
        },
    }


def _run_episode(
    env: FeatureSelectionEnv,
    agent: PPOAgent,
    *,
    deterministic: bool,
    buffer: RolloutBuffer | None = None,
) -> EpisodeSummary:
    observation = env.reset()
    while True:
        action, log_probability, value = agent.act(observation, deterministic=deterministic)
        reward, done, summary = env.step(action)
        if buffer is not None:
            buffer.add(
                observation,
                action,
                log_probability,
                value,
                reward,
                done,
            )
        if done:
            if summary is None:
                raise RuntimeError("terminal step did not return an episode summary")
            return summary
        observation = env.observation()


def _history_row(
    *,
    phase: str,
    episode: int,
    reward_candidate: Candidate,
    summary: EpisodeSummary,
    best: Candidate,
    audit_candidate: Candidate | None = None,
) -> dict[str, Any]:
    return {
        "phase": phase,
        "episode": episode,
        "selected_count": reward_candidate.selected_count,
        "reward_cv_dt_accuracy": reward_candidate.cv_accuracy,
        "audit_cv_dt_accuracy": ("" if audit_candidate is None else audit_candidate.cv_accuracy),
        "mean_abs_correlation": reward_candidate.redundancy,
        "reward_objective": reward_candidate.objective,
        "audit_objective": ("" if audit_candidate is None else audit_candidate.objective),
        "episode_reward": summary.total_reward,
        "best_audit_objective": best.objective,
        "best_origin": best.origin,
    }


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    started = time.perf_counter()
    _seed_everything(config.seed)
    device = _device(config.device)
    data = load_radar_data(config.data_dir, config.data_version)
    config.validate(data.n_features)

    validation_seed, tree_seed, final_tree_seed = _reference_random_states(config.seed)
    audit_seed = _audit_seed(config.seed)
    X_train_cv, y_train_cv, X_val, y_val, X_development, y_development, development_order = (
        _reference_development(
            data,
            validation_seed,
            config.reference_validation_fraction,
        )
    )
    graph = build_feature_graph(
        X_train_cv,
        y_train_cv,
        data.domains,
        threshold=config.ppo_graph_threshold,
        seed=config.seed,
        tree_seed=tree_seed,
    )
    reward_evaluator = SubsetEvaluator(
        X_train_cv,
        y_train_cv,
        seed=tree_seed,
        folds=config.inner_cv_folds,
        n_jobs=config.cv_jobs,
        row_indices=development_order[: len(X_train_cv)],
    )
    audit_evaluator = ValidationEvaluator(
        X_train_cv,
        y_train_cv,
        X_val,
        y_val,
        seed=audit_seed,
    )

    all_mask = np.ones(data.n_features, dtype=bool)
    mi_mask = _mask_from_top_scores(graph.mutual_information, config.feature_budget)
    reward_baselines = {
        "all_features": _candidate(
            all_mask,
            reward_evaluator.score(all_mask),
            graph,
            config,
            "all_features",
        ),
        "mi_kbest": _candidate(
            mi_mask,
            reward_evaluator.score(mi_mask),
            graph,
            config,
            "mi_warm_start",
        ),
    }
    audit_baselines = {
        "all_features": _candidate(
            all_mask,
            audit_evaluator.score(all_mask),
            graph,
            config,
            "all_features",
        ),
        "mi_kbest": _candidate(
            mi_mask,
            audit_evaluator.score(mi_mask),
            graph,
            config,
            "mi_warm_start",
        ),
    }
    reward_best = reward_baselines["mi_kbest"]
    best = audit_baselines["mi_kbest"]

    print(
        f"data={config.data_version} development={data.X_development.shape} "
        f"test={data.X_test.shape} corr_edges={graph.edge_count} "
        f"dependency_edges={graph.dependency_edge_count} device={device}",
        flush=True,
    )
    print(
        f"reward-CV all={reward_baselines['all_features'].cv_accuracy:.4f} "
        f"MI-{config.feature_budget}="
        f"{reward_baselines['mi_kbest'].cv_accuracy:.4f}; "
        f"audit-CV all={audit_baselines['all_features'].cv_accuracy:.4f} "
        f"MI-{config.feature_budget}="
        f"{audit_baselines['mi_kbest'].cv_accuracy:.4f}",
        flush=True,
    )

    model = GraphActorCritic(
        graph,
        config.hidden_dim,
        config.actor_prior_scale,
    )
    agent = PPOAgent(model, config, device)
    env = FeatureSelectionEnv(
        graph,
        reward_evaluator,
        config,
        baseline_objective=reward_baselines["mi_kbest"].objective,
        initial_mask=(mi_mask if config.search_mode == "swap" else None),
    )

    episode_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    completed = 0
    update_index = 0
    while completed < config.episodes:
        buffer = RolloutBuffer.empty()
        batch_best: Candidate | None = None
        batch_best_summary: EpisodeSummary | None = None
        batch_size = min(
            config.episodes_per_update,
            config.episodes - completed,
        )
        for offset in range(batch_size):
            episode = completed + offset + 1
            summary = _run_episode(
                env,
                agent,
                deterministic=False,
                buffer=buffer,
            )
            reward_current = _candidate_from_summary(summary, f"train_episode_{episode}")
            if _is_better(reward_current, reward_best):
                reward_best = reward_current
            if batch_best is None or _is_better(reward_current, batch_best):
                batch_best = reward_current
                batch_best_summary = summary
            episode_rows.append(
                _history_row(
                    phase="train",
                    episode=episode,
                    reward_candidate=reward_current,
                    summary=summary,
                    best=best,
                )
            )

        completed += batch_size
        update_index += 1
        agent.set_learning_rate(1.0 - completed / config.episodes)
        update_metrics = agent.update(buffer)

        checkpoint_summary = _run_episode(
            env,
            agent,
            deterministic=True,
        )
        checkpoint_reward = _candidate_from_summary(
            checkpoint_summary,
            f"checkpoint_{update_index}",
        )
        if batch_best is None or batch_best_summary is None:
            raise RuntimeError("PPO update did not produce a batch candidate")

        proposal_reward = checkpoint_reward
        proposal_summary = checkpoint_summary
        if _is_better(batch_best, proposal_reward):
            proposal_reward = batch_best
            proposal_summary = batch_best_summary
        proposal_audit = _candidate(
            proposal_reward.mask,
            audit_evaluator.score(proposal_reward.mask),
            graph,
            config,
            proposal_reward.origin,
        )
        if _is_archive_better(
            proposal_audit,
            best,
            config.archive_min_cv_gain,
        ):
            best = proposal_audit

        episode_rows.append(
            _history_row(
                phase="checkpoint",
                episode=update_index,
                reward_candidate=checkpoint_reward,
                summary=checkpoint_summary,
                best=best,
            )
        )
        episode_rows.append(
            _history_row(
                phase="proposal",
                episode=update_index,
                reward_candidate=proposal_reward,
                summary=proposal_summary,
                best=best,
                audit_candidate=proposal_audit,
            )
        )
        update_rows.append(
            {
                "update": update_index,
                "episodes_completed": completed,
                **update_metrics,
                "checkpoint_reward_cv": checkpoint_reward.cv_accuracy,
                "proposal_origin": proposal_reward.origin,
                "proposal_reward_cv": proposal_reward.cv_accuracy,
                "proposal_audit_cv": proposal_audit.cv_accuracy,
                "archive_audit_cv": best.cv_accuracy,
                "archive_selected_count": best.selected_count,
            }
        )
        print(
            f"episodes={completed:>3}/{config.episodes} "
            f"checkpoint_reward_cv={checkpoint_reward.cv_accuracy:.4f} "
            f"proposal_reward_cv={proposal_reward.cv_accuracy:.4f} "
            f"proposal_audit_cv={proposal_audit.cv_accuracy:.4f} "
            f"archive_cv={best.cv_accuracy:.4f} k={best.selected_count:>2} "
            f"entropy={update_metrics['entropy']:.3f} "
            f"kl={update_metrics['approximate_kl']:.4f}",
            flush=True,
        )

    for rollout in range(1, config.greedy_rollouts + 1):
        summary = _run_episode(env, agent, deterministic=True)
        reward_current = _candidate_from_summary(
            summary,
            f"final_greedy_{rollout}",
        )
        audit_current = _candidate(
            summary.selected,
            audit_evaluator.score(summary.selected),
            graph,
            config,
            f"final_greedy_{rollout}",
        )
        if _is_archive_better(
            audit_current,
            best,
            config.archive_min_cv_gain,
        ):
            best = audit_current
        episode_rows.append(
            _history_row(
                phase="final_greedy",
                episode=rollout,
                reward_candidate=reward_current,
                summary=summary,
                best=best,
                audit_candidate=audit_current,
            )
        )

    run_dir = config.output_dir / f"seed-{config.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    selection_elapsed = time.perf_counter() - started
    selection = _selection_record(
        best,
        data,
        graph,
        config,
        reward_evaluator,
        audit_evaluator,
        reward_baselines,
        audit_baselines,
        device,
        selection_elapsed,
        validation_seed,
        tree_seed,
        audit_seed,
        final_tree_seed,
        development_order,
    )
    _write_json(run_dir / "selection.json", selection)
    _write_csv(run_dir / "training.csv", episode_rows)
    _write_csv(run_dir / "ppo_updates.csv", update_rows)
    np.savez_compressed(
        run_dir / "feature_graph.npz",
        correlation_adjacency=graph.correlation_adjacency,
        dependency_adjacency=graph.dependency_adjacency,
        signed_correlation=graph.signed_correlation,
        static_node_features=graph.static_node_features,
        original_feature_ids=data.original_feature_ids,
        domains=data.domains,
    )
    torch.save(
        {
            "model_state_dict": agent.ppo_model.state_dict(),
            "config": config.as_dict(),
            "selected_mask": best.mask,
            "best_origin": best.origin,
        },
        run_dir / "checkpoint.pt",
    )
    print(f"selection sealed: {run_dir / 'selection.json'}", flush=True)

    final_methods = {
        "all_features": all_mask,
        "mi_kbest": mi_mask,
        "gnn_ppo_selected": best.mask,
    }
    final_rows: dict[str, dict[str, float | int]] = {}
    for name, mask in final_methods.items():
        final_rows[name] = evaluate_tree_on_test(
            X_development,
            y_development,
            data.X_test,
            data.y_test,
            mask,
            seed=final_tree_seed,
        )
    selected_metrics = final_rows["gnn_ppo_selected"]
    final = {
        "protocol": {
            "selection_artifact_written_before_test_evaluation": True,
            "final_model": "DecisionTreeClassifier",
            "final_fit_rows": int(data.X_development.shape[0]),
            "source_test_rows": int(data.X_test.shape[0]),
        },
        "seed": config.seed,
        "best_candidate_origin": best.origin,
        "methods": final_rows,
        "gnn_ppo_delta_test_accuracy": {
            "vs_all_features": float(
                selected_metrics["test_accuracy"] - final_rows["all_features"]["test_accuracy"]
            ),
            "vs_mi_kbest": float(selected_metrics["test_accuracy"] - final_rows["mi_kbest"]["test_accuracy"]),
        },
        "total_elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(run_dir / "final_results.json", final)
    print(
        "final DT test: "
        + ", ".join(
            f"{name}={metrics['test_accuracy']:.4f} (k={metrics['selected_count']})"
            for name, metrics in final_rows.items()
        ),
        flush=True,
    )
    return final
