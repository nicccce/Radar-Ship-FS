import time

import numpy as np
import torch

from harness.contract import StepRecord, make_selection
from radar_ship_fs.selection.types import StableTrainingResult, TrainingMetrics

from .config import ExperimentConfig
from .evaluator import CVResult, SubsetEvaluator, ValidationEvaluator
from .experiment import (
    _candidate,
    _candidate_from_summary,
    _is_archive_better,
    _is_better,
    _run_episode,
    _seed_everything,
)
from .ppo_agent import PPOAgent, RolloutBuffer
from .ppo_env import FeatureSelectionEnv
from .ppo_graph import build_feature_graph
from .ppo_model import GraphActorCritic


def _random_initial_mask(n_features: int, feature_budget: int, seed: int) -> np.ndarray:
    if not 1 <= feature_budget <= n_features:
        raise ValueError("require 1 <= feature_budget <= n_features")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x50504F]))
    mask = np.zeros(n_features, dtype=bool)
    mask[rng.choice(n_features, size=feature_budget, replace=False)] = True
    return mask


class _SharedProbeEvaluator:
    """Mask adapter over the exact fixed-fold scorer shared with DQN/classical methods."""

    def __init__(self, context) -> None:
        self.context = context
        self._seen: set[bytes] = set()

    @property
    def evaluated_subsets(self) -> int:
        return len(self._seen)

    def score(self, mask: np.ndarray) -> CVResult:
        mask = np.asarray(mask, dtype=bool)
        indices = tuple(int(value) for value in np.flatnonzero(mask))
        if not indices:
            raise ValueError("cannot evaluate an empty feature subset")
        self._seen.add(np.packbits(mask).tobytes())
        result = self.context.probe.probe(indices, self.context.split.validation)
        folds = self.context.probe.fold_accuracies(
            indices,
            self.context.split.validation,
        )
        return CVResult(float(result.accuracy), tuple(float(value) for value in folds))


def _accuracy_archive_better(candidate, incumbent) -> bool:
    """Match the DQN archive: accuracy first, then fewer features on exact ties."""
    gain = candidate.cv_accuracy - incumbent.cv_accuracy
    if gain > 1e-12:
        return True
    return abs(gain) <= 1e-12 and candidate.selected_count < incumbent.selected_count


def run_ppo_session(
    *,
    context,
    config,  # This is spec
    seed: int,
    method_name: str,
    artifacts,
    identity: dict,
) -> StableTrainingResult:
    _seed_everything(seed)
    spec = config

    ppo_config = ExperimentConfig(
        seed=seed,
        graph_threshold=0.8,
        inner_cv_folds=spec.dataset.inner_cv_folds,
        audit_cv_repeats=5,
        cv_jobs=4,
        feature_budget=spec.ppo.feature_budget,
        search_mode="swap",
        max_swaps=spec.ppo.max_swaps,
        swap_candidate_pool=spec.ppo.swap_candidate_pool,
        swap_exploration_pool=spec.ppo.swap_exploration_pool,
        episodes=spec.ppo.episodes,
        episodes_per_update=spec.ppo.episodes_per_update,
        greedy_rollouts=spec.ppo.greedy_rollouts,
        hidden_dim=spec.ppo.hidden_dim,
        actor_prior_scale=spec.ppo.actor_prior_scale,
        learning_rate=spec.ppo.learning_rate,
        gamma=spec.ppo.gamma,
        gae_lambda=spec.ppo.gae_lambda,
        clip_ratio=spec.ppo.clip_ratio,
        ppo_epochs=spec.ppo.ppo_epochs,
        minibatch_size=spec.ppo.minibatch_size,
        entropy_coef=spec.ppo.entropy_coef,
        value_coef=spec.ppo.value_coef,
        max_grad_norm=spec.ppo.max_grad_norm,
        target_kl=spec.ppo.target_kl,
        correlation_penalty=spec.ppo.correlation_penalty,
        sparsity_bonus=spec.ppo.sparsity_bonus,
        feature_id_seed=spec.ppo.feature_id_seed,
        feature_id_node_feature=spec.ppo.feature_id_node_feature,
        feature_id_reward_weight=spec.ppo.feature_id_reward_weight,
        archive_accuracy_tolerance=spec.ppo.archive_accuracy_tolerance,
        shaping_scale=spec.ppo.shaping_scale,
        terminal_reward_scale=spec.ppo.terminal_reward_scale,
        archive_min_cv_gain=spec.ppo.archive_min_cv_gain,
    )

    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_source_train = context.split.train.X
    y_source_train = context.split.train.y
    domains = np.zeros(X_source_train.shape[1], dtype=int)

    if spec.ppo.evaluation_protocol == "shared_inner_cv":
        X_search = X_source_train
        y_search = y_source_train
        reward_evaluator = _SharedProbeEvaluator(context)
        audit_evaluator = reward_evaluator
    else:
        from sklearn.model_selection import train_test_split

        X_search, X_val, y_search, y_val = train_test_split(
            X_source_train,
            y_source_train,
            test_size=0.2,
            stratify=y_source_train,
            random_state=seed,
        )
        reward_evaluator = SubsetEvaluator(
            X_search,
            y_search,
            seed=seed,
            folds=spec.dataset.inner_cv_folds,
            n_jobs=4,
        )
        audit_evaluator = ValidationEvaluator(
            X_search,
            y_search,
            X_val,
            y_val,
            seed=seed,
        )

    graph = build_feature_graph(
        X_search,
        y_search,
        domains,
        threshold=0.8,
        seed=seed,
        tree_seed=seed,
        feature_id_seed=ppo_config.feature_id_seed,
        include_feature_id_node_feature=ppo_config.feature_id_node_feature,
    )

    if spec.ppo.initialization == "random":
        initial_mask = _random_initial_mask(
            X_search.shape[1],
            ppo_config.feature_budget,
            seed,
        )
        initial_origin = "random_start"
    else:
        from sklearn.feature_selection import mutual_info_classif

        random_state = int(context.rng.numpy.integers(0, 2**32))
        relevance = mutual_info_classif(X_search, y_search, random_state=random_state)
        ranked_features = np.argsort(-relevance, kind="stable").tolist()

        initial_mask = np.zeros(X_search.shape[1], dtype=bool)
        best_acc = -1.0
        for feature in ranked_features:
            test_mask = initial_mask.copy()
            test_mask[feature] = True
            accuracy = reward_evaluator.score(test_mask).mean_accuracy
            if accuracy > best_acc:
                initial_mask = test_mask
                best_acc = accuracy

        current_count = int(initial_mask.sum())
        if current_count < ppo_config.feature_budget:
            for feature in ranked_features:
                if not initial_mask[feature]:
                    initial_mask[feature] = True
                    current_count += 1
                    if current_count == ppo_config.feature_budget:
                        break
        elif current_count > ppo_config.feature_budget:
            removed = 0
            for feature in reversed(ranked_features):
                if initial_mask[feature]:
                    initial_mask[feature] = False
                    removed += 1
                    if current_count - removed == ppo_config.feature_budget:
                        break
        initial_origin = (
            "mi_ordered_accept_then_pad_to_k"
            if spec.ppo.initialization == "mi_ordered_accept"
            else "mi_warm_start"
        )

    reward_best = _candidate(
        initial_mask,
        reward_evaluator.score(initial_mask),
        graph,
        ppo_config,
        initial_origin,
    )
    best = _candidate(
        initial_mask,
        audit_evaluator.score(initial_mask),
        graph,
        ppo_config,
        initial_origin,
    )
    initial_accuracy = reward_best.cv_accuracy

    model = GraphActorCritic(
        graph,
        ppo_config.hidden_dim,
        ppo_config.actor_prior_scale,
    )
    agent = PPOAgent(model, ppo_config, device)
    env = FeatureSelectionEnv(
        graph,
        reward_evaluator,
        ppo_config,
        baseline_objective=reward_best.objective,
        initial_mask=initial_mask,
    )

    completed = 0
    update_index = 0
    metrics = []

    while completed < spec.ppo.episodes:
        buffer = RolloutBuffer.empty()
        batch_best = None
        batch_size = min(spec.ppo.episodes_per_update, spec.ppo.episodes - completed)
        for offset in range(batch_size):
            episode = completed + offset + 1
            summary = _run_episode(env, agent, deterministic=False, buffer=buffer)
            reward_current = _candidate_from_summary(summary, f"train_episode_{episode}")
            if _is_better(reward_current, reward_best):
                reward_best = reward_current
            if batch_best is None or _is_better(reward_current, batch_best):
                batch_best = reward_current

        completed += batch_size
        update_index += 1
        agent.set_learning_rate(1.0 - completed / spec.ppo.episodes)
        update_metrics = agent.update(buffer)

        checkpoint_summary = _run_episode(env, agent, deterministic=True)
        checkpoint_reward = _candidate_from_summary(checkpoint_summary, f"checkpoint_{update_index}")

        proposal_reward = checkpoint_reward
        if batch_best and _is_better(batch_best, proposal_reward):
            proposal_reward = batch_best

        proposal_audit = _candidate(
            proposal_reward.mask,
            audit_evaluator.score(proposal_reward.mask),
            graph,
            ppo_config,
            proposal_reward.origin,
        )
        archive_better = (
            _accuracy_archive_better(proposal_audit, best)
            if spec.ppo.evaluation_protocol == "shared_inner_cv"
            else _is_archive_better(
                proposal_audit,
                best,
                ppo_config.archive_min_cv_gain,
                ppo_config.archive_accuracy_tolerance,
            )
        )
        if archive_better:
            best = proposal_audit

        # Log metric for runner
        elapsed = time.perf_counter() - started
        metrics.append(
            TrainingMetrics(
                step=update_index,
                subset=tuple(np.flatnonzero(best.mask)),
                subset_size=best.selected_count,
                accuracy=best.cv_accuracy,
                best_accuracy=best.cv_accuracy,
                epsilon=0.0,
                proposed_select_count=0,
                transition_applied=True,
                reward_min=0.0,
                reward_mean=0.0,
                reward_max=0.0,
                replay_size=0,
                update_performed=True,
                loss=update_metrics["policy_loss"],
                td_error_mean=0.0,
                td_error_max=0.0,
                q_mean=0.0,
                q_std=0.0,
                q_max=0.0,
                target_q_mean=0.0,
                gradient_norm=0.0,
                target_synced=False,
                advisor_override_count=0,
                elapsed_seconds=elapsed,
            )
        )

    for rollout in range(1, ppo_config.greedy_rollouts + 1):
        summary = _run_episode(env, agent, deterministic=True)
        audit_current = _candidate(
            summary.selected,
            audit_evaluator.score(summary.selected),
            graph,
            ppo_config,
            f"final_greedy_{rollout}",
        )
        archive_better = (
            _accuracy_archive_better(audit_current, best)
            if spec.ppo.evaluation_protocol == "shared_inner_cv"
            else _is_archive_better(
                audit_current,
                best,
                ppo_config.archive_min_cv_gain,
                ppo_config.archive_accuracy_tolerance,
            )
        )
        if archive_better:
            best = audit_current

    best_indices = tuple(np.flatnonzero(best.mask))
    per_step = tuple(StepRecord(subset=m.subset, accuracy=m.accuracy) for m in metrics)
    selection = make_selection(best_indices, per_step=per_step)

    return StableTrainingResult(
        selection=selection,
        metrics=tuple(metrics),
        initial_subset=tuple(np.flatnonzero(initial_mask)),
        initial_accuracy=initial_accuracy,
        learner_updates=update_index,
        rejected_transitions=0,
    )
