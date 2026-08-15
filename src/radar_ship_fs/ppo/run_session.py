import time
from typing import Any
import numpy as np
import torch
from .config import ExperimentConfig
from .ppo_env import FeatureSelectionEnv
from .ppo_graph import build_feature_graph
from .ppo_model import GraphActorCritic
from .ppo_agent import PPOAgent, RolloutBuffer
from .experiment import _candidate, _candidate_from_summary, _is_better, _is_archive_better, _run_episode
from radar_ship_fs.selection.types import StableTrainingResult, TrainingMetrics
from harness.contract import make_selection, StepRecord
from .evaluator import SubsetEvaluator

def run_ppo_session(
    *,
    context,
    config,  # This is spec
    seed: int,
    method_name: str,
    artifacts,
    identity: dict,
) -> StableTrainingResult:
    spec = config
    
    ppo_config = ExperimentConfig(
        seed=seed,
        graph_threshold=0.8,
        inner_cv_folds=spec.dataset.inner_cv_folds,
        audit_cv_repeats=5,
        cv_jobs=4,
        feature_budget=spec.ppo.feature_budget,
        search_mode="swap",
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
        shaping_scale=spec.ppo.shaping_scale,
        terminal_reward_scale=spec.ppo.terminal_reward_scale,
        archive_min_cv_gain=0.001,
    )

    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train = context.split.train.X
    y_train = context.split.train.y
    domains = np.zeros(X_train.shape[1], dtype=int)
    
    # We use validation_seed and tree_seed from the provided rng if possible, or just seed
    graph = build_feature_graph(
        X_train,
        y_train,
        domains,
        threshold=0.8,
        seed=seed,
        tree_seed=seed,
    )
    
    reward_evaluator = SubsetEvaluator(
        X_train,
        y_train,
        seed=seed,
        folds=spec.dataset.inner_cv_folds,
        n_jobs=4,
    )
    
    audit_evaluator = SubsetEvaluator(
        X_train,
        y_train,
        seed=seed,
        folds=spec.dataset.inner_cv_folds,
        n_jobs=4,
        repeats=5,
    )
    
    all_mask = np.ones(X_train.shape[1], dtype=bool)
    from sklearn.feature_selection import mutual_info_classif
    random_state = int(context.rng.numpy.integers(0, 2**32))
    relevance = mutual_info_classif(X_train, y_train, random_state=random_state)
    ranked_features = np.argsort(-relevance, kind="stable").tolist()
    
    mi_mask = np.zeros(X_train.shape[1], dtype=bool)
    best_acc = -1.0
    for f in ranked_features:
        test_mask = mi_mask.copy()
        test_mask[f] = True
        acc = reward_evaluator.score(test_mask).mean_accuracy
        if acc > best_acc:
            mi_mask = test_mask
            best_acc = acc
            
    # Pad to feature_budget to satisfy env constraint
    current_count = int(mi_mask.sum())
    if current_count < ppo_config.feature_budget:
        for f in ranked_features:
            if not mi_mask[f]:
                mi_mask[f] = True
                current_count += 1
                if current_count == ppo_config.feature_budget:
                    break
    elif current_count > ppo_config.feature_budget:
        # Should rarely happen, but truncate if needed
        removed = 0
        for f in reversed(ranked_features):
            if mi_mask[f]:
                mi_mask[f] = False
                removed += 1
                if current_count - removed == ppo_config.feature_budget:
                    break
    
    reward_best = _candidate(
        mi_mask,
        reward_evaluator.score(mi_mask),
        graph,
        ppo_config,
        "mi_warm_start",
    )
    best = _candidate(
        mi_mask,
        audit_evaluator.score(mi_mask),
        graph,
        ppo_config,
        "mi_warm_start",
    )
    
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
        initial_mask=mi_mask,
    )
    
    completed = 0
    update_index = 0
    metrics = []
    
    while completed < spec.ppo.episodes:
        buffer = RolloutBuffer.empty()
        batch_best = None
        batch_best_summary = None
        batch_size = min(spec.ppo.episodes_per_update, spec.ppo.episodes - completed)
        for offset in range(batch_size):
            episode = completed + offset + 1
            summary = _run_episode(
                env, agent, deterministic=False, buffer=buffer
            )
            reward_current = _candidate_from_summary(
                summary, f"train_episode_{episode}"
            )
            if _is_better(reward_current, reward_best):
                reward_best = reward_current
            if batch_best is None or _is_better(reward_current, batch_best):
                batch_best = reward_current
                batch_best_summary = summary
                
        completed += batch_size
        update_index += 1
        agent.set_learning_rate(1.0 - completed / spec.ppo.episodes)
        update_metrics = agent.update(buffer)
        
        checkpoint_summary = _run_episode(
            env, agent, deterministic=True
        )
        checkpoint_reward = _candidate_from_summary(
            checkpoint_summary, f"checkpoint_{update_index}"
        )
        
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
        if _is_archive_better(
            proposal_audit, best, 0.001
        ):
            best = proposal_audit
            
        # Log metric for runner
        elapsed = time.perf_counter() - started
        metrics.append(TrainingMetrics(
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
        ))
        
    for rollout in range(1, ppo_config.greedy_rollouts + 1):
        summary = _run_episode(env, agent, deterministic=True)
        audit_current = _candidate(
            summary.selected,
            audit_evaluator.score(summary.selected),
            graph,
            ppo_config,
            f"final_greedy_{rollout}",
        )
        if _is_archive_better(audit_current, best, 0.001):
            best = audit_current

    best_indices = tuple(np.flatnonzero(best.mask))
    per_step = tuple(
        StepRecord(subset=m.subset, accuracy=m.accuracy) for m in metrics
    )
    selection = make_selection(best_indices, per_step=per_step)
    
    return StableTrainingResult(
        selection=selection,
        metrics=tuple(metrics),
        initial_subset=tuple(np.flatnonzero(mi_mask)),
        initial_accuracy=reward_best.cv_accuracy,
        learner_updates=update_index,
        rejected_transitions=0,
    )
