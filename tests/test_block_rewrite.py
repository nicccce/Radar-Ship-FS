from pathlib import Path

import numpy as np
import torch

from radar_ship_fs.ppo.block_rewrite import (
    ConditionalMacroActorCritic,
    MacroObservation,
    MacroPPOAgent,
    MacroPPOConfig,
    MacroRolloutBuffer,
    action_mask,
    apply_macro_action,
    best_so_far_reward,
    generalized_advantages,
    signed_increment_reward,
)
from radar_ship_fs.ppo.ppo_graph import FeatureGraph


def toy_graph(n_features: int = 6) -> FeatureGraph:
    identity = np.eye(n_features, dtype=np.float32)
    return FeatureGraph(
        signed_correlation=identity,
        absolute_correlation=identity,
        correlation_adjacency=identity,
        dependency_adjacency=identity,
        static_node_features=np.column_stack(
            [
                np.linspace(0.0, 1.0, n_features),
                np.linspace(1.0, 0.0, n_features),
                np.zeros((n_features, 3)),
            ]
        ).astype(np.float32),
        relevance=np.linspace(0.0, 1.0, n_features).astype(np.float32),
        mutual_information=np.linspace(1.0, 0.0, n_features).astype(np.float32),
        feature_ids=np.arange(n_features, dtype=np.float32),
        threshold=0.5,
        edge_count=0,
        dependency_edge_count=0,
    )


def test_pair_action_is_ordered_legal_and_exact_size() -> None:
    selected = (0, 1, 2)
    result = apply_macro_action(selected, (2, 0, 4, 3), n_features=6)
    assert result == (1, 3, 4)
    assert len(result) == len(selected)
    assert set(result) <= set(range(6))

    mask = torch.tensor([[True, True, True, False, False, False]])
    actions = torch.tensor([[2, 0, 4, 3]])
    assert action_mask(mask, actions, 0, "pair").tolist() == [[True, True, True, False, False, False]]
    assert action_mask(mask, actions, 1, "pair").tolist() == [[True, True, False, False, False, False]]
    assert action_mask(mask, actions, 2, "pair").tolist() == [[False, False, False, True, True, True]]
    assert action_mask(mask, actions, 3, "pair").tolist() == [[False, False, False, True, False, True]]


def test_teacher_forcing_same_policy_has_unit_joint_ratio() -> None:
    torch.manual_seed(7)
    model = ConditionalMacroActorCritic(toy_graph(), mode="pair", hidden_dim=16)
    selected = torch.tensor(
        [
            [True, True, True, False, False, False],
            [False, True, False, True, True, False],
        ]
    )
    progress = torch.tensor([0.0, 0.5])
    current = torch.tensor([0.8, 0.7])
    best = torch.tensor([0.8, 0.75])
    with torch.no_grad():
        actions, old_log_probability, _, sampled_entropy = model.sample_actions(
            selected, progress, current, best
        )
        new_log_probability, entropy, _ = model.evaluate_actions(selected, progress, current, best, actions)
    ratio = torch.exp(new_log_probability - old_log_probability)
    assert torch.allclose(ratio, torch.ones_like(ratio), atol=1e-6)
    assert torch.allclose(entropy, sampled_entropy, atol=1e-6)
    assert model.prior_scale == 0.0


def test_declining_move_executes_and_reward_definitions_are_distinct() -> None:
    result = apply_macro_action((0, 1, 2), (0, 4), n_features=6)
    assert result == (1, 2, 4)
    reward, new_best = best_so_far_reward(0.8, 0.7)
    assert reward == 0.0 and new_best == 0.8
    signed, new_current = signed_increment_reward(0.8, 0.7)
    assert np.isclose(signed, -10.0) and new_current == 0.7


def test_gae_does_not_cross_episode_boundaries() -> None:
    advantages, returns = generalized_advantages(
        rewards=[1.0, 2.0, 10.0, 20.0],
        values=[0.0, 0.0, 0.0, 0.0],
        dones=[False, True, False, True],
        gamma=1.0,
        gae_lambda=1.0,
    )
    assert np.allclose(advantages, [3.0, 2.0, 30.0, 20.0])
    assert np.allclose(returns, advantages)


def test_real_macro_update_changes_parameters_and_checkpoint_loads(tmp_path: Path) -> None:
    torch.manual_seed(11)
    graph = toy_graph()
    agent = MacroPPOAgent(
        ConditionalMacroActorCritic(graph, mode="pair", hidden_dim=16),
        MacroPPOConfig(ppo_epochs=1, minibatch_size=8),
        torch.device("cpu"),
    )
    observations = []
    for index in range(4):
        mask = np.zeros(6, dtype=np.bool_)
        mask[[0, 1, 2]] = True
        observations.append(MacroObservation(mask, index / 4, 0.7, 0.75))
    actions, log_probabilities, values, _ = agent.act_many(observations)
    buffer = MacroRolloutBuffer.empty()
    for index, observation in enumerate(observations):
        buffer.add(
            observation,
            actions[index],
            log_probabilities[index],
            values[index],
            reward=float(index),
            done=index in {1, 3},
        )
    metrics = agent.update(buffer)
    assert np.isfinite(metrics["gradient_norm"])
    assert metrics["parameter_delta_l2"] > 0.0

    checkpoint = tmp_path / "model.pt"
    torch.save(agent.model.state_dict(), checkpoint)
    reloaded = ConditionalMacroActorCritic(graph, mode="pair", hidden_dim=16)
    reloaded.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    for original, restored in zip(agent.model.parameters(), reloaded.parameters()):
        assert torch.equal(original, restored)
