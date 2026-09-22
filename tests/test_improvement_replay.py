from dataclasses import replace

import numpy as np
import pytest
import torch
from test_block_rewrite import toy_graph

from radar_ship_fs.ppo.block_rewrite import ConditionalMacroActorCritic, MacroPPOAgent, MacroPPOConfig
from radar_ship_fs.ppo.improvement_replay import (
    ImprovementUpdateHook,
    ReplayPool,
    ReplayRecord,
    record_metrics,
    replay_update,
    trajectory_records,
)


def trajectory(scores, phase="training"):
    subsets = [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]]
    return dict(
        phase=phase,
        episode=0,
        start_score=scores[0],
        start=dict(selected_clean_indices_0based=subsets[0]),
        steps=[
            dict(
                next_score=score,
                next=dict(selected_clean_indices_0based=subsets[t + 1]),
                ordered_action_clean_indices_0based=[t, t + 3],
            )
            for t, score in enumerate(scores[1:])
        ],
    )


def agent():
    torch.set_num_threads(1)
    torch.manual_seed(43)
    return MacroPPOAgent(
        ConditionalMacroActorCritic(toy_graph(), mode="single", hidden_dim=16),
        MacroPPOConfig(),
        torch.device("cpu"),
    )


def test_future_best_includes_decline_but_excludes_current_and_respects_mi():
    records = trajectory_records(trajectory([0.8, 0.7, 0.9, 0.85]), 6, 0.88)
    assert len(records) == 2
    assert records[0].immediate_delta == pytest.approx(-0.1)
    assert records[1].current_score == 0.7 and records[1].best_score == 0.8
    assert records[1].progress == 1 / 3
    assert records[0].gain == pytest.approx(0.1)
    assert records[0].future_best == 0.9
    assert records[0].selected == (True, True, True, False, False, False)
    assert records[1].action == (1, 4)
    assert not trajectory_records(trajectory([0.8, 0.7, 0.79, 0.78]), 6, 0.7)
    assert not trajectory_records(trajectory([0.8, 0.7, 0.9, 0.85]), 6, 0.91)
    assert not trajectory_records(trajectory([0.8, 0.7, 0.9, 0.85], "frozen"), 6, 0.88)
    late = trajectory_records(trajectory([0.8, 0.85, 0.84, 0.9]), 6, 0.9)
    assert late[2].best_score == 0.85 and late[2].gain == pytest.approx(0.05)


def test_dedup_capacity_order_weights_and_state_identity():
    r = trajectory_records(trajectory([0.8, 0.7, 0.9, 0.85]), 6, 0.88)[0]
    assert r.weight == 2.0
    assert replace(r, gain=0.005).weight == 0.5
    pool = ReplayPool(2)
    higher = replace(r, future_best=0.95, gain=0.15)
    separate_state = replace(r, progress=0.125)
    separate_action = replace(r, action=(1, 3))
    stats = pool.add([r, higher, separate_state, separate_action])
    assert stats["before_dedup"] == 4 and stats["after_dedup"] == 3
    assert stats["duplicates"] == 1 and stats["truncated"] == 1
    assert pool.records[0] == higher
    other = ReplayPool(2)
    other.add(list(reversed([r, higher, separate_state, separate_action])))
    assert pool.records == other.records
    restored = ReplayPool()
    restored.load_state_dict(pool.state_dict())
    assert restored.records == pool.records


def test_replay_supervises_action_without_ratio_and_increases_probability():
    a = agent()
    r = ReplayRecord((True, True, True, False, False, False), 0, 0.8, 0.8, (0, 3), 0.01, 0.81, -0.02)
    pool = ReplayPool()
    pool.add([r])
    critic_before = [p.clone() for p in a.model.critic.parameters()]
    before = record_metrics(a, [r])["probability"]
    # No old log probabilities or rollout buffer are supplied to this update.
    metrics = replay_update(a, pool, np.random.default_rng(1), steps=1)
    assert metrics["replay_gradient_norm"] > 0
    assert record_metrics(a, [r])["probability"] > before
    assert all(torch.equal(p, q) for p, q in zip(critic_before, a.model.critic.parameters()))
    assert all(p.grad is None for p in a.model.critic.parameters())


def test_checkpoint_restores_pool_rng_and_exact_next_update(tmp_path):
    a = agent()
    hook = ImprovementUpdateHook("replay_only", 0.8, 17, 0)
    hook.pool.add(trajectory_records(trajectory([0.8, 0.7, 0.9, 0.85]), 6, 0.8))
    replay_update(a, hook.pool, hook.rng, steps=2)
    path = tmp_path / "checkpoint.pt"
    torch.save(
        dict(
            model=a.model.state_dict(),
            optimizer=a.optimizer.state_dict(),
            hook=hook.state_dict(),
            torch_rng=torch.get_rng_state(),
        ),
        path,
    )
    state = torch.load(path)
    b = agent()
    b.model.load_state_dict(state["model"])
    b.optimizer.load_state_dict(state["optimizer"])
    restored = ImprovementUpdateHook("replay_only", 0.8, 17, 0)
    restored.load_state_dict(state["hook"])
    torch.set_rng_state(state["torch_rng"])
    replay_update(a, hook.pool, hook.rng, steps=1)
    replay_update(b, restored.pool, restored.rng, steps=1)
    assert hook.pool.records == restored.pool.records
    assert hook.rng.bit_generator.state == restored.rng.bit_generator.state
    assert all(torch.equal(p, q) for p, q in zip(a.model.parameters(), b.model.parameters()))


def test_empty_pool_skips_and_plain_does_no_replay():
    a = agent()
    assert replay_update(a, ReplayPool(), np.random.default_rng(1))["replay_steps"] == 0
    pool = ReplayPool()
    pool.add(trajectory_records(trajectory([0.8, 0.7, 0.9, 0.85]), 6, 0.8))
    assert replay_update(a, pool, np.random.default_rng(1), steps=0)["replay_steps"] == 0


def test_batch_lifecycle_ppo_precedes_admission_and_replay(monkeypatch):
    import radar_ship_fs.ppo.improvement_replay as module
    from radar_ship_fs.ppo.block_rewrite import MacroObservation, MacroRolloutBuffer

    a = agent()
    hook = ImprovementUpdateHook("replay_ppo", 0.8, 17, 0, steps=1)
    path = trajectory([0.8, 0.7, 0.9, 0.85])
    buffer = MacroRolloutBuffer.empty()
    observation = MacroObservation(np.array([True, True, True, False, False, False]), 0, 0.8, 0.8)
    action, lp, value, _ = a.act_many([observation])
    buffer.add(observation, action[0], lp[0], value[0], 1, True)
    order = []
    original_ppo = a.update
    original_replay = module.replay_update

    def ppo(data):
        assert data is buffer
        assert not hook.pool.records
        order.append("ppo")
        return original_ppo(data)

    def replay(agent_arg, pool, *args):
        assert len(pool.records) == 2
        order.append("replay")
        return original_replay(agent_arg, pool, *args)

    monkeypatch.setattr(a, "update", ppo)
    monkeypatch.setattr(module, "replay_update", replay)
    hook.update(a, buffer, [path])
    assert order == ["ppo", "replay"]
    assert hook.rows[0]["ppo_updates"] == 1
    assert hook.rows[0]["replay_steps"] == 1


def test_weighted_loss_normalizes_clipped_weights():
    from radar_ship_fs.ppo.improvement_replay import evaluate_records

    a = agent()
    a.optimizer.param_groups[0]["lr"] = 0
    first = ReplayRecord((True, True, True, False, False, False), 0, 0.8, 0.8, (0, 3), 0.005, 0.805, 0)
    second = replace(first, action=(1, 4), gain=0.03)
    pool = ReplayPool()
    pool.add([first, second])
    seed = 83
    sampled = [pool.records[i] for i in np.random.default_rng(seed).integers(0, 2, size=64)]
    lp, _, _ = evaluate_records(a, sampled)
    weights = torch.tensor([r.weight for r in sampled])
    expected = float(-(lp * weights).sum() / weights.sum())
    observed = replay_update(a, pool, np.random.default_rng(seed), steps=1)
    assert observed["replay_loss"] == pytest.approx(expected)
