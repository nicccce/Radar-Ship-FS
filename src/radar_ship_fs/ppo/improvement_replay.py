"""Completed-path action supervision, separate from the on-policy PPO objective."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from radar_ship_fs.ppo.block_rewrite import MacroPPOAgent


@dataclass(frozen=True)
class ReplayRecord:
    selected: tuple[bool, ...]
    progress: float
    current_score: float
    best_score: float
    action: tuple[int, ...]
    gain: float
    future_best: float
    immediate_delta: float

    @property
    def key(self) -> tuple:
        return (self.selected, self.progress, self.current_score, self.best_score, self.action)

    @property
    def weight(self) -> float:
        return float(np.clip(100 * self.gain, 0, 2))


def trajectory_records(trajectory: dict, n_features: int, mi_score: float) -> list[ReplayRecord]:
    """Use J[t+1:H+1], strictly excluding the current state from future-best."""
    if trajectory["phase"] != "training":
        return []
    steps = trajectory["steps"]
    scores = [trajectory["start_score"]] + [step["next_score"] for step in steps]
    future = np.maximum.accumulate(np.asarray(scores[:0:-1]))[::-1]
    selected = trajectory["start"]["selected_clean_indices_0based"]
    best = scores[0]
    result = []
    for t, step in enumerate(steps):
        best = max(best, scores[t])
        gain = float(future[t] - best)
        if gain > 1e-12 and future[t] >= mi_score:
            mask = tuple(i in selected for i in range(n_features))
            result.append(
                ReplayRecord(
                    mask,
                    t / len(steps),
                    scores[t],
                    best,
                    tuple(step["ordered_action_clean_indices_0based"]),
                    gain,
                    float(future[t]),
                    scores[t + 1] - scores[t],
                )
            )
        selected = step["next"]["selected_clean_indices_0based"]
    return result


class ReplayPool:
    def __init__(self, capacity: int = 512):
        self.capacity = capacity
        self.records: list[ReplayRecord] = []
        self.stats = dict(candidates_before_dedup=0, duplicates=0, truncated=0, immediate_declines=0)

    def add(self, records: list[ReplayRecord]) -> dict[str, int]:
        self.stats["candidates_before_dedup"] += len(records)
        self.stats["immediate_declines"] += sum(r.immediate_delta < 0 for r in records)
        combined = self.records + records
        unique = {}
        for r in combined:
            previous = unique.get(r.key)
            if previous is None or (r.future_best, r.gain) > (previous.future_best, previous.gain):
                unique[r.key] = r
        duplicates = len(combined) - len(unique)
        truncated = max(0, len(unique) - self.capacity)
        self.stats["duplicates"] += duplicates
        self.stats["truncated"] += truncated
        self.records = sorted(unique.values(), key=lambda r: (-r.future_best, -r.gain, r.key))[
            : self.capacity
        ]
        return dict(
            batch_candidates=len(records),
            before_dedup=len(combined),
            after_dedup=len(unique),
            duplicates=duplicates,
            truncated=truncated,
            pool_size=len(self.records),
        )

    def state_dict(self) -> dict:
        return dict(
            capacity=self.capacity, records=[asdict(r) for r in self.records], stats=self.stats.copy()
        )

    def load_state_dict(self, state: dict) -> None:
        self.capacity = state["capacity"]
        self.stats = state["stats"].copy()
        self.records = [ReplayRecord(**r) for r in state["records"]]


def evaluate_records(agent: MacroPPOAgent, records: list[ReplayRecord]):
    device = agent.device
    return agent.model.evaluate_actions(
        torch.tensor([r.selected for r in records], dtype=torch.bool, device=device),
        torch.tensor([r.progress for r in records], dtype=torch.float32, device=device),
        torch.tensor([r.current_score for r in records], dtype=torch.float32, device=device),
        torch.tensor([r.best_score for r in records], dtype=torch.float32, device=device),
        torch.tensor([r.action for r in records], dtype=torch.long, device=device),
    )


def record_metrics(agent: MacroPPOAgent, records: list[ReplayRecord]) -> dict[str, Any]:
    if not records:
        return {}
    with torch.no_grad():
        lp, entropy, _ = evaluate_records(agent, records)
    return dict(
        nll=float(-lp.mean()),
        probability=float(lp.exp().mean()),
        entropy=float(entropy.mean()),
        action_probabilities=lp.exp().cpu().tolist(),
        head_entropies=entropy.cpu().tolist(),
    )


def replay_update(
    agent: MacroPPOAgent, pool: ReplayPool, rng: np.random.Generator, steps: int = 16, batch_size: int = 64
) -> dict[str, float]:
    """Teacher force old ordered actions; no old log-probability, ratio or value loss."""
    if not pool.records or steps == 0:
        return dict(replay_steps=0, replay_loss=0.0, replay_gradient_norm=0.0)
    losses, norms = [], []
    for _ in range(steps):
        records = [pool.records[i] for i in rng.integers(0, len(pool.records), size=batch_size)]
        lp, _, _ = evaluate_records(agent, records)
        weights = torch.tensor([r.weight for r in records], dtype=lp.dtype, device=agent.device).detach()
        loss = -(weights * lp).sum() / weights.sum()
        agent.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if any(p.grad is not None for p in agent.model.critic.parameters()):
            raise AssertionError("replay must not train critic head")
        norm = torch.nn.utils.clip_grad_norm_(agent.model.parameters(), agent.config.max_grad_norm)
        agent.optimizer.step()
        losses.append(float(loss.detach()))
        norms.append(float(norm))
    return dict(
        replay_steps=steps, replay_loss=float(np.mean(losses)), replay_gradient_norm=float(np.mean(norms))
    )


class ImprovementUpdateHook:
    """Batch lifecycle: PPO, admit completed trajectories, supervised replay, discard buffer."""

    def __init__(
        self,
        method: str,
        mi_score: float,
        seed: int,
        fold: int,
        steps: int = 16,
        capacity: int = 512,
        batch_size: int = 64,
    ):
        self.method = method
        self.mi_score = mi_score
        self.steps = steps
        self.batch_size = batch_size
        self.pool = ReplayPool(capacity)
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, fold, 23, 3]))
        self.rows = []
        self.probe = []
        self.probe_before = {}
        self.initial_hash = None

    def state_dict(self) -> dict:
        return dict(
            method=self.method,
            mi_score=self.mi_score,
            steps=self.steps,
            batch_size=self.batch_size,
            pool=self.pool.state_dict(),
            rng=self.rng.bit_generator.state,
            rows=self.rows,
            probe=[asdict(r) for r in self.probe],
            probe_before=self.probe_before,
            initial_hash=self.initial_hash,
        )

    def load_state_dict(self, state: dict) -> None:
        for key in ("method", "mi_score", "steps", "batch_size"):
            if getattr(self, key) != state[key]:
                raise ValueError(f"checkpoint mismatch: {key}")
        self.pool.load_state_dict(state["pool"])
        self.rng.bit_generator.state = state["rng"]
        self.rows = state["rows"]
        self.probe = [ReplayRecord(**r) for r in state["probe"]]
        self.probe_before = state["probe_before"]
        self.initial_hash = state["initial_hash"]

    def update(self, agent, buffer, trajectories) -> dict[str, float]:
        import hashlib

        if not self.probe:
            self.initial_hash = hashlib.sha256(
                b"".join(p.detach().cpu().numpy().tobytes() for p in agent.model.parameters())
            ).hexdigest()
            self.probe = [
                ReplayRecord(
                    tuple(buffer.selected[i].tolist()),
                    buffer.progress[i],
                    buffer.current_score[i],
                    buffer.episode_best_score[i],
                    tuple(buffer.actions[i].tolist()),
                    0.0,
                    0.0,
                    0.0,
                )
                for i in range(min(16, len(buffer)))
            ]
            self.probe_before = record_metrics(agent, self.probe)
        before = torch.cat([p.detach().flatten() for p in agent.model.parameters()])
        ppo = agent.update(buffer) if self.method != "replay_only" else {}
        records = [
            r
            for trajectory in trajectories
            for r in trajectory_records(trajectory, agent.model.n_features, self.mi_score)
        ]
        counts = self.pool.add(records)
        pre = record_metrics(agent, self.pool.records)
        replay = replay_update(
            agent,
            self.pool,
            self.rng,
            self.steps if self.method != "plain_single_ppo" else 0,
            self.batch_size,
        )
        post = record_metrics(agent, self.pool.records)
        probe = record_metrics(agent, self.probe)
        after = torch.cat([p.detach().flatten() for p in agent.model.parameters()])
        row = dict(
            batch=len(self.rows),
            first_episode=trajectories[0]["episode"],
            last_episode=trajectories[-1]["episode"],
            ppo_updates=int(bool(ppo)),
            **counts,
            **replay,
            pool_nll_before=pre.get("nll"),
            pool_nll_after=post.get("nll"),
            pool_entropy=post.get("entropy"),
            probe_nll=probe["nll"],
            probe_probability=probe["probability"],
            probe_entropy=probe["entropy"],
            immediate_declines=sum(r.immediate_delta < 0 for r in records),
            parameter_delta_l2=float(torch.linalg.vector_norm(after - before)),
            **{f"ppo_{k}": v for k, v in ppo.items()},
        )
        self.rows.append(row)
        return {k: float(v) for k, v in row.items() if v is not None}

    def finalize(self, agent, case_dir) -> dict:
        from run_block_rewrite import write_csv, write_json

        final_probe = record_metrics(agent, self.probe)
        write_json(
            case_dir / "probe.json",
            dict(states=[asdict(r) for r in self.probe], before=self.probe_before, after=final_probe),
        )
        write_json(case_dir / "replay_pool.json", self.pool.state_dict())
        write_csv(case_dir / "replay_updates.csv", self.rows)
        gains = [r.gain for r in self.pool.records]
        return dict(
            pool_size=len(gains),
            pool_stats=self.pool.stats,
            gain_quantiles=np.quantile(gains, [0, 0.25, 0.5, 0.75, 1]).tolist() if gains else [],
            future_best_quantiles=np.quantile(
                [r.future_best for r in self.pool.records], [0, 0.25, 0.5, 0.75, 1]
            ).tolist()
            if gains
            else [],
            pool_immediate_declines=sum(r.immediate_delta < 0 for r in self.pool.records),
            final_pool=record_metrics(agent, self.pool.records),
            probe_before=self.probe_before,
            probe_after=final_probe,
            initial_parameter_sha256=self.initial_hash,
            ppo_update_count=sum(r["ppo_updates"] for r in self.rows),
            replay_gradient_steps=sum(r["replay_steps"] for r in self.rows),
            replay_batches=sum(r["replay_steps"] > 0 for r in self.rows),
        )
