#!/usr/bin/env python3
"""09B D: train-only STG starts and equal-request residual policy pilot."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.distributions import Categorical

import run_block_rewrite as base

CONFIG = Path("configs/v16n/stg_residual_rl_v1.toml")
KINDS = ("SWAP2", "SWAP3", "SWAP4", "ADD", "DELETE", "STOP")
ARMS = ("residual_rl", "random", "greedy", "scratch_mi_rl", "scratch_random_rl")


def ranked_mask(rank: np.ndarray, k: int) -> tuple[int, ...]:
    return tuple(sorted(int(i) for i in rank[:k]))


class StochasticGate(nn.Module):
    def __init__(self, hidden: int, sigma: float):
        super().__init__()
        self.mu = nn.Parameter(torch.full((65,), 0.5))
        self.sigma = sigma
        self.classifier = nn.Sequential(nn.Linear(65, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        z = (self.mu + self.sigma * torch.randn_like(self.mu)).clamp(0, 1)
        return self.classifier(x * z).squeeze(-1)


def train_stg(raw, labels, data, config, seed):
    """Gaussian stochastic gates; every scaler and parameter uses outer-train only."""
    torch.manual_seed(seed + 1000 * data.fold)
    torch.set_num_threads(int(config["policy"]["torch_threads"]))
    cols = np.asarray(data.final_ids) - 1
    x = StandardScaler().fit_transform(raw[data.train_rows][:, cols]).astype(np.float32)
    y = (labels[data.train_rows] == 1).astype(np.float32)
    ds = torch.utils.data.TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    gen = torch.Generator().manual_seed(seed + 1000 * data.fold)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=int(config["stg"]["batch_size"]), shuffle=True, generator=gen
    )
    net = StochasticGate(int(config["stg"]["hidden_dim"]), float(config["stg"]["sigma"]))
    opt = torch.optim.Adam(net.parameters(), lr=float(config["stg"]["learning_rate"]))
    started = time.perf_counter()
    losses = []
    for _ in range(int(config["stg"]["epochs"])):
        epoch_loss = 0.0
        for xb, yb in loader:
            logits = net(xb)
            # E[clip(N(mu,sigma),0,1)>0] = Phi(mu/sigma).
            probability_on = torch.special.ndtr(net.mu / net.sigma)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, yb)
            loss = loss + float(config["stg"]["lambda"]) * probability_on.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach())
        losses.append(epoch_loss / len(loader))
    mu = net.mu.detach().numpy()
    rank = np.lexsort((np.asarray(data.final_ids), -mu))
    return rank, {
        "method": "STG",
        "mu": mu.tolist(),
        "rank_clean_indices_0based": rank.tolist(),
        "rank_original_feature_ids_1based": list(data.mapping.clean_indices_to_original_ids_1based(rank)),
        "loss_by_epoch": losses,
        "wall_seconds": time.perf_counter() - started,
        "fit_rows": data.train_rows.tolist(),
        "seed": seed,
    }


def legal(mask, min_k, max_k):
    k = len(mask)
    return (
        k >= 2 and 65 - k >= 2,
        k >= 3 and 65 - k >= 3,
        k >= 4 and 65 - k >= 4,
        k < max_k,
        k > min_k,
        True,
    )


def apply(mask, action, min_k, max_k):
    kind, remove, add = action
    if kind not in KINDS or not legal(mask, min_k, max_k)[KINDS.index(kind)]:
        raise ValueError("illegal action type")
    if len(remove) != len(set(remove)) or len(add) != len(set(add)):
        raise ValueError("repeated feature in action")
    selected = set(mask)
    if not set(remove) <= selected or set(add) & selected or not set(add) <= set(range(65)):
        raise ValueError("illegal action feature")
    expected = int(kind[-1]) if kind.startswith("SWAP") else 1 if kind != "STOP" else 0
    if kind.startswith("SWAP") and (len(remove) != expected or len(add) != expected):
        raise ValueError("wrong group size")
    if kind == "ADD" and (remove or len(add) != 1):
        raise ValueError("wrong ADD size")
    if kind == "DELETE" and (len(remove) != 1 or add):
        raise ValueError("wrong DELETE size")
    if kind == "STOP" and (remove or add):
        raise ValueError("wrong STOP size")
    result = tuple(sorted((selected - set(remove)) | set(add)))
    if not min_k <= len(result) <= max_k:
        raise ValueError("K outside range")
    return result


def random_action(mask, rng, min_k, max_k):
    allowed = [kind for kind, ok in zip(KINDS, legal(mask, min_k, max_k)) if ok]
    kind = str(rng.choice(allowed))
    n = int(kind[-1]) if kind.startswith("SWAP") else 1
    remove = (
        tuple(int(i) for i in rng.choice(mask, n, replace=False))
        if kind.startswith("SWAP") or kind == "DELETE"
        else ()
    )
    outside = np.setdiff1d(np.arange(65), np.asarray(mask), assume_unique=True)
    add = (
        tuple(int(i) for i in rng.choice(outside, n, replace=False))
        if kind.startswith("SWAP") or kind == "ADD"
        else ()
    )
    return kind, remove, add


class ResidualPolicy(nn.Module):
    def __init__(self, rank, mi, hidden):
        super().__init__()
        priority = np.empty(65, dtype=np.float32)
        priority[np.asarray(rank)] = np.linspace(1, 0, 65)
        self.register_buffer("priority", torch.from_numpy(priority))
        self.register_buffer("mi", torch.as_tensor(mi, dtype=torch.float32))
        self.body = nn.Sequential(
            nn.Linear(65 * 3 + 6, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh()
        )
        self.kind = nn.Linear(hidden, len(KINDS))
        self.remove = nn.Linear(hidden, 65)
        self.add = nn.Linear(hidden, 65)
        self.value = nn.Linear(hidden, 1)

    def sample(self, mask, score, best, errors, min_k, max_k):
        bits = torch.zeros(65)
        bits[list(mask)] = 1
        state = torch.cat(
            (
                bits,
                self.priority,
                self.mi,
                torch.tensor([len(mask) / 65, score, best, *errors], dtype=torch.float32),
            )
        )
        hidden = self.body(state)
        kd = Categorical(logits=self.kind(hidden).masked_fill(~torch.tensor(legal(mask, min_k, max_k)), -1e9))
        ki = kd.sample()
        kind = KINDS[int(ki)]
        logp, entropy = kd.log_prob(ki), kd.entropy()
        selected = torch.zeros(65, dtype=torch.bool)
        selected[list(mask)] = True
        remove, add = [], []
        nr = int(kind[-1]) if kind.startswith("SWAP") else int(kind == "DELETE")
        na = int(kind[-1]) if kind.startswith("SWAP") else int(kind == "ADD")
        rlogits, alogits = self.remove(hidden), self.add(hidden)
        for _ in range(nr):
            d = Categorical(logits=rlogits.masked_fill(~selected, -1e9))
            i = d.sample()
            remove.append(int(i))
            logp, entropy = logp + d.log_prob(i), entropy + d.entropy()
            selected[int(i)] = False
        outside = torch.ones(65, dtype=torch.bool)
        outside[list(mask)] = False
        for _ in range(na):
            d = Categorical(logits=alogits.masked_fill(~outside, -1e9))
            i = d.sample()
            add.append(int(i))
            logp, entropy = logp + d.log_prob(i), entropy + d.entropy()
            outside[int(i)] = False
        return (kind, tuple(remove), tuple(add)), logp, self.value(hidden).squeeze(), entropy


def rank_key(mask, score, data):
    return (-score, tuple(data.final_ids[i] for i in mask))


def endpoint(mask, score, data):
    return base.subset_payload(mask, data) | {"K": len(mask), "inner_j": float(score)}


def search(arm, start, start_score, rank, raw, labels, data, config, seed, jobs, smoke):
    settings = config["search"]
    min_k, max_k = int(settings["min_k"]), int(settings["max_k"])
    target = 4 if smoke else int(settings["unique_per_start"])
    train_target = 4 if smoke else int(settings["training_unique_per_start"])
    batch_size = int(settings["batch_size"])
    rng = np.random.default_rng(
        np.random.SeedSequence([seed, data.fold, ARMS.index(arm), len(start), 20260923])
    )
    scorer = base.SVCScorer(raw, labels, data, jobs)
    scorer.score_many([start])  # Real request and three fits; no inherited score credit.
    if abs(scorer.cache[start]["objective"] - start_score) > 1e-12:
        raise AssertionError("start score mismatch")
    archive = {start: start_score}
    mi = np.asarray(data.graph.mutual_information, dtype=np.float32)
    mi = (mi - mi.mean()) / (mi.std() + 1e-8)
    torch.manual_seed(seed + data.fold * 1000 + ARMS.index(arm) * 100 + len(start))
    policy = ResidualPolicy(rank, mi, int(config["policy"]["hidden_dim"])) if arm.endswith("rl") else None
    optimizer = (
        torch.optim.Adam(policy.parameters(), lr=float(config["policy"]["learning_rate"])) if policy else None
    )
    current = start
    candidates, attempts, updates = [], [], []
    begun = time.perf_counter()
    while len(candidates) < target:
        phase = "training" if len(candidates) < train_target else "frozen"
        parent = current
        parent_score = archive[parent]
        errors = tuple(1 - float(s) for s in scorer.cache[parent]["fold_scores"])
        batch = []
        local = set()
        stop_seen = False
        for _ in range(int(settings["max_attempts_per_batch"])):
            if len(batch) == batch_size:
                break
            if policy:
                with torch.set_grad_enabled(phase == "training"):
                    action, logp, value, entropy = policy.sample(
                        parent, parent_score, max(archive.values()), errors, min_k, max_k
                    )
            else:
                action = random_action(parent, rng, min_k, max_k)
                logp = value = entropy = None
            child = apply(parent, action, min_k, max_k)
            repeated = child in archive or child in local
            attempts.append(
                {
                    "phase": phase,
                    "action": action,
                    "parent": list(parent),
                    "child": list(child),
                    "duplicate": repeated,
                }
            )
            if action[0] == "STOP":
                stop_seen = True
                continue
            if repeated:
                continue
            local.add(child)
            batch.append((action, child, logp, value, entropy))
        if len(batch) != batch_size:
            raise RuntimeError("could not fill unique batch")
        before = scorer.costs()
        scores = scorer.score_many([item[1] for item in batch])
        cost = base.cost_delta(before, scorer.costs())
        if cost["classifier_fit_count"] != 3 * batch_size or cost["cache_hits"]:
            raise AssertionError("candidate request accounting failed")
        for item, score in zip(batch, scores):
            action, child = item[:2]
            archive[child] = score
            candidates.append(
                {
                    "phase": phase,
                    "action": action,
                    "parent": list(parent),
                    "child": list(child),
                    "K": len(child),
                    "inner_j": score,
                    "original_feature_ids_1based": list(
                        data.mapping.clean_indices_to_original_ids_1based(child)
                    ),
                }
            )
        if policy and phase == "training":
            rewards = torch.tensor([100 * (score - parent_score) for score in scores])
            logps = torch.stack([item[2] for item in batch])
            values = torch.stack([item[3] for item in batch])
            entropies = torch.stack([item[4] for item in batch])
            loss = -(logps * (rewards - values.detach())).mean() + 0.5 * (values - rewards).square().mean()
            loss -= float(config["policy"]["entropy_coef"]) * entropies.mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1)
            optimizer.step()
            updates.append({"after_requests": len(candidates), "loss": float(loss.detach())})
        if arm == "random":
            current = batch[int(rng.integers(batch_size))][1]
        else:
            current = min(
                [parent] + [item[1] for item in batch], key=lambda mask: rank_key(mask, archive[mask], data)
            )
        if stop_seen or len(candidates) % 12 == 0:
            current = min(archive, key=lambda mask: rank_key(mask, archive[mask], data))
    winner = min(archive, key=lambda mask: rank_key(mask, archive[mask], data))
    if scorer.fit_count != 3 * (target + 1) or scorer.requests != target + 1:
        raise AssertionError("unequal physical fit budget")
    return {
        "start": endpoint(start, start_score, data),
        "endpoint": endpoint(winner, archive[winner], data),
        "candidate_requests": target,
        "start_requests": 1,
        "physical_svc_fits": scorer.fit_count,
        "proposal_attempts": len(attempts),
        "stop_proposals": sum(row["action"][0] == "STOP" for row in attempts),
        "duplicate_proposals": sum(row["duplicate"] for row in attempts),
        "wall_seconds": time.perf_counter() - begun,
        "candidates": candidates,
        "attempts": attempts,
        "updates": updates,
    }


def run_case(raw, labels, data, config, seed, root, jobs, smoke=False):
    case_dir = root / "cases" / f"fold-{data.fold}-seed-{seed}"
    case_dir.mkdir(parents=True, exist_ok=True)
    if smoke:
        train = set(int(i) for i in data.train_rows)
        held = set(int(i) for i in data.validation_rows)
        assert train and held and train.isdisjoint(held)
        assert train | held == set(range(len(labels)))
        assert all(
            set(split["fit"]) <= train
            and set(split["held_out"]) <= train
            and set(split["fit"]).isdisjoint(split["held_out"])
            for split in data.inner_folds
        )
        assert len(data.final_ids) == 65 and data.mapping.clean_feature_count == 65
        for k in (16, 32, 48):
            mask = tuple(range(k))
            for kind, allowed in zip(KINDS, legal(mask, 16, 48)):
                if not allowed:
                    continue
                n = int(kind[-1]) if kind.startswith("SWAP") else 1
                remove = tuple(range(n)) if kind.startswith("SWAP") or kind == "DELETE" else ()
                add = tuple(range(k, k + n)) if kind.startswith("SWAP") or kind == "ADD" else ()
                child = apply(mask, (kind, remove, add), 16, 48)
                assert 16 <= len(child) <= 48 and len(set(child)) == len(child)
        try:
            apply(tuple(range(16)), ("DELETE", (0,), ()), 16, 48)
        except ValueError:
            pass
        else:
            raise AssertionError("DELETE at minimum K was accepted")
    started = time.perf_counter()
    stg_rank, gate = train_stg(raw, labels, data, config, seed)
    base.write_json(case_dir / "stg_gate.json", gate)
    mi_rank = np.lexsort((np.asarray(data.final_ids), -np.asarray(data.graph.mutual_information)))
    rng = np.random.default_rng(np.random.SeedSequence([seed, data.fold, 991]))
    random_rank = rng.permutation(65)
    starts = {f"stg{k}": ranked_mask(stg_rank, k) for k in (16, 32, 48)}
    starts |= {
        "mi16": ranked_mask(mi_rank, 16),
        "mi32": ranked_mask(mi_rank, 32),
        "mi48": ranked_mask(mi_rank, 48),
        "random16": ranked_mask(random_rank, 16),
        "random32": ranked_mask(random_rank, 32),
        "random48": ranked_mask(random_rank, 48),
    }
    for mask in starts.values():
        assert len(mask) in (16, 32, 48)
        assert (
            data.mapping.original_ids_1based_to_clean_indices(
                data.mapping.clean_indices_to_original_ids_1based(mask)
            )
            == mask
        )
    start_scorer = base.SVCScorer(raw, labels, data, jobs)
    names = list(starts)
    scores = start_scorer.score_many([starts[name] for name in names])
    start_scores = dict(zip(names, scores))
    branches = {}
    for arm in ARMS:
        branch_names = names[:3] if arm in ARMS[:3] else names[3:6] if arm == "scratch_mi_rl" else names[6:]
        for name in branch_names:
            print(f"fold={data.fold} seed={seed} {arm}/{name}", flush=True)
            rank = stg_rank if arm in ARMS[:3] else mi_rank if arm == "scratch_mi_rl" else random_rank
            result = search(
                arm, starts[name], start_scores[name], rank, raw, labels, data, config, seed, jobs, smoke
            )
            branches[f"{arm}/{name}"] = {
                k: v for k, v in result.items() if k not in ("candidates", "attempts")
            }
            base.write_jsonl(case_dir / f"{arm}-{name}-candidates.jsonl", result["candidates"])
            base.write_jsonl(case_dir / f"{arm}-{name}-attempts.jsonl", result["attempts"])
    # Frozen selection uses inner J only. The first outer access is below.
    selected = {}
    for arm in ARMS:
        options = [(name, row) for name, row in branches.items() if name.startswith(arm + "/")]
        name, row = min(
            options,
            key=lambda pair: (
                -pair[1]["endpoint"]["inner_j"],
                pair[1]["endpoint"]["selected_original_feature_ids_1based"],
            ),
        )
        selected[arm] = row["endpoint"] | {"selected_branch": name}
    case = {
        "fold": data.fold,
        "seed": seed,
        "source": "STG",
        "gate": "stg_gate.json",
        "starts": {name: endpoint(mask, start_scores[name], data) for name, mask in starts.items()},
        "start_scorer_costs": start_scorer.costs(),
        "branches": branches,
        "selected": selected,
        "source_test_open_count": 0,
        "outer_calls_during_search": 0,
    }
    if not smoke:
        outer_cache = {}

        def outer(mask):
            key = tuple(mask)
            if key not in outer_cache:
                outer_cache[key] = base.endpoint_metrics(raw, labels, data, key)
            return outer_cache[key]

        for row in branches.values():
            row["endpoint"]["outer"] = outer(row["endpoint"]["selected_clean_indices_0based"])
        for arm, row in selected.items():
            row["outer"] = branches[row["selected_branch"]]["endpoint"]["outer"]
        for name in names[:3]:
            case["starts"][name]["outer"] = outer(starts[name])
        case["outer_endpoint_fits"] = len(outer_cache)
    case["wall_seconds"] = time.perf_counter() - started
    base.write_json(case_dir / ("smoke.json" if smoke else "case.json"), case)
    return case


def report(root, config):
    cases = [
        base.read_json(root / "cases" / f"fold-{fold}-seed-{seed}" / "case.json")
        for fold in config["experiment"]["outer_folds"]
        for seed in config["experiment"]["seeds"]
    ]
    rows = []
    for case in cases:
        for name, branch in case["branches"].items():
            arm, start = name.split("/")
            mask = branch["endpoint"]["selected_clean_indices_0based"]
            selected = case["selected"][arm]
            start_outer = case["starts"].get(start, {}).get("outer", {})
            rows.append(
                {
                    "fold": case["fold"],
                    "seed": case["seed"],
                    "arm": arm,
                    "start": start,
                    "selected": mask == selected["selected_clean_indices_0based"],
                    "K": branch["endpoint"]["K"],
                    "inner_j": branch["endpoint"]["inner_j"],
                    "start_inner_j": branch["start"]["inner_j"],
                    "start_outer_bacc": start_outer.get("balanced_accuracy"),
                    "outer_bacc": branch["endpoint"]["outer"]["balanced_accuracy"],
                    "outer_accuracy": branch["endpoint"]["outer"]["accuracy"],
                    "candidate_requests": branch["candidate_requests"],
                    "physical_svc_fits": branch["physical_svc_fits"],
                    "wall_seconds": branch["wall_seconds"],
                }
            )
    base.write_csv(root / "branches.csv", rows)
    paired = []
    for case in cases:
        chosen = case["selected"]
        b = {arm: chosen[arm]["outer"]["balanced_accuracy"] for arm in ARMS}
        a = {arm: chosen[arm]["outer"]["accuracy"] for arm in ARMS}
        start_name = chosen["residual_rl"]["selected_branch"].split("/")[1]
        start_b = case["starts"][start_name]["outer"]["balanced_accuracy"]
        paired.append(
            {
                "fold": case["fold"],
                "seed": case["seed"],
                "rl_start": start_name,
                "rl_K": chosen["residual_rl"]["K"],
                "start_bacc": start_b,
                **{f"{arm}_bacc": b[arm] for arm in ARMS},
                **{f"{arm}_accuracy": a[arm] for arm in ARMS},
                "rl_minus_start_pp": 100 * (b["residual_rl"] - start_b),
                "rl_minus_random_pp": 100 * (b["residual_rl"] - b["random"]),
                "rl_minus_greedy_pp": 100 * (b["residual_rl"] - b["greedy"]),
                "rl_minus_scratch_mi_pp": 100 * (b["residual_rl"] - b["scratch_mi_rl"]),
                "rl_minus_scratch_random_pp": 100 * (b["residual_rl"] - b["scratch_random_rl"]),
                "case_wall_seconds": case["wall_seconds"],
            }
        )
    base.write_csv(root / "paired.csv", paired)
    by_start = []
    for case in cases:
        for start in ("stg16", "stg32", "stg48"):
            original = case["starts"][start]["outer"]["balanced_accuracy"]
            scores = {
                arm: case["branches"][f"{arm}/{start}"]["endpoint"]["outer"]["balanced_accuracy"]
                for arm in ("residual_rl", "random", "greedy")
            }
            by_start.append(
                {
                    "fold": case["fold"],
                    "seed": case["seed"],
                    "start": start,
                    "start_bacc": original,
                    **{f"{arm}_bacc": value for arm, value in scores.items()},
                    "rl_minus_start_pp": 100 * (scores["residual_rl"] - original),
                    "rl_minus_random_pp": 100 * (scores["residual_rl"] - scores["random"]),
                    "rl_minus_greedy_pp": 100 * (scores["residual_rl"] - scores["greedy"]),
                }
            )
    base.write_csv(root / "paired_by_start.csv", by_start)
    summary = {
        "case_count": len(cases),
        "source": "STG",
        "paired_means": {
            key: float(np.mean([row[key] for row in paired]))
            for key in paired[0]
            if key.endswith("_bacc") or key.endswith("_accuracy") or key.endswith("_pp")
        },
        "total_physical_svc_fits": sum(
            sum(b["physical_svc_fits"] for b in c["branches"].values())
            + c["start_scorer_costs"]["classifier_fit_count"]
            + c["outer_endpoint_fits"]
            for c in cases
        ),
        "total_wall_seconds": sum(c["wall_seconds"] for c in cases),
        "outer_folds_are_independent_units": 2,
        "source_test_open_count": 0,
    }
    base.write_json(root / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("smoke", "run", "report"))
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    config = base.load_config(args.config)
    if float(config["scorer"]["C"]) != 1 or abs(float(config["scorer"]["gamma"]) - 1 / 65) > 1e-15:
        raise ValueError("this driver requires common C=1, gamma=1/65 scorer")
    root = Path(config["output"]["root"])
    root.mkdir(parents=True, exist_ok=True)
    base.write_json(root / "config.json", {"config": config, "sha256": base.sha256(args.config)})
    if args.stage == "report":
        print(json.dumps(report(root, config), indent=2))
        return
    raw, labels = base.load_train(config)
    folds = (
        config["experiment"]["outer_folds"][:1]
        if args.stage == "smoke"
        else config["experiment"]["outer_folds"]
    )
    seeds = config["experiment"]["seeds"][:1] if args.stage == "smoke" else config["experiment"]["seeds"]
    for fold in folds:
        data = base.prepare_fold(raw, labels, config, int(fold), root)
        for seed in seeds:
            path = (
                root
                / "cases"
                / f"fold-{fold}-seed-{seed}"
                / ("smoke.json" if args.stage == "smoke" else "case.json")
            )
            if not path.exists():
                run_case(raw, labels, data, config, int(seed), root, args.jobs, args.stage == "smoke")
    if args.stage == "run":
        print(json.dumps(report(root, config), indent=2))


if __name__ == "__main__":
    main()
