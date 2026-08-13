"""Physical-domain virtual-node GCN controls and checked-in screen matrix."""

from __future__ import annotations

from collections import Counter

import torch

from config import load_config
from harness.orchestrator import build_run_context
from radar_ship_fs.experiment.config import load_experiment_spec
from radar_ship_fs.experiment.runner import ExperimentRunner
from radar_ship_fs.feedback.encoders import (
    DomainHierarchyGCNBatchStateEncoder,
    TrainableGCNBatchStateEncoder,
    build_batch_encoder,
    radar_feature_domains,
    shuffled_domain_assignment,
)

_RADAR_IDS = (
    1,
    2,
    3,
    4,
    5,
    21,
    22,
    23,
    24,
    25,
    32,
    33,
    34,
    35,
    36,
    39,
    40,
    41,
    42,
    43,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
)


def _radar_metadata_context(seed: int = 42):
    context = build_run_context(load_config({"seeds": (seed,)}), seed=seed)
    train = context.split.train._replace(metadata={"final_feature_ids": list(_RADAR_IDS)})
    return context._replace(split=context.split._replace(train=train))


def test_radar_domains_follow_original_ids_and_shuffle_preserves_group_sizes() -> None:
    context = _radar_metadata_context()
    domains = radar_feature_domains(context)

    assert domains == (0,) * 5 + (1,) * 5 + (2,) * 5 + (3,) * 5 + (4,) * 2 + (5,) * 8
    shuffled = shuffled_domain_assignment(domains, random_state=7)
    assert shuffled == shuffled_domain_assignment(domains, random_state=7)
    assert shuffled != domains
    assert Counter(shuffled) == Counter(domains)


def test_domain_hierarchy_is_exact_base_at_zero_gates_and_differentiable_when_enabled() -> None:
    context = _radar_metadata_context()
    domains = radar_feature_domains(context)
    base = TrainableGCNBatchStateEncoder(4, 1, "relu", random_state=7)
    hierarchy = DomainHierarchyGCNBatchStateEncoder(
        4,
        1,
        "relu",
        random_state=7,
        feature_domains=domains,
        connect_domains=True,
    )
    subset = tuple(range(0, context.n_features, 2))

    expected = base.encode_batch([subset], context)
    initial = hierarchy.encode_batch([subset], context)
    assert torch.equal(initial, expected)

    with torch.no_grad():
        hierarchy.membership_gate.fill_(0.5)
        hierarchy.inter_domain_gate.fill_(0.5)
    enabled = hierarchy.encode_batch([subset], context)
    assert not torch.equal(enabled, expected)
    enabled.sum().backward()
    assert hierarchy.membership_gate.grad is not None
    assert hierarchy.inter_domain_gate.grad is not None
    assert torch.isfinite(hierarchy.membership_gate.grad)
    assert torch.isfinite(hierarchy.inter_domain_gate.grad)


def test_domain_encoder_factory_keeps_initial_gcn_weights_paired() -> None:
    base_context = _radar_metadata_context(seed=17)
    domain_context = _radar_metadata_context(seed=17)
    shuffled_context = _radar_metadata_context(seed=17)

    base = build_batch_encoder("trained_gcn", base_context)
    domain = build_batch_encoder("domain_clique_gcn", domain_context)
    shuffled = build_batch_encoder("shuffled_domain_clique_gcn", shuffled_context)

    assert isinstance(base, TrainableGCNBatchStateEncoder)
    assert isinstance(domain, DomainHierarchyGCNBatchStateEncoder)
    assert isinstance(shuffled, DomainHierarchyGCNBatchStateEncoder)
    assert torch.equal(base.W, domain.W)
    assert torch.equal(base.bias, domain.bias)
    assert Counter(domain.feature_domains.tolist()) == Counter(shuffled.feature_domains.tolist())
    assert not torch.equal(domain.feature_domains, shuffled.feature_domains)


def test_checked_in_domain_screen_is_a_paired_four_arm_identification_matrix() -> None:
    spec = load_experiment_spec("configs/v16n/domain_gcn_screen.toml")
    matrix = ExperimentRunner(spec, seed_filter=[42]).dry_run()

    assert spec.training.feature_budget is None
    assert spec.training.steps == 120
    assert [row["method"] for row in matrix] == [
        "gcn_base",
        "gcn_domain_node",
        "gcn_domain_clique",
        "gcn_domain_clique_shuffled",
    ]
