"""Strict TOML configuration and matrix expansion for the stable runner."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import sklearn
import torch

from radar_ship_fs.experiment.artifact import ArtifactObserver, ArtifactStore
from radar_ship_fs.experiment.config import load_experiment_spec
from radar_ship_fs.experiment.runner import ExperimentRunner


def test_checked_in_stable_config_expands_only_enabled_methods() -> None:
    spec = load_experiment_spec(Path("configs/v16n/stable.toml"))
    matrix = ExperimentRunner(spec, seed_filter=[42]).dry_run()

    assert spec.algorithm_version == "stable_v1"
    assert [row["method"] for row in matrix] == ["marlfs", "full_irfs_fixed"]
    assert len({row["config_hash"] for row in matrix}) == 1
    assert len(matrix[0]["config_hash"]) == 64


def test_filters_must_reference_values_declared_in_toml() -> None:
    spec = load_experiment_spec("configs/v16n/stable.toml")
    with pytest.raises(ValueError, match="absent from TOML"):
        ExperimentRunner(spec, seed_filter=[999])
    with pytest.raises(ValueError, match="absent from TOML"):
        ExperimentRunner(spec, method_filter=["unknown"])


def test_unimplemented_credit_and_invalid_training_values_fail_fast() -> None:
    spec = load_experiment_spec("configs/v16n/stable.toml")
    with pytest.raises(ValueError, match="reserved but not implemented"):
        replace(spec, training=replace(spec.training, per_agent_credit="marginal")).validate()
    with pytest.raises(ValueError, match="batch_size cannot exceed"):
        replace(
            spec,
            training=replace(spec.training, batch_size=4096, replay_capacity=32),
        ).validate()


def test_unknown_toml_field_is_rejected(tmp_path: Path) -> None:
    source = Path("configs/v16n/stable.toml").read_text(encoding="utf-8")
    path = tmp_path / "bad.toml"
    path.write_text(source.replace("steps = 250", "steps = 250\nunknown_knob = 1"), encoding="utf-8")
    with pytest.raises(ValueError, match=r"unknown fields in \[training\]"):
        load_experiment_spec(path)


def test_artifact_root_is_reserved_for_one_algorithm_and_config(tmp_path: Path) -> None:
    spec = load_experiment_spec("configs/v16n/stable.toml")
    root = tmp_path / "results"
    ArtifactStore.prepare_root(root, spec)
    ArtifactStore.prepare_root(root, spec)

    changed = replace(spec, training=replace(spec.training, learning_rate=1e-4))
    with pytest.raises(ValueError, match="different algorithm or config"):
        ArtifactStore.prepare_root(root, changed)

    unversioned = tmp_path / "legacy-results"
    unversioned.mkdir()
    (unversioned / "selection.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="unversioned result root"):
        ArtifactStore.prepare_root(unversioned, spec)


def test_artifact_observer_recovers_jsonl_without_duplicate_steps(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "run")
    observer = ArtifactObserver(store)
    observer.on_step(SimpleNamespace(metrics=SimpleNamespace(as_dict=lambda: {"step": 1, "subset": [0]})))
    observer.on_step(SimpleNamespace(metrics=SimpleNamespace(as_dict=lambda: {"step": 2, "subset": [0, 1]})))

    resumed = ArtifactObserver(store)
    resumed.on_step(SimpleNamespace(metrics=SimpleNamespace(as_dict=lambda: {"step": 2, "subset": [1]})))
    resumed.on_checkpoint(SimpleNamespace(step=2))

    rows = store.training_jsonl_path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 2
    assert '"subset": [1]' in rows[-1]
    assert len(store.training_path.read_text(encoding="utf-8").splitlines()) == 3


def test_manifest_versions_and_data_summary_match_the_process(tmp_path: Path) -> None:
    spec = load_experiment_spec("configs/v16n/stable.toml")
    context = SimpleNamespace(
        n_features=2,
        probe=SimpleNamespace(
            n_splits=2,
            random_state=123,
            fold_indices=lambda: [
                {"fit": [0], "held_out": [1, 2]},
                {"fit": [1, 2], "held_out": [0]},
            ],
        ),
        split=SimpleNamespace(
            train=SimpleNamespace(
                X=np.zeros((3, 2)),
                y=np.array([0, 1, 1]),
                metadata={"source": "synthetic"},
            ),
            test=SimpleNamespace(X=np.zeros((1, 2))),
        ),
    )
    identity = {"development_fingerprint": "abc"}
    store = ArtifactStore(tmp_path / "manifest-run")
    store.write_manifest(
        spec=spec,
        method=spec.enabled_methods[0],
        seed=42,
        context=context,
        identity=identity,
    )

    manifest = json.loads(store.manifest_path.read_text(encoding="utf-8"))
    assert manifest["runtime"]["numpy"] == np.__version__
    assert manifest["runtime"]["pandas"] == pd.__version__
    assert manifest["runtime"]["scikit_learn"] == sklearn.__version__
    assert manifest["runtime"]["torch"] == str(torch.__version__)
    assert manifest["selection_cv"] == {
        "probe": "SimpleNamespace",
        "n_splits": 2,
        "random_state": 123,
        "fold_indices": [
            {"fit": [0], "held_out": [1, 2]},
            {"fit": [1, 2], "held_out": [0]},
        ],
    }
    assert manifest["data_summary"] == {
        "class_counts": {"0": 1, "1": 2},
        "development_fingerprint": "abc",
        "held_out_rows": 1,
        "n_features": 2,
        "search_rows": 3,
    }


def test_research_baseline_freezes_shared_ppo_and_disables_random_id_reward() -> None:
    spec = load_experiment_spec("configs/v16n/research_baseline.toml")

    assert spec.dataset.version == "v16n_2x_noise"
    assert spec.training.feature_budget == spec.ppo.feature_budget == 32
    assert spec.ppo.initialization == "mi_ordered_accept"
    assert spec.ppo.evaluation_protocol == "shared_inner_cv"
    assert spec.ppo.feature_id_reward_weight == 0.0
    assert spec.ppo.feature_id_node_feature is False
    methods = {method.name: method.encoder for method in spec.enabled_methods}
    assert methods["mi_topk"] == "relevance_topk"
    assert methods["mi_ordered_accept"] == "mi_ordered_accept"
    assert methods["forward_greedy"] == "forward_greedy"
