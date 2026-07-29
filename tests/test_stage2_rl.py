"""阶段 2 RL 搜索与最终 LR 解耦入口的轻量测试。"""

from __future__ import annotations

import json

import numpy as np

from data.splitter import Partition, Split
from harness.contract import SelectionContext, StepRecord
from run_stage2_rl_final_lr import _development_and_test, _load_selection
from run_stage2_rl_selection import _config_for_encoder, _method_signature, _trajectory_rows


def test_trajectory_rows_capture_accuracy_and_subset_stability() -> None:
    steps = (
        StepRecord(subset=(0, 1), accuracy=0.7),
        StepRecord(subset=(1, 2), accuracy=0.8),
        StepRecord(subset=(1, 2), accuracy=0.75),
    )

    rows = _trajectory_rows(
        steps,
        selected_subset=(1, 2),
        original_feature_ids=(10, 20, 30, 40),
        elapsed_by_step=(1.0, 3.0, 4.5),
        rolling_window=2,
        fold_accuracies_by_step=((0.6, 0.8), (0.7, 0.9), (0.7, 0.8)),
    )

    assert [row["running_best_inner_cv_accuracy"] for row in rows] == [0.7, 0.8, 0.8]
    assert [row["is_new_best_accuracy"] for row in rows] == [True, True, False]
    assert np.isclose(rows[0]["dt_inner_cv_fold_std"], 0.1)
    assert rows[0]["jaccard_with_previous"] is None
    assert rows[1]["jaccard_with_previous"] == 1 / 3
    assert rows[2]["jaccard_with_previous"] == 1.0
    assert rows[1]["jaccard_with_selected_subset"] == 1.0
    assert rows[1]["changed_feature_count"] == 2
    assert rows[2]["selected_original_feature_ids"] == [20, 30]
    assert rows[2]["step_elapsed_seconds"] == 1.5


def test_resume_signature_is_stable_across_json_round_trip() -> None:
    config = _config_for_encoder("trained_gcn")
    signature = _method_signature(
        seed=42,
        report_name="full_irfs_trained_gcn",
        engine_name="full_irfs",
        state_encoder="trained_gcn",
        config=config,
    )

    assert json.loads(json.dumps(signature)) == signature
    assert signature["protocol_version"] == 3
    assert signature["inner_cv_folds"] == 5
    assert signature["effective_irfs_config"]["hybrid_switch_step"] == 83
    assert signature["effective_irfs_config"]["hybrid_withdraw_step"] == 166


def test_final_lr_combines_train_and_validation_before_source_test() -> None:
    X = np.arange(60, dtype=np.float32).reshape(12, 5)
    y = np.asarray([-1, 1] * 6)
    names = [f"feature_{index}" for index in range(5)]

    def partition(indices: list[int]) -> Partition:
        index_array = np.asarray(indices)
        return Partition(X[index_array], y[index_array], index_array, names)

    context = SelectionContext(
        split=Split(partition(list(range(6))), partition([6, 7, 8]), partition([9, 10, 11])),
        probe=None,
        config=None,
        rng=None,
    )

    X_development, y_development, X_test, y_test = _development_and_test(context)

    np.testing.assert_array_equal(X_development, X[:9])
    np.testing.assert_array_equal(y_development, y[:9])
    np.testing.assert_array_equal(X_test, X[9:])
    np.testing.assert_array_equal(y_test, y[9:])


def test_final_lr_loader_requires_isolated_selection(tmp_path, monkeypatch) -> None:
    root = tmp_path / "selections"
    path = root / "seed-42" / "marlfs" / "selection.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "experiment_signature": {
                    "dataset": "radar_ship",
                    "seed": 42,
                    "report_name": "marlfs",
                },
                "protocol": {
                    "test_used_during_selection": False,
                    "lr_final_called": False,
                },
                "selected_clean_indices": [0, 2],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("run_stage2_rl_final_lr.SELECTION_ROOT", root)

    artifact, loaded_path = _load_selection(42, "marlfs")

    assert artifact["selected_clean_indices"] == [0, 2]
    assert loaded_path == path
