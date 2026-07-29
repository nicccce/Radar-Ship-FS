#!/usr/bin/env python3
"""Run v15 baselines, the main RL matrix, and the best v10 tuning points."""

from __future__ import annotations

import run_basic_baselines
import run_stage2_beta_sweep_dt_test as beta_dt_test
import run_stage2_beta_sweep_selection as beta_selection
import run_stage2_budget_sweep as budget_sweep
import run_stage2_dt_test
import run_stage2_rl_final_lr
import run_stage2_rl_selection
from stage2_rl_config import (
    BETA_SWEEP_DT_TEST_ROOT,
    BETA_SWEEP_GCN_DT_TEST_ROOT,
    BETA_SWEEP_GCN_SELECTION_ROOT,
    BETA_SWEEP_GCN_TABLE_PREFIX,
    BETA_SWEEP_SELECTION_ROOT,
    BETA_SWEEP_TABLE_PREFIX,
    DATA_VERSION,
)


def _run_beta_point(
    beta: float,
    *,
    report_name: str,
    state_encoder: str,
    selection_root,
    dt_test_root,
    table_prefix: str,
) -> None:
    """Run one old-data-selected beta value through sealed selection and DT test."""
    beta_selection.BETA_SWEEP_VALUES = (float(beta),)
    beta_dt_test.BETA_SWEEP_VALUES = (float(beta),)
    beta_selection.configure_variant(
        report_name=report_name,
        state_encoder=state_encoder,
        selection_root=selection_root,
        table_prefix=table_prefix,
    )
    beta_dt_test.configure_output(
        selection_root=selection_root,
        dt_test_root=dt_test_root,
        table_prefix=table_prefix,
    )
    beta_selection.main()
    beta_dt_test.main()


def main() -> None:
    """Sequence existing resumable entry points without changing their test-sealing protocol."""
    if DATA_VERSION != "v15":
        raise RuntimeError(
            f"this selected experiment entry requires DATA_VERSION='v15', got {DATA_VERSION!r}"
        )

    run_basic_baselines.main()
    run_stage2_rl_selection.main()
    run_stage2_dt_test.main()
    run_stage2_rl_final_lr.main()

    _run_beta_point(
        0.5,
        report_name="full_irfs_fixed",
        state_encoder="fixed",
        selection_root=BETA_SWEEP_SELECTION_ROOT,
        dt_test_root=BETA_SWEEP_DT_TEST_ROOT,
        table_prefix=BETA_SWEEP_TABLE_PREFIX,
    )
    _run_beta_point(
        0.1,
        report_name="full_irfs_trained_gcn",
        state_encoder="trained_gcn",
        selection_root=BETA_SWEEP_GCN_SELECTION_ROOT,
        dt_test_root=BETA_SWEEP_GCN_DT_TEST_ROOT,
        table_prefix=BETA_SWEEP_GCN_TABLE_PREFIX,
    )

    budget_sweep.BUDGET_SWEEP_VALUES = (0.025,)
    budget_sweep.main()


if __name__ == "__main__":
    main()
