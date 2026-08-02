#!/usr/bin/env python3
"""One-click trained-GCN beta sweep followed by held-out DT validation."""

from __future__ import annotations

import run_stage2_beta_sweep_dt_test as dt_test
import run_stage2_beta_sweep_selection as selection
from stage2_rl_config import (
    BETA_SWEEP_GCN_DT_TEST_ROOT,
    BETA_SWEEP_GCN_SELECTION_ROOT,
    BETA_SWEEP_GCN_TABLE_PREFIX,
)


def _configure_trained_gcn() -> None:
    selection.configure_variant(
        report_name="full_irfs_trained_gcn",
        state_encoder="trained_gcn",
        selection_root=BETA_SWEEP_GCN_SELECTION_ROOT,
        table_prefix=BETA_SWEEP_GCN_TABLE_PREFIX,
    )
    dt_test.configure_output(
        selection_root=BETA_SWEEP_GCN_SELECTION_ROOT,
        dt_test_root=BETA_SWEEP_GCN_DT_TEST_ROOT,
        table_prefix=BETA_SWEEP_GCN_TABLE_PREFIX,
    )


def main() -> None:
    _configure_trained_gcn()
    selection.main()
    dt_test.main()


if __name__ == "__main__":
    main()
