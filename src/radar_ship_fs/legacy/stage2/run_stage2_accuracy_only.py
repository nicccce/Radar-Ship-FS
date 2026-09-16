#!/usr/bin/env python3
"""Run the budget-matched Full-IRFS control whose reward is inner-CV DT accuracy only."""

from __future__ import annotations

import run_stage2_budget_sweep as sweep
from stage2_rl_config import (
    ACCURACY_ONLY_DT_TEST_ROOT,
    ACCURACY_ONLY_SELECTION_ROOT,
    ACCURACY_ONLY_TABLE_PREFIX,
)


def configure_accuracy_only() -> None:
    """Configure the reusable sweep for beta=0 and lambda=0."""
    sweep.REPORT_NAME = "full_irfs_fixed_accuracy_only"
    sweep.BUDGET_SWEEP_BETA = 0.0
    sweep.BUDGET_SWEEP_VALUES = (0.0,)
    sweep.BUDGET_SWEEP_SELECTION_ROOT = ACCURACY_ONLY_SELECTION_ROOT
    sweep.BUDGET_SWEEP_DT_TEST_ROOT = ACCURACY_ONLY_DT_TEST_ROOT
    sweep.BUDGET_SWEEP_TABLE_PREFIX = ACCURACY_ONLY_TABLE_PREFIX


def main() -> None:
    configure_accuracy_only()
    sweep.main()


if __name__ == "__main__":
    main()
