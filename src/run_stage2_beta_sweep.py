#!/usr/bin/env python3
"""One-click beta sweep: finish every RL selection, then run held-out DT validation."""

from run_stage2_beta_sweep_dt_test import main as run_dt_test
from run_stage2_beta_sweep_selection import main as run_selection


def main() -> None:
    run_selection()
    run_dt_test()


if __name__ == "__main__":
    main()
