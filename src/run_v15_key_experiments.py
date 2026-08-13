#!/usr/bin/env python3
"""Run v15 baselines, the main RL matrix, and the best v10 tuning points."""

from __future__ import annotations

import run_basic_baselines
import run_stage2_dt_test
import run_stage2_rl_final_lr
import run_stage2_rl_selection
from stage2_rl_config import (
    DATA_VERSION,
)





def main() -> None:
    """Sequence existing resumable entry points without changing their test-sealing protocol."""
    pass
    run_basic_baselines.main()
    run_stage2_rl_selection.main()
    run_stage2_dt_test.main()
    run_stage2_rl_final_lr.main()




if __name__ == "__main__":
    main()
