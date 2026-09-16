"""Thin compatibility entry point for the frozen legacy-v1 implementation."""

from __future__ import annotations

import sys

from radar_ship_fs.legacy.stage2 import run_stage2_beta_sweep_trained_gcn as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
