"""Frozen legacy-v1 compatibility boundary.

New training facilities must not import this package.  It exists only to expose the
historical selector through the common subset-selection contract.
"""

from radar_ship_fs.legacy.selector import LegacySelector, build_legacy_selector

__all__ = ["LegacySelector", "build_legacy_selector"]
