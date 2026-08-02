"""Stable, namespaced implementation of the Radar-Ship feature-selection stack.

The original top-level packages remain available as the frozen ``legacy_v1``
implementation.  New development lives below :mod:`radar_ship_fs` so the
stable training core can evolve without accumulating compatibility branches.
"""

__version__ = "0.1.0"
