"""Accuracy metrics (COMP-004) — Best and Average Accuracy over a window.

Reduces a per-step accuracy series into the two headline numbers the study uses to compare every
reinforced method: Best Accuracy (the maximum over the window) and Average Accuracy (the mean over
the window). The per-step series itself is produced by the exploration loop (PHASE-002); this
component only summarizes it, so it takes the series as an argument and stays free of the loader,
split, and probe.

The window is an optional ``(start, end)`` pair interpreted as a half-open slice ``[start:end]`` —
the same convention as Python slicing, so a configured ``metric_window`` (ASM-006, COMP-025) flows
straight in with no off-by-one translation. ``None`` means the full series (ASM-006 default). The
wiring layer passes ``config.metric_window``; this module does not import :mod:`config`.

Satisfies COMP-004 -> REQ-004.
"""

from __future__ import annotations

from typing import Optional, Sequence


def compute_windowed_metrics(
    accuracy_series: Sequence[float],
    window: Optional[tuple[int, int]] = None,
) -> tuple[float, float]:
    """Return ``(best_accuracy, average_accuracy)`` over the selected window.

    ``accuracy_series`` is the per-step accuracy sequence. ``window`` is an optional ``(start,
    end)`` pair applied as a half-open slice ``accuracy_series[start:end]``; when ``None`` (ASM-006
    default) the full series is used. Best Accuracy is the maximum over the window and Average
    Accuracy is its mean, both returned as plain ``float``.

    Raises ``ValueError`` if ``accuracy_series`` is empty or if the resolved window selects no
    elements, so the metrics are never derived from an empty slice.
    """
    series = list(accuracy_series)
    if not series:
        raise ValueError("accuracy_series must contain at least one accuracy value")

    if window is None:
        selected = series
    else:
        start, end = window
        selected = series[start:end]
        if not selected:
            raise ValueError(f"window {window!r} selects no elements from the series")

    best_accuracy = float(max(selected))
    average_accuracy = float(sum(selected) / len(selected))
    return best_accuracy, average_accuracy
