"""Robust summary statistics, and an explicit verdict on noise.

A benchmark sample set is small and its outliers are real events rather than
measurement error, so the median and the median absolute deviation are what
the summary leads with. The mean and standard deviation are reported beside
them because they answer different questions, not because either is the
headline.

Every sample is kept. A report is regenerated from records, so discarding the
raw observations would make a stored run unanalysable later.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from typing import Any


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


#: A case whose samples deviate by more than this fraction of the median is
#: reported as noisy rather than quietly averaged.
NOISE_THRESHOLD_PERCENT = 15.0


def summarize(samples: Sequence[float]) -> dict[str, Any]:
    """Return robust and classical statistics plus a noise verdict."""
    if not samples:
        return {}
    median = statistics.median(samples)
    deviations = [abs(sample - median) for sample in samples]
    median_absolute_deviation = statistics.median(deviations)
    variability = 0.0 if median == 0.0 else median_absolute_deviation / median * 100.0
    return {
        "median_seconds": median,
        "mean_seconds": statistics.fmean(samples),
        "stdev_seconds": (statistics.stdev(samples) if len(samples) > 1 else 0.0),
        "min_seconds": min(samples),
        "max_seconds": max(samples),
        "p95_seconds": _percentile(samples, 0.95),
        "median_absolute_deviation_seconds": median_absolute_deviation,
        "variability_percent": variability,
        "noisy": variability > NOISE_THRESHOLD_PERCENT,
        "sample_count": len(samples),
        "samples_seconds": list(samples),
    }


__all__ = ["NOISE_THRESHOLD_PERCENT", "summarize"]
