"""Centered, scaled deviations for variance and standard deviation.

Both statistics need the same intermediate: values centered on their mean and
divided by a safe scale, so that squaring them cannot overflow. The primitive
takes a value sequence and the flat indices of one reduction group, which
keeps it independent of how those values are stored.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from tensors.utils.summation import stable_float_mean


def scaled_deviations(
    values: Sequence[float],
    group: Sequence[int],
) -> tuple[float, list[float], float]:
    """Return a safe scale, centered scaled values, and their deviation.

    ``values`` is any logical value sequence and ``group`` holds the flat
    indices contributing to one reduction output. Centering is retried against
    the raw magnitudes when subtracting the mean itself leaves the float range.
    """
    selected = [float(values[index]) for index in group]
    if any((not math.isfinite(item) for item in selected)):
        return (math.nan, [math.nan] * len(selected), math.nan)
    count = len(selected)
    average = stable_float_mean(selected)
    centered = [item - average for item in selected]
    if all((math.isfinite(item) for item in centered)):
        scale = max((abs(item) for item in centered), default=0.0)
        if scale == 0.0:
            return (0.0, [0.0] * count, 0.0)
        normalized_centered = [item / scale for item in centered]
    else:
        scale = max((abs(item) for item in selected), default=0.0)
        normalized = [item / scale for item in selected]
        normalized_average = stable_float_mean(normalized)
        normalized_centered = [item - normalized_average for item in normalized]
    variance = math.fsum((item * item / count for item in normalized_centered))
    return (scale, normalized_centered, math.sqrt(variance))


__all__ = ["scaled_deviations"]
