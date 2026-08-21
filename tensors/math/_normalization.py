"""Shared stable normalization helpers for softmax-family operations."""

from __future__ import annotations

import math


def shifted_normalization(
    values: list[float],
) -> tuple[float, float, list[float], list[float]]:
    """Return maximum, log-normalizer shift, probabilities, and complements.

    ``values`` must be finite and non-empty.  Keeping the logarithmic
    normalizer relative to the maximum preserves tails that would disappear
    if a tiny correction were first added to a large absolute value.
    """
    maximum = max(values)
    deltas = [value - maximum for value in values]
    maxima = sum(delta == 0.0 for delta in deltas)
    tail = math.fsum(math.exp(delta) for delta in deltas if delta != 0.0)
    correction = math.log(maxima) + math.log1p(tail / maxima)
    probabilities = [math.exp(delta - correction) for delta in deltas]
    complements = [1.0 - probability for probability in probabilities]
    if maxima == 1:
        maximum_index = deltas.index(0.0)
        complements[maximum_index] = -math.expm1(-correction)
    return maximum, correction, probabilities, complements


__all__ = ["shifted_normalization"]
