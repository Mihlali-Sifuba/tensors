"""Shift-and-normalize arithmetic for the softmax family.

Softmax, log-softmax, log-sum-exp and the cross-entropy losses all need the
same shifted normalizer, and so do their derivative rules. The primitive
works on a plain list of finite floats so that neither the kernels evaluating
those operations nor the graph code differentiating them has to depend on the
other.
"""

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
