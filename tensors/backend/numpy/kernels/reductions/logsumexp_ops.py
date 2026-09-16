"""Log-sum-exp and the shifted-exponential terms it shares.

``_normalization_terms`` lives here rather than under ``nn`` so that the
softmax family can depend on it one-way; the reverse would make the two
packages import each other.
"""

from __future__ import annotations
import numpy
from typing import Any
from tensors.backend.numpy.conversion import _errstate


def _normalization_terms(
    values: Any, axis: int | tuple[int, ...]
) -> tuple[Any, Any, Any]:
    """Return stable maxima, corrections, and probabilities."""
    maximum = numpy.max(values, axis=axis, keepdims=True)
    with _errstate(over="ignore", under="ignore", invalid="ignore"):
        deltas = values - maximum
        maxima = numpy.sum(deltas == 0.0, axis=axis, keepdims=True)
        tails = numpy.sum(
            numpy.where(deltas == 0.0, 0.0, numpy.exp(deltas)), axis=axis, keepdims=True
        )
        correction = numpy.log(maxima) + numpy.log1p(tails / maxima)
        probabilities = numpy.exp(deltas - correction)
    return (maximum, correction, probabilities)
