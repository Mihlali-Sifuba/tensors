"""Log-sum-exp and the shifted-exponential terms it shares.

``_normalization_terms`` lives here rather than under ``nn`` so that the
softmax family can depend on it one-way; the reverse would make the two
packages import each other.
"""

from __future__ import annotations
import cupy
from typing import Any
from tensors.backend.cuda.conversion import _errstate


def _normalization_terms(
    values: Any, axis: int | tuple[int, ...]
) -> tuple[Any, Any, Any]:
    """Return stable maxima, corrections, and probabilities."""
    maximum = cupy.max(values, axis=axis, keepdims=True)
    with _errstate(over="ignore", under="ignore", invalid="ignore"):
        deltas = values - maximum
        maxima = cupy.sum(deltas == 0.0, axis=axis, keepdims=True)
        tails = cupy.sum(
            cupy.where(deltas == 0.0, 0.0, cupy.exp(deltas)), axis=axis, keepdims=True
        )
        correction = cupy.log(maxima) + cupy.log1p(tails / maxima)
        probabilities = cupy.exp(deltas - correction)
    return (maximum, correction, probabilities)
