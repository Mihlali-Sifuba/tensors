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
    with _errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        nan_group = cupy.any(cupy.isnan(values), axis=axis, keepdims=True)
        positive_infinity = cupy.isposinf(values)
        positive_count = cupy.sum(positive_infinity, axis=axis, keepdims=True)
        all_negative_infinity = cupy.all(
            cupy.isneginf(values), axis=axis, keepdims=True
        )
        deltas = values - maximum
        maxima = cupy.sum(deltas == 0.0, axis=axis, keepdims=True)
        tails = cupy.sum(
            cupy.where(deltas == 0.0, 0.0, cupy.exp(deltas)),
            axis=axis,
            keepdims=True,
        )
        ordinary_correction = cupy.log(maxima) + cupy.log1p(tails / maxima)
        special = (positive_count > 0) | all_negative_infinity
        correction = cupy.where(special, 0.0, ordinary_correction)
        correction = cupy.where(nan_group, cupy.nan, correction)
        ordinary_probabilities = cupy.exp(deltas - ordinary_correction)
        infinity_probabilities = positive_infinity / cupy.where(
            positive_count == 0, 1, positive_count
        )
        probabilities = cupy.where(
            positive_count > 0, infinity_probabilities, ordinary_probabilities
        )
        probabilities = cupy.where(nan_group, cupy.nan, probabilities)
    return (maximum, correction, probabilities)
