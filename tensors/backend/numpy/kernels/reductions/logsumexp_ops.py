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
    with _errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        nan_group = numpy.any(numpy.isnan(values), axis=axis, keepdims=True)
        positive_infinity = numpy.isposinf(values)
        positive_count = numpy.sum(positive_infinity, axis=axis, keepdims=True)
        all_negative_infinity = numpy.all(
            numpy.isneginf(values), axis=axis, keepdims=True
        )
        deltas = values - maximum
        maxima = numpy.sum(deltas == 0.0, axis=axis, keepdims=True)
        tails = numpy.sum(
            numpy.where(deltas == 0.0, 0.0, numpy.exp(deltas)),
            axis=axis,
            keepdims=True,
        )
        ordinary_correction = numpy.log(maxima) + numpy.log1p(tails / maxima)
        special = (positive_count > 0) | all_negative_infinity
        correction = numpy.where(special, 0.0, ordinary_correction)
        correction = numpy.where(nan_group, numpy.nan, correction)
        ordinary_probabilities = numpy.exp(deltas - ordinary_correction)
        infinity_probabilities = positive_infinity / numpy.where(
            positive_count == 0, 1, positive_count
        )
        probabilities = numpy.where(
            positive_count > 0, infinity_probabilities, ordinary_probabilities
        )
        probabilities = numpy.where(nan_group, numpy.nan, probabilities)
    return (maximum, correction, probabilities)
