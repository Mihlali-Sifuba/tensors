"""NumPy-native log-softmax."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy

from tensors.backend.numpy.conversion import _errstate, _storage
from tensors.backend.numpy.kernels.reductions.logsumexp_ops import _normalization_terms
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def log_softmax(
    value_values: Any,
    input_shape: tuple[int, ...],
    axis: int,
    *,
    dtype: DataType,
) -> Storage | None:
    """Compute stable log-softmax values with NumPy-native values."""
    values = numpy.asarray(value_values).reshape(input_shape).astype(numpy.float64)
    maximum, correction, _ = _normalization_terms(values, axis)
    with _errstate(over="ignore", invalid="ignore", divide="ignore"):
        nan_group = numpy.any(numpy.isnan(values), axis=axis, keepdims=True)
        positive = numpy.isposinf(values)
        positive_count = numpy.sum(positive, axis=axis, keepdims=True)
        all_negative_infinity = numpy.all(
            numpy.isneginf(values), axis=axis, keepdims=True
        )
        ordinary = values - maximum - correction
        at_positive_infinity = numpy.where(
            positive, -numpy.log(positive_count), -numpy.inf
        )
        result = numpy.where(positive_count > 0, at_positive_infinity, ordinary)
        result = numpy.where(nan_group, numpy.nan, result)
    if bool(numpy.any(all_negative_infinity)):
        raise ValueError(
            "log_softmax is undefined when every value along an axis is -inf"
        )
    return _storage(result, dtype=dtype, output_shape=input_shape)
