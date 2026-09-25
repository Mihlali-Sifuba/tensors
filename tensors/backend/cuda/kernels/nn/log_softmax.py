"""CUDA-native log-softmax."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cupy

from tensors.backend.cuda.conversion import _errstate, _storage, _widen
from tensors.backend.cuda.kernels.reductions.logsumexp_ops import _normalization_terms
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
    """Compute stable log-softmax values with device-native values."""
    values = _widen(cupy.asarray(value_values).reshape(input_shape))
    maximum, correction, _ = _normalization_terms(values, axis)
    with _errstate(over="ignore", invalid="ignore", divide="ignore"):
        nan_group = cupy.any(cupy.isnan(values), axis=axis, keepdims=True)
        positive = cupy.isposinf(values)
        positive_count = cupy.sum(positive, axis=axis, keepdims=True)
        all_negative_infinity = cupy.all(
            cupy.isneginf(values), axis=axis, keepdims=True
        )
        ordinary = values - maximum - correction
        at_positive_infinity = cupy.where(
            positive, -cupy.log(positive_count), -cupy.inf
        )
        result = cupy.where(positive_count > 0, at_positive_infinity, ordinary)
        result = cupy.where(nan_group, cupy.nan, result)
    if bool(cupy.any(all_negative_infinity)):
        raise ValueError(
            "log_softmax is undefined when every value along an axis is -inf"
        )
    return _storage(result, dtype=dtype, output_shape=input_shape)
