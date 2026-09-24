"""NumPy implementation of the stable log-sum-exp VJP."""

from __future__ import annotations
import numpy
from typing import Any
from tensors.backend.storage import Storage
from tensors.dtype import DataType
from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.conversion import _finite_operands
from tensors.backend.numpy.conversion import _arithmetic_storage as _storage

from tensors.backend.numpy.kernels.reductions.logsumexp_ops import _normalization_terms


def logsumexp_gradient(
    upstream: Any,
    values: Any,
    input_shape: tuple[int, ...],
    axes: tuple[int, ...],
    *,
    keepdims: bool,
    dtype: DataType,
) -> Storage | None:
    """Run a stable log-sum-exp VJP on finite values."""
    values = values.astype(numpy.float64, copy=False)
    upstream = upstream.astype(numpy.float64, copy=False)
    _, _, probabilities = _normalization_terms(values, axes)
    all_negative_infinity = numpy.all(numpy.isneginf(values), axis=axes)
    if bool(numpy.any(all_negative_infinity)):
        raise ValueError(
            "logsumexp gradient is undefined when every reduced value is -inf"
        )
    expanded_shape = tuple(
        (1 if dimension in axes else size for dimension, size in enumerate(input_shape))
    )
    try:
        expanded = upstream.reshape(expanded_shape)
    except ValueError:
        return None
    with _errstate(over="ignore", under="ignore", invalid="ignore"):
        result = expanded * probabilities
    return _storage(result, dtype=dtype, output_shape=input_shape)
