"""NumPy-native softmax."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy

from tensors.backend.numpy.conversion import _storage
from tensors.backend.numpy.kernels.reductions.logsumexp_ops import _normalization_terms
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def softmax(
    value_values: Any,
    input_shape: tuple[int, ...],
    axis: int,
    *,
    dtype: DataType,
) -> Storage | None:
    """Compute stable softmax probabilities with NumPy-native values."""
    values = numpy.asarray(value_values).reshape(input_shape).astype(numpy.float64)
    all_negative_infinity = numpy.all(numpy.isneginf(values), axis=axis, keepdims=True)
    if bool(numpy.any(all_negative_infinity)):
        raise ValueError("softmax is undefined when every value along an axis is -inf")
    _, _, probabilities = _normalization_terms(values, axis)
    return _storage(probabilities, dtype=dtype, output_shape=input_shape)
