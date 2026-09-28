"""CUDA-native softmax."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cupy

from tensors.backend.cuda.conversion import _storage, _widen
from tensors.backend.cuda.kernels.reductions.logsumexp_ops import _normalization_terms
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
    """Compute stable softmax probabilities with device-native values."""
    values = _widen(cupy.asarray(value_values).reshape(input_shape))
    all_negative_infinity = cupy.all(cupy.isneginf(values), axis=axis, keepdims=True)
    if bool(cupy.any(all_negative_infinity)):
        raise ValueError("softmax is undefined when every value along an axis is -inf")
    _, _, probabilities = _normalization_terms(values, axis)
    return _storage(probabilities, dtype=dtype, output_shape=input_shape)
