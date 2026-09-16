"""Dispatch for reductions, extrema indices, and shape summation."""

from __future__ import annotations
from typing import TYPE_CHECKING
from tensors.backend.loading import _backend_kernel
from tensors.backend.policy import (
    _NUMPY_REDUCTION_MIN_SIZE,
    _array_work_is_large_enough,
)
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def execute_argmin(
    value: Tensor, axis: int | None, *, keepdims: bool, output_shape: tuple[int, ...]
) -> Storage:
    """Run an argmin/argmax reduction with an accelerated backend."""
    from tensors.backend.python.kernels.reductions.argmin import argmin as reference

    if not _array_work_is_large_enough(value.size, _NUMPY_REDUCTION_MIN_SIZE):
        return reference(value, axis, keepdims=keepdims, output_shape=output_shape)
    arg_extremum = _backend_kernel("argmin")
    result = arg_extremum(value, axis, keepdims=keepdims, output_shape=output_shape)
    if result is not None:
        return result
    return reference(value, axis, keepdims=keepdims, output_shape=output_shape)
