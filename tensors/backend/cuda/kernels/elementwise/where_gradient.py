"""CUDA implementation of the where VJP."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import cupy

from tensors.backend.cuda.conversion import _narrow, _widen
from tensors.backend.cuda.storage import CudaStorage
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def where_gradient(
    grad_values: Any,
    condition_values: Any,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
    needs_input_grad: tuple[bool, ...] = (True, True),
) -> tuple[Storage | None, Storage | None]:
    """Route device upstream values into the requested data branches."""
    expected_shape = tuple(output_shape)
    if any(
        tuple(values.shape) != expected_shape
        for values in (grad_values, condition_values)
    ):
        raise RuntimeError(
            "where_gradient kernel received operands not prepared for output_shape"
        )
    need_left, need_right = needs_input_grad
    selected = condition_values != 0
    upstream = _widen(grad_values)
    left = None
    if need_left:
        left = CudaStorage(
            _narrow(cupy.where(selected, upstream, 0.0), cupy.dtype(dtype.name)),
            dtype,
        )
    right = None
    if need_right:
        right = CudaStorage(
            _narrow(cupy.where(selected, 0.0, upstream), cupy.dtype(dtype.name)),
            dtype,
        )
    expected = math.prod(output_shape)
    if (left is not None and left.size != expected) or (
        right is not None and right.size != expected
    ):
        raise RuntimeError("where VJP kernel returned an unexpected result size")
    return left, right
