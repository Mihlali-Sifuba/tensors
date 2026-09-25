"""CUDA implementation of outer-product VJPs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cupy

from tensors.backend.cuda.conversion import _errstate, _storage, _widen

if TYPE_CHECKING:
    from tensors.backend.storage import Storage
    from tensors.dtype import DataType


def outer_gradient(
    grad_values: Any,
    left_values: Any,
    right_values: Any,
    *,
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
    dtype: DataType,
    needs_input_grad: tuple[bool, ...] = (True, True),
) -> tuple[Storage | None, Storage | None] | None:
    """Execute requested outer VJPs with CUDA-native contractions."""
    if dtype.kind != "floating":
        return None
    try:
        upstream = _widen(grad_values)
        left = _widen(left_values)
        right = _widen(right_values)
    except (TypeError, ValueError):
        return None
    if not bool(
        cupy.all(cupy.isfinite(upstream))
        & cupy.all(cupy.isfinite(left))
        & cupy.all(cupy.isfinite(right))
    ):
        return None
    need_left, need_right = needs_input_grad
    with _errstate(over="ignore", under="ignore", invalid="ignore"):
        left_result = cupy.matmul(upstream, right) if need_left else None
        right_result = cupy.matmul(left, upstream) if need_right else None
    if left_result is not None and bool(cupy.any(~cupy.isfinite(left_result))):
        return None
    if right_result is not None and bool(cupy.any(~cupy.isfinite(right_result))):
        return None
    left_storage = (
        _storage(
            cupy.where(left_result == 0.0, 0.0, left_result),
            dtype=dtype,
            output_shape=left_shape,
        )
        if left_result is not None
        else None
    )
    right_storage = (
        _storage(
            cupy.where(right_result == 0.0, 0.0, right_result),
            dtype=dtype,
            output_shape=right_shape,
        )
        if right_result is not None
        else None
    )
    if (need_left and left_storage is None) or (need_right and right_storage is None):
        return None
    return (left_storage, right_storage)
