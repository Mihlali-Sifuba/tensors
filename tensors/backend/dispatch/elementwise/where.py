"""Strict dispatch for where."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tensors.backend import config
from tensors.backend.config import BackendOperationUnsupportedError
from tensors.backend.loading import load_backend
from tensors.backend.validation import validate_backend_residency

if TYPE_CHECKING:
    from tensors.backend.storage import Storage
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


def execute_where(
    condition: Tensor,
    left: Tensor,
    right: Tensor,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Run where on the selected backend without fallback."""
    selected = config.get_backend()
    validate_backend_residency((condition, left, right), selected)
    backend: Any = load_backend(selected)

    if selected == "python":
        from tensors.utils.broadcasting import broadcast_to

        lowered_condition = broadcast_to(condition, output_shape)._data
        lowered_left = broadcast_to(left, output_shape)._data
        lowered_right = broadcast_to(right, output_shape)._data
    elif selected == "numpy":
        import numpy

        native = numpy.dtype(dtype.name)
        lowered_condition = numpy.broadcast_to(
            condition._logical_storage_for("numpy").buffer.reshape(condition.shape),
            output_shape,
        )
        lowered_left = numpy.broadcast_to(
            left._logical_storage_for("numpy").buffer.reshape(left.shape), output_shape
        )
        lowered_right = numpy.broadcast_to(
            right._logical_storage_for("numpy").buffer.reshape(right.shape),
            output_shape,
        )
        if lowered_left.dtype != native:
            lowered_left = lowered_left.astype(native, copy=False)
        if lowered_right.dtype != native:
            lowered_right = lowered_right.astype(native, copy=False)
    else:
        import cupy

        native = cupy.dtype(dtype.name)
        lowered_condition = cupy.broadcast_to(
            condition._logical_storage_for("cuda").buffer.reshape(condition.shape),
            output_shape,
        )
        lowered_left = cupy.broadcast_to(
            left._logical_storage_for("cuda").buffer.reshape(left.shape), output_shape
        )
        lowered_right = cupy.broadcast_to(
            right._logical_storage_for("cuda").buffer.reshape(right.shape), output_shape
        )
        if lowered_left.dtype != native:
            lowered_left = lowered_left.astype(native, copy=False)
        if lowered_right.dtype != native:
            lowered_right = lowered_right.astype(native, copy=False)

    result = backend.where(
        lowered_condition,
        lowered_left,
        lowered_right,
        dtype=dtype,
        output_shape=output_shape,
    )
    if result is None:
        raise BackendOperationUnsupportedError(
            f"The {selected} backend cannot execute where at dtype "
            f"{dtype.name} conformingly. where runs on the selected backend; "
            f"select another backend to run it elsewhere."
        )
    validate_backend_residency((result,), selected)
    return result
