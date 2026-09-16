"""Dedicated arithmetic dispatch; resolve the selection once per call."""

from __future__ import annotations
from typing import TYPE_CHECKING
from tensors.backend.config import get_backend
from tensors.backend.loading import load_backend
from tensors.backend.policy import _shape_size, should_accelerate_elementwise
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors._typing import Scalar
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


def execute_subtract(
    left: Tensor | Scalar,
    right: Tensor | Scalar,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Run subtract on the selected backend, or as Python reference."""
    from tensors.backend.python.kernels.arithmetic.subtract import subtract as reference

    selected = get_backend()
    if not should_accelerate_elementwise(selected, _shape_size(output_shape)):
        return reference(left, right, dtype=dtype, output_shape=output_shape)
    backend = load_backend(selected)
    result = backend.subtract(left, right, dtype=dtype, output_shape=output_shape)
    if result is not None:
        return result
    return reference(left, right, dtype=dtype, output_shape=output_shape)
