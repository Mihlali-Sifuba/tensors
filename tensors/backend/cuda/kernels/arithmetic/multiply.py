"""CuPy implementation of multiplication."""

from __future__ import annotations
import cupy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.cuda.conversion import _errstate
from tensors.backend.cuda.conversion import _operand
from tensors.backend.cuda.conversion import _storage

if TYPE_CHECKING:
    from tensors._typing import Scalar
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


def multiply(
    left: Tensor | Scalar,
    right: Tensor | Scalar,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Return native storage, or decline when reference semantics require it."""
    try:
        left_array = _operand(left, dtype)
        right_array = _operand(right, dtype)
    except (OverflowError, TypeError, ValueError):
        return None
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        result = cupy.multiply(left_array, right_array)
    return _storage(result, dtype=dtype, output_shape=output_shape)
