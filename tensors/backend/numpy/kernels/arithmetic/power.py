"""NumPy implementation of exponentiation."""

from __future__ import annotations
import numpy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.conversion import _operand
from tensors.backend.numpy.conversion import _storage
from tensors.backend.numpy.conversion import _finite_operands

if TYPE_CHECKING:
    from tensors._typing import Scalar
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


def power(
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
        result = numpy.power(left_array, right_array)
    if (
        dtype.kind == "floating"
        and _finite_operands(left_array, right_array)
        and (not bool(numpy.all(numpy.isfinite(result))))
    ):
        return None
    return _storage(result, dtype=dtype, output_shape=output_shape)
