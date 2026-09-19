"""NumPy implementation of division."""

from __future__ import annotations
import numpy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.conversion import _arithmetic_operand
from tensors.backend.numpy.conversion import _arithmetic_storage

if TYPE_CHECKING:
    from tensors._typing import Scalar
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


def divide(
    left: Tensor | Scalar,
    right: Tensor | Scalar,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Return native storage at the declared dtype."""
    left_array = _arithmetic_operand(left, dtype)
    right_array = _arithmetic_operand(right, dtype)
    # No zero test here. Floating division delivers the IEEE
    # result, and reading the denominator would force a host
    # synchronisation on every call. Integer operands are
    # rejected by the operation layer before reaching a kernel.
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        result = numpy.true_divide(left_array, right_array)
    return _arithmetic_storage(result, dtype=dtype, output_shape=output_shape)
