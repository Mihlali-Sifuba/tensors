"""CuPy implementation of subtraction."""

from __future__ import annotations
import cupy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.cuda.conversion import _errstate
from tensors.backend.cuda.kernels.arithmetic import _ieee32
from tensors.backend.cuda.conversion import _arithmetic_operand
from tensors.backend.cuda.conversion import _arithmetic_storage

if TYPE_CHECKING:
    from tensors._typing import Scalar
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


def subtract(
    left: Tensor | Scalar,
    right: Tensor | Scalar,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Return native storage at the declared dtype."""
    left_array = _arithmetic_operand(left, dtype)
    right_array = _arithmetic_operand(right, dtype)
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        # float32 needs the named IEEE instruction to underflow
        # gradually; float64 already does on the device.
        if dtype.typecode == "f":
            result = _ieee32.apply("subtract", left_array, right_array)
        else:
            result = cupy.subtract(left_array, right_array)
    return _arithmetic_storage(result, dtype=dtype, output_shape=output_shape)
