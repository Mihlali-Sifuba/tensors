"""CuPy implementation of the sign function."""

from __future__ import annotations
import math
import cupy
from typing import TYPE_CHECKING, Any
from tensors.backend.cuda.conversion import _errstate
from tensors.backend.cuda.kernels.arithmetic import ieee32
from tensors.backend.cuda.storage import CudaStorage
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def sign(
    values: Any,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Return device storage at the declared dtype.

    ``cupy.sign`` reads a binary32 operand with flush-to-zero in force, so a
    subnormal arrives as zero and is classified as zero rather than as ±1.
    Widening through the PTX conversion first keeps the operand, and the
    classification then runs in binary64. Narrowing the result back is exact
    whatever the operand was, because sign only ever produces -1, 0, 1 or NaN.
    ``float64`` already underflows gradually on the device, and an integer
    dtype classifies in its own width so a large ``int64`` keeps its value.
    """
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        if dtype.typecode == "f":
            result = cupy.sign(ieee32.widen(values))
        else:
            result = cupy.sign(values)
    storage = CudaStorage(result, dtype)
    if storage.size != math.prod(output_shape):
        raise RuntimeError("Sign kernel returned an unexpected result size")
    return storage
