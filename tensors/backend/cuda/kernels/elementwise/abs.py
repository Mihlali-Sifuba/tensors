"""CuPy implementation of absolute value."""

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


def abs(
    values: Any,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Return device storage at the declared dtype.

    binary32 crosses the format boundary twice here, and **both** crossings
    flush a subnormal on this toolchain. ``cupy.abs`` reads a binary32
    operand with flush-to-zero in force, so the operand arrives as zero; and
    a CuPy conversion of the binary64 magnitude back down flushes again, so
    a subnormal magnitude would be lost on the way out. Abs differs from sign
    in that second respect: its result *is* the operand's magnitude, which
    can itself be subnormal, so the PTX conversion is needed in both
    directions rather than only on the way in.

    ``float64`` already underflows gradually on the device, and an integer
    dtype computes in its own width, so a large ``int64`` keeps its value.
    The one integer input the device would answer wrongly — a signed dtype's
    least value, which wraps back to itself — has already been refused by
    `execute_abs`.
    """
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        if dtype.typecode == "f":
            result = ieee32.narrow(cupy.abs(ieee32.widen(values)))
        else:
            result = cupy.abs(values)
    storage = CudaStorage(result, dtype)
    if storage.size != math.prod(output_shape):
        raise RuntimeError("Abs kernel returned an unexpected result size")
    return storage
