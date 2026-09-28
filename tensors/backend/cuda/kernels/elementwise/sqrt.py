"""CuPy implementation of square root."""

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


def sqrt(
    values: Any,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Return device storage at the declared dtype.

    ``cupy.sqrt`` reads a binary32 operand with flush-to-zero in force, so a
    subnormal arrives as zero and its root is reported as zero. Widening
    through the PTX conversion first keeps the operand, and the root is then
    taken in binary64.

    Narrowing that binary64 root back down yields the *correctly rounded*
    binary32 result. The conversion still rounds; what it does not do is
    disagree with rounding the true root directly, because binary64 carries
    53 significand bits and square root needs only ``2p + 2 = 50`` for the
    two roundings to agree. No binary32 PTX root instruction is needed. Unlike
    abs, the narrowing also cannot lose a subnormal, because the square root
    of even the smallest binary32 subnormal is a normal number.

    An integer operand is converted to binary64 explicitly before the root is
    taken, which is the specified conversion rather than a promise about the
    mathematical integer. A negative operand is a value, not an error: the
    device returns NaN and nothing here checks for it, so no operand value is
    read back to the host.
    """
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        if dtype.typecode == "f":
            result = ieee32.narrow(cupy.sqrt(ieee32.widen(values)))
        else:
            if values.dtype.kind in "iu":
                values = values.astype(cupy.float64, copy=False)
            result = cupy.sqrt(values)
    storage = CudaStorage(result, dtype)
    if storage.size != math.prod(output_shape):
        raise RuntimeError("Sqrt kernel returned an unexpected result size")
    return storage
