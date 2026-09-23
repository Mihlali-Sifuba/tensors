"""CuPy implementation of the logistic function."""

from __future__ import annotations
import math
import cupy
from typing import TYPE_CHECKING, Any
from tensors.backend.cuda.conversion import _errstate, _narrow, _widen
from tensors.backend.cuda.storage import CudaStorage
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def sigmoid(
    values: Any,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Return device storage at the declared dtype.

    The branch and the working precision are the NumPy kernel's; what is
    specific here is the two format crossings. On this toolchain ``astype``
    flushes a binary32 subnormal in **both** directions, so a subnormal
    operand would reach the comparison as zero and a subnormal result would
    be rounded away on the way into storage. ``sigmoid`` reaches the binary32
    subnormal band at about ``x = -88`` and section 5.4 requires gradual
    underflow, so both crossings go through the PTX conversions instead.

    ``_widen`` and ``_narrow`` are no-ops for any dtype that is not binary32,
    so an integer operand promoted to ``float64`` by the operation passes
    through them unchanged.
    """
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        working = _widen(cupy.asarray(values))
        magnitude = cupy.exp(-cupy.abs(working))
        result = cupy.where(
            working >= 0.0, 1.0 / (1.0 + magnitude), magnitude / (1.0 + magnitude)
        )
        narrowed = _narrow(result, cupy.dtype(dtype.name))
    storage = CudaStorage(narrowed, dtype)
    if storage.size != math.prod(output_shape):
        raise RuntimeError("Sigmoid kernel returned an unexpected result size")
    return storage
