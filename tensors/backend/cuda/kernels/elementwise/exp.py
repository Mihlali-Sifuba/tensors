"""CuPy implementation of the exponential."""

from __future__ import annotations
import math
import cupy
from typing import TYPE_CHECKING, Any
from tensors.backend.cuda.conversion import _errstate, _narrow, _widen
from tensors.backend.cuda.storage import CudaStorage
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def exp(
    values: Any,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Return device storage at the declared dtype.

    Evaluated in binary64 and narrowed once, as on the other two backends.
    What is specific here is the two format crossings: ``astype`` flushes a
    binary32 subnormal in both directions on this toolchain, and exp reaches
    the binary32 subnormal band below about ``x = -87``, so both crossings
    go through the PTX conversions. ``_widen`` and ``_narrow`` are no-ops for
    any dtype that is not binary32.

    Narrowing an overflowing binary64 result gives ``+inf``, which is the
    specified answer. The result is built directly rather than through the
    declining conversion path, which treated that ``+inf`` as a reason to
    hand the work to another backend; under the execution contract a decline
    is a raise, so a legitimate overflow would have become an error.
    """
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        if values.dtype.kind in "iu":
            values = values.astype(cupy.float64, copy=False)
        working = _widen(values)
        result = cupy.exp(working)
        narrowed = _narrow(result, cupy.dtype(dtype.name))
    storage = CudaStorage(narrowed, dtype)
    if storage.size != math.prod(output_shape):
        raise RuntimeError("Exp kernel returned an unexpected result size")
    return storage
