"""NumPy implementation of the exponential."""

from __future__ import annotations
import math
import numpy
from typing import TYPE_CHECKING, Any
from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.storage import NumPyStorage
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def exp(
    values: Any,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Return native storage at the declared dtype.

    Evaluated in binary64 and rounded once on the way into storage, which is
    what the Python reference does and is why the two agree to the digit. It
    also keeps a binary32 result that overflows: ``exp(100)`` is beyond
    binary32's range, and narrowing gives ``+inf`` rather than failing.

    The result is built directly rather than through the declining
    conversion path, which treated exactly that ``+inf`` as a reason to hand
    the work to another backend. Under the execution contract a decline is a
    raise, so a legitimate overflow would have become an error.

    Host arithmetic has no flush-to-zero, so a subnormal result — reached
    below about ``x = -87`` in binary32 — survives the narrowing.
    """
    if values.dtype.kind in "iu":
        values = values.astype(numpy.float64, copy=False)
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        working = numpy.asarray(values, dtype=numpy.float64)
        result = numpy.exp(working)
        narrowed = result.astype(numpy.dtype(dtype.name), copy=False)
    storage = NumPyStorage(narrowed, dtype)
    if storage.size != math.prod(output_shape):
        raise RuntimeError("Exp kernel returned an unexpected result size")
    return storage
