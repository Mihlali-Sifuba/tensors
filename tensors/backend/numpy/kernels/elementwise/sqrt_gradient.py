"""NumPy implementation of the square root VJP."""

from __future__ import annotations
import math
import numpy
from typing import TYPE_CHECKING, Any
from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.storage import NumPyStorage
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def sqrt_gradient(
    grad_values: Any,
    values: Any,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Return native storage at the declared dtype.

    The three steps of `docs/sqrt-semantics.md` section 6 are written out in
    order and each rounds in the declared dtype: NumPy computes in the
    array's own format, so a ``float32`` VJP is never evaluated in
    ``float64`` and narrowed once at the end.

    A negative primal needs no branch. ``numpy.sqrt`` already answers NaN
    for it, and NaN carries through the multiplication and the division, so
    only the zero primal is detected.
    """
    native = numpy.dtype(dtype.name)
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        if bool(numpy.any(values == 0)):
            raise ValueError("sqrt derivative is undefined at zero")
        root = numpy.sqrt(values)
        denominator = native.type(2) * root
        result = grad_values / denominator
    storage = NumPyStorage(result, dtype)
    if storage.size != math.prod(output_shape):
        raise RuntimeError("Sqrt VJP kernel returned an unexpected result size")
    return storage
