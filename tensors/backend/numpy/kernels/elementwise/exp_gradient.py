"""NumPy implementation of the exponential VJP."""

from __future__ import annotations
import math
import numpy
from typing import TYPE_CHECKING, Any
from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.storage import NumPyStorage
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def exp_gradient(
    grad_values: Any,
    values: Any,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Return native storage at the declared dtype.

    Exp is its own derivative, so this evaluates the forward's function and
    multiplies. Both are done in binary64 and rounded once, and an
    overflowing derivative is ``+inf`` rather than a reason to decline — the
    product that follows is then ordinary IEEE arithmetic, so a zero
    upstream against an infinite derivative gives NaN.
    """
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        upstream = numpy.asarray(grad_values, dtype=numpy.float64)
        working = numpy.asarray(values, dtype=numpy.float64)
        result = upstream * numpy.exp(working)
        narrowed = result.astype(numpy.dtype(dtype.name), copy=False)
    storage = NumPyStorage(narrowed, dtype)
    if storage.size != math.prod(output_shape):
        raise RuntimeError("Exp VJP kernel returned an unexpected result size")
    return storage
