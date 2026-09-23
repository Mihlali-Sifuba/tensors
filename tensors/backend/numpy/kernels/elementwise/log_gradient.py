"""NumPy implementation of the natural logarithm VJP."""

from __future__ import annotations
import math
import numpy
from typing import TYPE_CHECKING, Any
from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.storage import NumPyStorage
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def log_gradient(
    grad_values: Any,
    values: Any,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Return native storage at the declared dtype.

    ``G / x``, an ordinary division, governed by
    `docs/arithmetic-semantics.md` section 7.2: the IEEE result stands and
    nothing raises. Dividing once is deliberate — forming ``1 / x`` and
    multiplying rounds twice and loses a digit for no benefit.

    No domain check runs here. The forward refuses a non-positive operand,
    so a primal arriving from a completed forward pass is positive, and
    repeating the test would cost a reduction on every reverse pass.
    """
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        upstream = numpy.asarray(grad_values, dtype=numpy.float64)
        working = numpy.asarray(values, dtype=numpy.float64)
        result = upstream / working
        narrowed = result.astype(numpy.dtype(dtype.name), copy=False)
    storage = NumPyStorage(narrowed, dtype)
    if storage.size != math.prod(output_shape):
        raise RuntimeError("Log VJP kernel returned an unexpected result size")
    return storage
