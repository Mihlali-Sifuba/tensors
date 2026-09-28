"""NumPy implementation of softplus."""

from __future__ import annotations
import math
import numpy
from typing import TYPE_CHECKING, Any
from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.storage import NumPyStorage
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def softplus(
    values: Any,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Return native storage at the declared dtype.

    ``log1p(exp(-|x|)) + max(x, 0)``, the Python reference's form. Written as
    ``log(1 + exp(x))`` the exponent overflows at about ``x = 710`` while the
    result is merely ``710``; keeping the exponent negative holds every
    intermediate in range, and ``log1p`` keeps the digits an ordinary
    ``log(1 + z)`` loses for the small ``z`` that is the whole result for a
    large negative ``x``.

    Evaluation is in ``float64`` whatever the declared dtype, and the result
    is rounded once on the way into storage. Host arithmetic has no
    flush-to-zero, so a ``float32`` result in the subnormal band survives.
    """
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        working = numpy.asarray(values, dtype=numpy.float64)
        result = numpy.log1p(numpy.exp(-numpy.abs(working))) + numpy.maximum(
            working, 0.0
        )
    storage = NumPyStorage(result, dtype)
    if storage.size != math.prod(output_shape):
        raise RuntimeError("Softplus kernel returned an unexpected result size")
    return storage
