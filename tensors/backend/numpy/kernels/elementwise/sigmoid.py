"""NumPy implementation of the logistic function."""

from __future__ import annotations
import math
import numpy
from typing import TYPE_CHECKING, Any
from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.storage import NumPyStorage
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def sigmoid(
    values: Any,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Return native storage at the declared dtype.

    The branch is the Python reference's, stated as a selection: each side is
    evaluated where its exponent is negative, so neither ``exp`` overflows.
    Both sides are computed and one is discarded, which is what a vectorised
    form costs; the discarded side is finite by construction, so nothing is
    lost to the arithmetic that produced it.

    Evaluation is in ``float64`` whatever the declared dtype, and the result
    is rounded once on the way into storage. Host arithmetic has no
    flush-to-zero, so a ``float32`` result in the subnormal band — which
    ``sigmoid`` reaches at about ``x = -88`` — survives that rounding.

    An integer operand arrives promoted by the operation, so nothing here
    decides a dtype; ``asarray`` only puts the values in working precision.
    """
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        working = numpy.asarray(values, dtype=numpy.float64)
        magnitude = numpy.exp(-numpy.abs(working))
        result = numpy.where(
            working >= 0.0, 1.0 / (1.0 + magnitude), magnitude / (1.0 + magnitude)
        )
    storage = NumPyStorage(result, dtype)
    if storage.size != math.prod(output_shape):
        raise RuntimeError("Sigmoid kernel returned an unexpected result size")
    return storage
