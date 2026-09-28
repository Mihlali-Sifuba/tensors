"""NumPy implementation of the sign function."""

from __future__ import annotations
import math
import numpy
from typing import TYPE_CHECKING, Any
from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.storage import NumPyStorage
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def sign(
    values: Any,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Return native storage at the declared dtype.

    ``numpy.sign`` classifies in the dtype it is given, so an integer array
    stays integral and a large ``int64`` is never routed through ``float64``.
    It already returns canonical positive zero for both signed zeros and
    propagates NaN.
    """
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        result = numpy.sign(values)
    storage = NumPyStorage(result, dtype)
    if storage.size != math.prod(output_shape):
        raise RuntimeError("Sign kernel returned an unexpected result size")
    return storage
