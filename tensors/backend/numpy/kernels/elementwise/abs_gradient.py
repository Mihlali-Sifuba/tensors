"""NumPy implementation of the absolute-value VJP."""

from __future__ import annotations
import math
import numpy
from typing import TYPE_CHECKING, Any
from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.storage import NumPyStorage
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def abs_gradient(
    grad_values: Any,
    values: Any,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Return native storage at the declared dtype.

    **Routing, not multiplication.** The previous kernel materialised a
    derivative and multiplied, which made the kink's result carry the
    upstream's sign and turned an infinite or NaN upstream into NaN there.
    Selecting a literal zero instead keeps the kink at canonical ``+0.0``
    whatever the upstream is. See docs/abs-semantics.md §6.
    """
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        rectified = numpy.where(
            values > 0,
            grad_values,
            numpy.where(values < 0, -grad_values, 0),
        )
        result = numpy.where(numpy.isnan(values), values, rectified)
    storage = NumPyStorage(result, dtype)
    if storage.size != math.prod(output_shape):
        raise RuntimeError("Abs VJP kernel returned an unexpected result size")
    return storage
