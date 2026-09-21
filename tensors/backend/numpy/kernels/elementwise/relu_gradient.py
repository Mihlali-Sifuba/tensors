"""NumPy implementation of the ReLU VJP."""

from __future__ import annotations
import math
import numpy
from typing import TYPE_CHECKING, Any
from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.storage import NumPyStorage
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def relu_gradient(
    grad_values: Any,
    values: Any,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Return native storage at the declared dtype.

    **Routing, not multiplication.** The previous kernel materialised a
    derivative and multiplied, which turned an infinite or NaN upstream on
    the inactive side into NaN and let a negative upstream leave ``-0.0``
    there. Selecting a literal zero keeps the inactive side at canonical
    ``+0.0``. See docs/relu-semantics.md section 6.

    Host arithmetic has no flush-to-zero, so a positive subnormal primal is
    correctly seen as positive and a subnormal upstream passes through.
    """
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        rectified = numpy.where(values > 0, grad_values, 0)
        result = numpy.where(numpy.isnan(values), values, rectified)
    storage = NumPyStorage(result, dtype)
    if storage.size != math.prod(output_shape):
        raise RuntimeError("ReLU VJP kernel returned an unexpected result size")
    return storage
