"""NumPy implementation of division."""

from __future__ import annotations
import numpy
from typing import TYPE_CHECKING, Any
from tensors.backend.storage import Storage
from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.conversion import _arithmetic_storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def divide(
    left: Any,
    right: Any,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Return native storage at the declared dtype."""
    # No zero test here. Floating division delivers the IEEE
    # result, and reading the denominator would force a host
    # synchronisation on every call. Integer operands are
    # rejected by the operation layer before reaching a kernel.
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        result = numpy.true_divide(left, right)
    return _arithmetic_storage(result, dtype=dtype, output_shape=output_shape)
