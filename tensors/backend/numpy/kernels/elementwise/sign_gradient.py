"""NumPy implementation of the sign function VJP."""

from __future__ import annotations
import math
import numpy
from typing import TYPE_CHECKING, Any
from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.storage import NumPyStorage
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def sign_gradient(
    grad_values: Any,
    values: Any,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Return native storage at the declared dtype.

    The result is **routed, not multiplied**. The previous kernel built a
    derivative and multiplied the upstream gradient by it, which made a
    negative upstream produce ``-0.0`` and an infinite or NaN upstream
    produce NaN — neither of which the Python reference did. The
    specification settles that disagreement in favour of routing: the result
    is canonical ``+0.0`` away from NaN, whatever the upstream is. See
    docs/sign-semantics.md §6.
    """
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        if bool(numpy.any(values == 0)):
            raise ValueError("sign derivative is undefined at zero")
        result = numpy.where(numpy.isnan(values), values, 0)
    storage = NumPyStorage(result, dtype)
    if storage.size != math.prod(output_shape):
        raise RuntimeError("Sign VJP kernel returned an unexpected result size")
    return storage
