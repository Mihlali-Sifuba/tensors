"""Reference the sign function VJP for the Python backend."""

from __future__ import annotations
from collections.abc import Iterable
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import DataType
import math


def sign_gradient(
    grad_values: Iterable[float],
    values: Iterable[float],
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Route each element; the sign function is piecewise constant.

    The result is **routed, not multiplied**. Away from zero the derivative
    is zero, and the specification makes the result canonical ``+0.0``
    whatever the upstream gradient is — so a negative upstream cannot make it
    ``-0.0`` and an infinite or NaN upstream cannot make it NaN. See
    docs/sign-semantics.md §6.
    """
    result = []
    for item in values:
        if item == 0:
            raise ValueError("sign derivative is undefined at zero")
        if isinstance(item, float) and math.isnan(item):
            result.append(math.nan)
        else:
            result.append(0.0)
    if len(result) != math.prod(output_shape):
        raise RuntimeError("Sign VJP kernel returned an unexpected result size")
    return PythonStorage.from_values(result, dtype)
