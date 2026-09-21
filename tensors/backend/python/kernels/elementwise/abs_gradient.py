"""Reference the absolute-value VJP for the Python backend."""

from __future__ import annotations
from collections.abc import Iterable
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import DataType
import math


def abs_gradient(
    grad_values: Iterable[float],
    values: Iterable[float],
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Route the upstream gradient by the sign of the primal.

    **Routing, not multiplication.** At the kink the result is canonical
    ``+0.0`` whatever the upstream is, so a negative, infinite or NaN
    upstream cannot leak a sign or a NaN into it. See
    docs/abs-semantics.md §6.
    """
    result = []
    for upstream, item in zip(grad_values, values):
        if isinstance(item, float) and math.isnan(item):
            result.append(math.nan)
        elif item > 0:
            result.append(upstream)
        elif item < 0:
            result.append(-upstream)
        else:
            result.append(0.0)
    if len(result) != math.prod(output_shape):
        raise RuntimeError("Abs VJP kernel returned an unexpected result size")
    return PythonStorage.from_values(result, dtype)
