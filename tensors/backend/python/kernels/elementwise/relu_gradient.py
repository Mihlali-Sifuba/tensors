"""Reference the ReLU VJP for the Python backend."""

from __future__ import annotations
from collections.abc import Iterable
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import DataType
import math


def relu_gradient(
    grad_values: Iterable[float],
    values: Iterable[float],
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Pass the upstream gradient where the primal is positive.

    **Routing, not multiplication.** On the inactive side the result is
    canonical ``+0.0`` whatever the upstream is, so an infinite or NaN
    upstream cannot manufacture a NaN there. See docs/relu-semantics.md
    section 6.
    """
    result = []
    for upstream, item in zip(grad_values, values):
        if isinstance(item, float) and math.isnan(item):
            result.append(math.nan)
        elif item > 0:
            result.append(upstream)
        else:
            result.append(0.0)
    if len(result) != math.prod(output_shape):
        raise RuntimeError("ReLU VJP kernel returned an unexpected result size")
    return PythonStorage.from_values(result, dtype)
