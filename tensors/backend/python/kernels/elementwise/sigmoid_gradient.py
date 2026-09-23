"""Reference the logistic function VJP for the Python backend."""

from __future__ import annotations
import math
from collections.abc import Iterable
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import DataType


def sigmoid_gradient(
    grad_values: Iterable[float],
    values: Iterable[float],
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Scale the upstream gradient by ``s(x) * (1 - s(x))``.

    Written as ``z / (1 + z) ** 2`` with ``z = exp(-|x|)`` rather than by
    evaluating the sigmoid and subtracting it from one. The subtraction is
    what loses the result: for ``x`` around 20 the sigmoid rounds to exactly
    ``1.0``, so ``s * (1 - s)`` is ``0.0`` where the derivative is about
    ``2e-9``. The form here keeps every digit the format has, and it is
    symmetric in ``|x|``, which is why the two tails agree exactly.
    """
    result = []
    for upstream, item in zip(grad_values, values):
        magnitude = math.exp(-item) if item >= 0.0 else math.exp(item)
        denominator = 1.0 + magnitude
        result.append(upstream * (magnitude / (denominator * denominator)))
    if len(result) != math.prod(output_shape):
        raise RuntimeError("Sigmoid VJP kernel returned an unexpected result size")
    return PythonStorage.from_values(result, dtype)
