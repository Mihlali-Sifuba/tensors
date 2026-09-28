"""Reference the softplus VJP for the Python backend."""

from __future__ import annotations
import math
from collections.abc import Iterable
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import DataType


def softplus_gradient(
    grad_values: Iterable[float],
    values: Iterable[float],
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Scale the upstream gradient by ``sigmoid(x)``.

    Softplus is the integral of the logistic function, so its derivative is
    that function exactly, evaluated from whichever side keeps the exponent
    negative — the same branch the sigmoid kernel takes, and for the same
    range reason rather than as an approximation.
    """
    result = []
    for upstream, item in zip(grad_values, values):
        item = float(item)
        if item >= 0:
            magnitude = math.exp(-item)
            derivative = 1.0 / (1.0 + magnitude)
        else:
            magnitude = math.exp(item)
            derivative = magnitude / (1.0 + magnitude)
        result.append(upstream * derivative)
    if len(result) != math.prod(output_shape):
        raise RuntimeError("Softplus VJP kernel returned an unexpected result size")
    return PythonStorage.from_values(result, dtype)
