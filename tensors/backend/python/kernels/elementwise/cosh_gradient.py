"""Reference the hyperbolic cosine VJP for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensors.tensor import Tensor
import math


def _sinh(value):
    try:
        return math.sinh(float(value))
    except OverflowError:
        return math.copysign(math.inf, value)


def cosh_gradient(grad: Tensor, value: Tensor) -> Storage:
    """Scale the upstream gradient by ``sinh(x)``."""
    evaluate = lambda upstream, item: upstream * _sinh(item)
    return PythonStorage.from_values(
        [evaluate(upstream, item) for upstream, item in zip(grad._data, value._data)],
        grad.dtype,
    )
