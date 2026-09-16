"""Reference the arccosine VJP for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensors.tensor import Tensor
import math


def _arccos_gradient(upstream, value):
    if value == -1.0 or value == 1.0:
        raise ValueError("arccos derivative is undefined at -1 and 1")
    return -upstream / math.sqrt(1.0 - float(value) ** 2.0)


def arccos_gradient(grad: Tensor, value: Tensor) -> Storage:
    """Scale the upstream gradient by ``-1 / sqrt(1 - x**2)``."""
    evaluate = _arccos_gradient
    return PythonStorage.from_values(
        [evaluate(upstream, item) for upstream, item in zip(grad._data, value._data)],
        grad.dtype,
    )
