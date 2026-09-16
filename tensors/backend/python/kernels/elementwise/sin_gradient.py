"""Reference the sine VJP for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensors.tensor import Tensor
import math as _math


def sin_gradient(grad: Tensor, value: Tensor) -> Storage:
    """Scale the upstream gradient by ``cos(x)``."""
    evaluate = lambda upstream, value: upstream * _math.cos(float(value))
    return PythonStorage.from_values(
        [evaluate(upstream, item) for upstream, item in zip(grad._data, value._data)],
        grad.dtype,
    )
