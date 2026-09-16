"""Reference the rectified linear unit VJP for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensors.tensor import Tensor
import math


def _gradient(upstream, value):
    if isinstance(value, float) and math.isnan(value):
        return math.nan
    return upstream if value > 0 else 0


def relu_gradient(grad: Tensor, value: Tensor) -> Storage:
    """Pass the upstream gradient only where the input was positive."""
    evaluate = _gradient
    return PythonStorage.from_values(
        [evaluate(upstream, item) for upstream, item in zip(grad._data, value._data)],
        grad.dtype,
    )
