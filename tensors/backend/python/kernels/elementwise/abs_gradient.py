"""Reference the absolute value VJP for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensors.tensor import Tensor
import math


def _abs_gradient(upstream, value):
    if isinstance(value, float) and math.isnan(value):
        return math.nan
    if value > 0:
        return upstream
    if value < 0:
        return -upstream
    return 0.0


def abs_gradient(grad: Tensor, value: Tensor) -> Storage:
    """Route the upstream gradient by the sign of each input."""
    evaluate = _abs_gradient
    return PythonStorage.from_values(
        [evaluate(upstream, item) for upstream, item in zip(grad._data, value._data)],
        grad.dtype,
    )
