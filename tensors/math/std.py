"""Standard-deviation public API."""

import builtins
import math as _math
from typing import Any

from ..tensor import Tensor


def std(value: Any) -> Tensor:
    """Return population standard deviation as a scalar Tensor."""
    from ..variable import Variable

    if isinstance(value, Variable):
        raise NotImplementedError("Differentiable std is not implemented")
    if not isinstance(value, Tensor):
        value = Tensor(value)
    if value.size == 0:
        return Tensor([0.0])
    average = builtins.sum(value._data) / value.size
    variance = builtins.sum((item - average) ** 2 for item in value._data) / value.size
    return Tensor([_math.sqrt(variance)])


__all__ = ["std"]
