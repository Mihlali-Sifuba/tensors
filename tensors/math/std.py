"""Standard-deviation public API."""

import builtins
import math as _math
from typing import Any

from ..tensor import Tensor


class Std:
    """Population standard-deviation operation."""

    @staticmethod
    def forward(value: Tensor) -> Tensor:
        if value.size == 0:
            return Tensor([0.0])
        average = builtins.sum(value._data) / value.size
        variance = builtins.sum((item - average) ** 2 for item in value._data) / value.size
        return Tensor([_math.sqrt(variance)])


def std(value: Any) -> Tensor:
    """Return population standard deviation as a scalar Tensor."""
    from ..variable import Variable

    if isinstance(value, Variable):
        raise NotImplementedError("Differentiable std is not implemented")
    if not isinstance(value, Tensor):
        value = Tensor(value)
    return Std.forward(value)


__all__ = ["Std", "std"]
