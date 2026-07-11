"""Maximum-value public API."""

import builtins
from typing import Any

from ..tensor import Tensor


class Max:
    """Maximum-value operation."""

    @staticmethod
    def forward(value: Tensor) -> Tensor:
        if value.size == 0:
            raise ValueError("Cannot compute max of empty tensor")
        return Tensor([builtins.max(value._data)], dtype=value.dtype)


def max(value: Any) -> Tensor:
    """Return the maximum as a scalar Tensor."""
    from ..variable import Variable

    if isinstance(value, Variable):
        raise NotImplementedError("Differentiable max is not implemented")
    if not isinstance(value, Tensor):
        value = Tensor(value)
    return Max.forward(value)


__all__ = ["Max", "max"]
