"""Maximum-value public API."""

import builtins
from typing import Any

from ..tensor import Tensor


def max(value: Any) -> Tensor:
    """Return the maximum as a scalar Tensor."""
    from ..variable import Variable

    if isinstance(value, Variable):
        raise NotImplementedError("Differentiable max is not implemented")
    if not isinstance(value, Tensor):
        value = Tensor(value)
    if value.size == 0:
        raise ValueError("Cannot compute max of empty tensor")
    return Tensor([builtins.max(value._data)], dtype=value.dtype)


__all__ = ["max"]
