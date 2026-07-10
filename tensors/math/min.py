"""Minimum-value public API."""

import builtins
from typing import Any

from ..tensor import Tensor


def min(value: Any) -> Tensor:
    """Return the minimum as a scalar Tensor.

    A differentiable minimum rule is intentionally not provided yet.
    """
    from ..variable import Variable

    if isinstance(value, Variable):
        raise NotImplementedError("Differentiable min is not implemented")
    if not isinstance(value, Tensor):
        value = Tensor(value)
    if value.size == 0:
        raise ValueError("Cannot compute min of empty tensor")
    return Tensor([builtins.min(value._data)], dtype=value.dtype)


__all__ = ["min"]
