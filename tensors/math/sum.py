"""Sum and its differentiation rule."""

import builtins
from typing import Any, List

from ..tensor import Tensor


def _sum_impl(a: Tensor) -> float:
    return builtins.sum(a._data)


class Sum:
    """Sum all tensor elements with a reverse-mode gradient rule."""

    forward = staticmethod(_sum_impl)

    @staticmethod
    def backward(grad: Tensor, *inputs: Tensor, **kwargs: object) -> List[Tensor]:
        a = inputs[0]
        grad_value = float(next(iter(grad._data)))
        return [Tensor([grad_value] * a.size, dtype=grad.dtype, shape=a.shape)]


def sum(value: Any) -> Any:
    """Return the sum as a Tensor or differentiable Variable scalar."""
    from ..autograd.variable import Variable, sum as variable_sum

    if isinstance(value, Variable):
        return variable_sum(value)
    if not isinstance(value, Tensor):
        value = Tensor(value)
    return Tensor([Sum.forward(value)], dtype=value.dtype)


__all__ = ["Sum", "sum"]
