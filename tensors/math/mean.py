"""Mean and its differentiation rule."""

import builtins
from typing import Any, List

from ..dtype import float64
from ..tensor import Tensor


def _mean_impl(a: Tensor) -> float:
    if a.size == 0:
        return 0.0
    return builtins.sum(a._data) / a.size


class Mean:
    """Mean of all tensor elements with a reverse-mode gradient rule."""

    forward = staticmethod(_mean_impl)

    @staticmethod
    def backward(grad: Tensor, *inputs: Tensor, **kwargs: object) -> List[Tensor]:
        a = inputs[0]
        if a.size == 0:
            return [Tensor([], dtype=grad.dtype, shape=a.shape)]
        scale = 1.0 / a.size
        grad_value = float(next(iter(grad._data)))
        return [Tensor([grad_value * scale] * a.size, dtype=grad.dtype, shape=a.shape)]


def mean(value: Any) -> Any:
    """Return the mean as a Tensor or differentiable Variable scalar."""
    from ..autograd.variable import Variable, mean as variable_mean

    if isinstance(value, Variable):
        return variable_mean(value)
    if not isinstance(value, Tensor):
        value = Tensor(value)
    dtype = value.dtype if value.dtype.typecode in {"f", "d"} else float64
    return Tensor([Mean.forward(value)], dtype=dtype)


__all__ = ["Mean", "mean"]
