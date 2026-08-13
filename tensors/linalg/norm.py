"""Differentiable Euclidean norm."""

import math
from typing import Any, List

from ..dtype import float64
from ..tensor import Tensor


class Norm:
    """Whole-tensor Euclidean norm with reverse-mode gradient rules."""

    @staticmethod
    def forward(value: Tensor) -> Tensor:
        """Return ``sqrt(sum(value[i] ** 2))`` as a scalar Tensor."""
        dtype = value.dtype if value.dtype.typecode in {"f", "d"} else float64
        result = math.sqrt(sum(float(item) ** 2 for item in value._data))
        return Tensor([result], dtype=dtype, shape=())

    @staticmethod
    def backward(grad: Tensor, *inputs: Tensor, **kwargs: object) -> List[Tensor]:
        """Differentiate the Euclidean norm with respect to its input."""
        value = inputs[0]
        magnitude = Norm.forward(value).item()
        upstream = grad.item()
        if magnitude == 0:
            values = [0.0] * value.size
        else:
            values = [upstream * item / magnitude for item in value._data]
        return [Tensor(values, dtype=grad.dtype, shape=value.shape)]

    @staticmethod
    def backward_graph(grad, *inputs, **kwargs: object):
        """Build a differentiable VJP for a nonzero Euclidean norm."""
        value = inputs[0]
        if Norm.forward(value.data).item() == 0:
            raise ValueError(
                "Higher-order derivatives of norm are undefined at zero"
            )
        return [grad * (value / norm(value))]


def norm(value: Any) -> Any:
    """Return the whole-tensor Euclidean norm as a Tensor or Variable."""
    from ..variable import Variable

    if isinstance(value, Variable):
        return Variable._from_operation(
            Norm.forward(value.data),
            "norm",
            Norm,
            [value],
        )
    if not isinstance(value, Tensor):
        value = Tensor(value)
    return Norm.forward(value)


__all__ = ["Norm", "norm"]
