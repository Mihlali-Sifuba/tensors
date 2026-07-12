"""Mean and its differentiation rule."""

import builtins
from array import array
from typing import Any, List, Optional

from ..dtype import float64
from ..tensor import Tensor
from .sum import _reduce_axis, _sum_impl


class Mean:
    """Mean with a reverse-mode gradient rule."""

    @staticmethod
    def forward(a: Tensor, axis: Optional[int] = None,
                keepdims: bool = False) -> Tensor:
        if a.size == 0:
            return Tensor([0.0], dtype=a.dtype if a.dtype.typecode in {"f", "d"} else float64)
        result = _sum_impl(a, axis=axis, keepdims=keepdims)
        if axis is None:
            scale = 1.0 / a.size
            return Tensor([float(result._data[0]) * scale], dtype=result.dtype)
        result_data = array(
            result.dtype.typecode,
            (float(x) / a.shape[axis] for x in result._data),
        )
        return Tensor(result_data, dtype=result.dtype, shape=result.shape)

    @staticmethod
    def backward(grad: Tensor, *inputs: Tensor, **kwargs: object) -> List[Tensor]:
        a = inputs[0]
        axis = kwargs.get("axis", None)

        if axis is None:
            if a.size == 0:
                return [Tensor([], dtype=grad.dtype, shape=a.shape)]
            scale = 1.0 / a.size
            grad_value = float(next(iter(grad._data)))
            return [Tensor([grad_value * scale] * a.size, dtype=grad.dtype, shape=a.shape)]

        _, _, axis_size = _reduce_axis(a, axis)
        scale = 1.0 / axis_size

        # Reuse sum's backward logic scaled by 1/axis_size
        from .sum import Sum
        sum_grads = Sum.backward(grad, a, axis=axis)
        result_data = array(
            sum_grads[0].dtype.typecode,
            (float(x) * scale for x in sum_grads[0]._data),
        )
        return [Tensor(result_data, dtype=grad.dtype, shape=a.shape)]

    @staticmethod
    def backward_graph(grad, *inputs, **kwargs: object):
        """Build a differentiable VJP for whole-tensor means."""
        from ..variable import Variable
        axis = kwargs.get("axis", None)
        value = inputs[0]
        if axis is not None:
            raise NotImplementedError("Higher-order derivatives for axis means are not implemented")
        if value.size == 0:
            raise NotImplementedError("Higher-order derivatives for empty means are not implemented")
        ones = Variable(
            Tensor([1.0] * value.size, dtype=grad.dtype, shape=value.shape),
            requires_grad=False,
        )
        return [(grad * ones) / value.size]


def mean(value: Any, axis: Optional[int] = None,
         keepdims: bool = False) -> Any:
    """Return the mean as a Tensor or differentiable Variable."""
    from ..variable import Variable

    if isinstance(value, Variable):
        return Variable._from_operation(
            Mean.forward(value.data, axis=axis, keepdims=keepdims),
            "mean",
            Mean,
            [value],
            axis=axis,
            keepdims=keepdims,
        )
    if not isinstance(value, Tensor):
        value = Tensor(value)
    return Mean.forward(value, axis=axis, keepdims=keepdims)


__all__ = ["Mean", "mean"]
