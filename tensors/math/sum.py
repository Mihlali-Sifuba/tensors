"""Sum and its differentiation rule."""

import builtins
from array import array
from typing import Any, List, Optional

from ..tensor import Tensor


def _reduce_axis(a: Tensor, axis: int) -> tuple:
    """Compute stride and axis_size for summing along an axis."""
    if axis < 0:
        axis += a.ndim
    if not (0 <= axis < a.ndim):
        raise ValueError(f"Axis {axis} out of bounds for {a.ndim}D tensor")
    stride = 1
    for d in a.shape[axis + 1:]:
        stride *= d
    return axis, stride, a.shape[axis]


def _sum_impl(a: Tensor, axis: Optional[int] = None,
              keepdims: bool = False) -> Tensor:
    """Sum along an axis or all elements."""
    if axis is None:
        return Tensor([builtins.sum(a._data)], dtype=a.dtype)

    axis, stride, axis_size = _reduce_axis(a, axis)
    block = axis_size * stride
    group_count = a.size // axis_size

    new_shape = list(a.shape)
    if keepdims:
        new_shape[axis] = 1
    else:
        del new_shape[axis]

    result_data = array(a.dtype.typecode, [])
    for g in range(group_count):
        base = (g // stride) * block + (g % stride)
        total = 0
        for k in range(axis_size):
            total += a._data[base + k * stride]
        result_data.append(total)

    return Tensor(result_data, dtype=a.dtype, shape=tuple(new_shape))


class Sum:
    """Sum with a reverse-mode gradient rule."""

    @staticmethod
    def forward(a: Tensor, axis: Optional[int] = None,
                keepdims: bool = False) -> Tensor:
        return _sum_impl(a, axis=axis, keepdims=keepdims)

    @staticmethod
    def backward(grad: Tensor, *inputs: Tensor, **kwargs: object) -> List[Tensor]:
        a = inputs[0]
        axis = kwargs.get("axis", None)

        if axis is None:
            grad_value = float(next(iter(grad._data)))
            return [Tensor([grad_value] * a.size, dtype=grad.dtype, shape=a.shape)]

        axis, stride, axis_size = _reduce_axis(a, axis)
        block = axis_size * stride

        result = [0.0] * a.size
        for g, val in enumerate(grad._data):
            base = (g // stride) * block + (g % stride)
            for k in range(axis_size):
                result[base + k * stride] = val

        return [Tensor(result, dtype=grad.dtype, shape=a.shape)]

    @staticmethod
    def backward_graph(grad, *inputs, **kwargs: object):
        """Build a differentiable VJP for whole-tensor sums."""
        from ..variable import Variable
        axis = kwargs.get("axis", None)
        if axis is not None:
            raise NotImplementedError("Higher-order derivatives for axis sums are not implemented")
        value = inputs[0]
        ones = Variable(
            Tensor([1.0] * value.size, dtype=grad.dtype, shape=value.shape),
            requires_grad=False,
        )
        return [grad * ones]


def sum(value: Any, axis: Optional[int] = None,
        keepdims: bool = False) -> Any:
    """Return the sum as a Tensor or differentiable Variable."""
    from ..variable import Variable

    if isinstance(value, Variable):
        return Variable._from_operation(
            Sum.forward(value.data, axis=axis, keepdims=keepdims),
            "sum",
            Sum,
            [value],
            axis=axis,
            keepdims=keepdims,
        )
    if not isinstance(value, Tensor):
        value = Tensor(value)
    return Sum.forward(value, axis=axis, keepdims=keepdims)


__all__ = ["Sum", "sum"]
