"""Mean and its differentiation rule."""

import math
from typing import Any, List

from ..dtype import float64
from ..tensor import Tensor
from ._reduction import (
    Axis, immutable_axis, keepdims_shape, normalize_axes, reduction_groups,
    reduction_size,
)
from .sum import Sum, _stable_float_sum, _sum_exact_ratios


def _stable_float_mean(values: list[float]) -> float:
    """Return a mean without overflowing its sum or underflowing its terms."""
    if not values:
        return 0.0
    if any(not math.isfinite(value) for value in values):
        return _stable_float_sum(values) / len(values)
    return _sum_exact_ratios(
        [value.as_integer_ratio() for value in values],
        divisor=len(values),
    )


class Mean:
    """Mean with a reverse-mode gradient rule."""

    @staticmethod
    def forward(a: Tensor, axis: Axis = None,
                keepdims: bool = False) -> Tensor:
        _, output_shape, groups = reduction_groups(
            a,
            axis,
            keepdims,
            scalar_as_vector=True,
        )
        dtype = a.dtype if a.dtype.typecode in {"f", "d"} else float64
        values = [
            _stable_float_mean([
                float(a._data[index]) for index in group
            ])
            if group else 0.0
            for group in groups
        ]
        return Tensor(values, dtype=dtype, shape=output_shape)

    @staticmethod
    def backward(grad: Tensor, *inputs: Tensor, **kwargs: object) -> List[Tensor]:
        a = inputs[0]
        axis = kwargs.get("axis")
        keepdims = bool(kwargs.get("keepdims", False))
        count = reduction_size(a.shape, normalize_axes(a.ndim, axis))
        if count == 0:
            return [Tensor([], dtype=grad.dtype, shape=a.shape)]
        summed = Sum.backward(grad, a, axis=axis, keepdims=keepdims)[0]
        return [Tensor(
            [float(item) / count for item in summed._data],
            dtype=grad.dtype,
            shape=a.shape,
        )]

    @staticmethod
    def backward_graph(grad, *inputs, **kwargs: object):
        """Build a differentiable VJP for an axis-aware mean."""
        from ..variable import Variable
        from .reshape import reshape

        axis = kwargs.get("axis")
        keepdims = bool(kwargs.get("keepdims", False))
        value = inputs[0]
        count = reduction_size(value.shape, normalize_axes(value.ndim, axis))
        if count == 0:
            raise NotImplementedError("Higher-order derivatives for empty means are not implemented")
        expanded = grad if keepdims else reshape(grad, keepdims_shape(value.shape, axis))
        ones = Variable(
            Tensor([1.0] * value.size, dtype=grad.dtype, shape=value.shape),
            requires_grad=False,
        )
        return [(expanded * ones) / count]


def mean(value: Any, axis: Axis = None,
         keepdims: bool = False) -> Any:
    """Compute the mean over one, several, or all axes."""
    from ..variable import Variable

    axis = immutable_axis(axis)

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
