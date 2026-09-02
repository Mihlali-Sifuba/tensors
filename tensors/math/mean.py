"""Mean and its differentiation rule."""

import math
from typing import Any, List, overload

from .._typing import TensorData, TensorLike, TensorResult, TensorValue
from ..backend import execute_reduction, execute_reduction_gradient
from ..dtype import float64
from ..tensor import Tensor
from ._reduction import (
    Axis, immutable_axis, keepdims_shape, normalize_axes, reduction_groups,
    reduction_shape, reduction_size,
)
from .sum import Sum, _stable_float_sum, _sum_exact_ratios


def _stable_float_mean(values: list[float]) -> float:
    """Return a mean without overflowing its sum or underflowing its terms."""
    if not values:
        return math.nan
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
        axes = normalize_axes(a.ndim, axis)
        output_shape = reduction_shape(a.shape, axes, keepdims)
        if axis is None and not keepdims:
            output_shape = (1,)
        dtype = a.dtype if a.dtype.typecode in {"f", "d"} else float64
        accelerated = execute_reduction(
            "mean",
            a,
            axes,
            keepdims=keepdims,
            dtype=dtype,
            output_shape=output_shape,
        )
        if accelerated is not None:
            return Tensor._from_owned_storage(accelerated, dtype=dtype, shape=output_shape)
        data = a._data
        if axes == tuple(range(a.ndim)):
            return Tensor(
                [_stable_float_mean([float(value) for value in data])],
                dtype=dtype,
                shape=output_shape,
            )
        _, output_shape, groups = reduction_groups(
            a, axis, keepdims, scalar_as_vector=True
        )
        values = [
            _stable_float_mean([
                float(data[index]) for index in group
            ])
            for group in groups
        ]
        return Tensor(values, dtype=dtype, shape=output_shape)

    @staticmethod
    def backward(grad: Tensor, *inputs: Tensor, **kwargs: object) -> List[Tensor]:
        a = inputs[0]
        axis = kwargs.get("axis")
        keepdims = kwargs.get("keepdims", False)
        axes = normalize_axes(a.ndim, axis)
        output_shape = reduction_shape(a.shape, axes, keepdims)
        if axis is None and not keepdims:
            output_shape = (1,)
        if grad.shape != output_shape:
            raise ValueError(
                f"Gradient shape {grad.shape} does not match output shape "
                f"{output_shape}"
            )
        count = reduction_size(a.shape, axes)
        if count == 0:
            return [Tensor([], dtype=grad.dtype, shape=a.shape)]
        accelerated = execute_reduction_gradient(
            "mean",
            grad,
            a,
            axes,
            keepdims=keepdims,
        )
        if accelerated is not None:
            return [Tensor._from_owned_storage(accelerated, dtype=grad.dtype, shape=a.shape)]
        summed = Sum.backward(grad, a, axis=axis, keepdims=keepdims)[0]
        return [Tensor(
            [float(item) / count for item in summed._data],
            dtype=grad.dtype,
            shape=a.shape,
        )]

    @staticmethod
    def backward_graph(grad, *inputs, **kwargs: object):
        """Build a differentiable VJP for an axis-aware mean."""
        from ..creation import ones
        from ..ops._utils import zero_like_graph
        from ..variable import Variable
        from .reshape import reshape

        axis = kwargs.get("axis")
        keepdims = kwargs.get("keepdims", False)
        value = inputs[0]
        count = reduction_size(value.shape, normalize_axes(value.ndim, axis))
        if count == 0:
            return [zero_like_graph(value)]
        expanded = grad if keepdims else reshape(grad, keepdims_shape(value.shape, axis))
        unit = Variable(
            ones(value.shape, dtype=grad.dtype),
            requires_grad=False,
        )
        return [(expanded * unit) / count]


@overload
def mean(
    value: TensorValue,
    axis: Axis = None,
    keepdims: bool = False,
) -> TensorValue: ...


@overload
def mean(
    value: TensorData,
    axis: Axis = None,
    keepdims: bool = False,
) -> Tensor: ...


def mean(
    value: TensorLike,
    axis: Axis = None,
    keepdims: bool = False,
) -> TensorResult:
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
