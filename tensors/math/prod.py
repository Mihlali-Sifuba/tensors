"""Axis-aware product reduction and its differentiation rule."""

from __future__ import annotations

from typing import Any, overload

from .._typing import TensorData, TensorLike, TensorResult, TensorValue
from ..backend import execute_reduction, execute_reduction_gradient
from ..graph.operation import Operation
from ..tensor import Tensor
from ._reduction import (
    Axis,
    immutable_axis,
    normalize_axes,
    reduction_groups,
    reduction_shape,
)


def _product(values: list[int | float]) -> int | float:
    result: int | float = 1
    for value in values:
        result *= value
    return result


class Prod(Operation):
    """Product reduction with zero-safe reverse-mode differentiation."""

    __slots__ = ("axis", "keepdims")
    name = "prod"

    def __init__(
        self,
        *,
        axis: Axis = None,
        keepdims: bool = False,
    ) -> None:
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "keepdims", keepdims)

    def forward(self, value: Tensor) -> Tensor:
        axis = self.axis
        keepdims = self.keepdims
        axes = normalize_axes(value.ndim, axis)
        output_shape = reduction_shape(value.shape, axes, keepdims)
        if axis is None and not keepdims:
            output_shape = (1,)
        accelerated = execute_reduction(
            "prod",
            value,
            axes,
            keepdims=keepdims,
            dtype=value.dtype,
            output_shape=output_shape,
        )
        if accelerated is not None:
            return Tensor._from_owned_storage(
                accelerated,
                dtype=value.dtype,
                shape=output_shape,
            )
        _, output_shape, groups = reduction_groups(
            value,
            axis,
            keepdims,
            scalar_as_vector=True,
        )
        values = [
            _product([value._data[index] for index in group])
            for group in groups
        ]
        return Tensor(values, dtype=value.dtype, shape=output_shape)

    def backward(
        self,
        grad: Tensor,
        *inputs: Tensor,
        needs_input_grad: tuple[bool, ...],
    ) -> list[Tensor]:
        value = inputs[0]
        axis = self.axis
        keepdims = self.keepdims
        axes = normalize_axes(value.ndim, axis)
        output_shape = reduction_shape(value.shape, axes, keepdims)
        if axis is None and not keepdims:
            output_shape = (1,)
        if grad.shape != output_shape:
            raise ValueError(
                f"Gradient shape {grad.shape} does not match output shape "
                f"{output_shape}"
            )
        accelerated = execute_reduction_gradient(
            "prod",
            grad,
            value,
            axes,
            keepdims=keepdims,
        )
        if accelerated is not None:
            return [Tensor._from_owned_storage(accelerated, dtype=grad.dtype, shape=value.shape)]
        _, _, groups = reduction_groups(
            value,
            axis,
            keepdims,
            scalar_as_vector=True,
        )
        gradients = [0.0] * value.size
        for output_index, group in enumerate(groups):
            for input_index in group:
                other_values = [
                    value._data[index]
                    for index in group
                    if index != input_index
                ]
                gradients[input_index] = (
                    grad._data[output_index] * _product(other_values)
                )
        return [Tensor(gradients, dtype=grad.dtype, shape=value.shape)]

    def backward_graph(
        self,
        grad,
        *inputs,
        needs_input_grad: tuple[bool, ...],
    ):
        """Build the product VJP explicitly from products excluding each input."""
        from ..ops._utils import zero_like_graph
        from .concat import concat
        from .reshape import reshape

        value = inputs[0]
        axis = self.axis
        keepdims = self.keepdims
        _, _, groups = reduction_groups(
            value.data,
            axis,
            keepdims,
            scalar_as_vector=True,
        )
        if value.size == 0:
            return [zero_like_graph(value)]

        flat_value = reshape(value, (value.size,))
        flat_grad = reshape(grad, (grad.size,))
        group_for_input = {}
        for output_index, group in enumerate(groups):
            for input_index in group:
                group_for_input[input_index] = (output_index, group)

        terms = []
        for input_index in range(value.size):
            output_index, group = group_for_input[input_index]
            derivative = zero_like_graph(flat_value[input_index]) + 1.0
            for other_index in group:
                if other_index != input_index:
                    derivative = derivative * flat_value[other_index]
            term = flat_grad[output_index] * derivative
            terms.append(reshape(term, (1,)))
        return [reshape(concat(terms), value.shape)]


@overload
def prod(
    value: TensorValue,
    axis: Axis = None,
    keepdims: bool = False,
) -> TensorValue: ...


@overload
def prod(
    value: TensorData,
    axis: Axis = None,
    keepdims: bool = False,
) -> Tensor: ...


def prod(
    value: TensorLike,
    axis: Axis = None,
    keepdims: bool = False,
) -> TensorResult:
    """Multiply values over one, several, or all axes."""
    from ..variable import Variable

    axis = immutable_axis(axis)
    if isinstance(value, Variable):
        operation = Prod(axis=axis, keepdims=keepdims)
        return Variable._from_operation(
            operation.forward(value.data),
            operation,
            (value,),
        )
    if not isinstance(value, Tensor):
        value = Tensor(value)
    return Prod(axis=axis, keepdims=keepdims).forward(value)


__all__ = ["Prod", "prod"]
