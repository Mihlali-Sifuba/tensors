"""Axis-aware product reduction and its differentiation rule."""

from __future__ import annotations

from typing import Any

from ..tensor import Tensor
from ._reduction import Axis, immutable_axis, reduction_groups


def _product(values: list[int | float]) -> int | float:
    result: int | float = 1
    for value in values:
        result *= value
    return result


class Prod:
    """Product reduction with zero-safe reverse-mode differentiation."""

    @staticmethod
    def forward(
        value: Tensor,
        axis: Axis = None,
        keepdims: bool = False,
    ) -> Tensor:
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

    @staticmethod
    def backward(
        grad: Tensor,
        *inputs: Tensor,
        **kwargs: object,
    ) -> list[Tensor]:
        value = inputs[0]
        axis = kwargs.get("axis")
        keepdims = kwargs.get("keepdims", False)
        _, output_shape, groups = reduction_groups(
            value,
            axis,
            keepdims,
            scalar_as_vector=True,
        )
        if grad.shape != output_shape:
            raise ValueError(
                f"Gradient shape {grad.shape} does not match output shape "
                f"{output_shape}"
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

    @staticmethod
    def backward_graph(grad, *inputs, **kwargs: object):
        """Build the product VJP explicitly from products excluding each input."""
        from ..ops._utils import zero_like_graph
        from .concat import concat
        from .reshape import reshape

        value = inputs[0]
        axis = kwargs.get("axis")
        keepdims = kwargs.get("keepdims", False)
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


def prod(
    value: Any,
    axis: Axis = None,
    keepdims: bool = False,
) -> Any:
    """Multiply values over one, several, or all axes."""
    from ..variable import Variable

    axis = immutable_axis(axis)
    if isinstance(value, Variable):
        return Variable._from_operation(
            Prod.forward(value.data, axis=axis, keepdims=keepdims),
            "prod",
            Prod,
            [value],
            axis=axis,
            keepdims=keepdims,
        )
    if not isinstance(value, Tensor):
        value = Tensor(value)
    return Prod.forward(value, axis=axis, keepdims=keepdims)


__all__ = ["Prod", "prod"]
