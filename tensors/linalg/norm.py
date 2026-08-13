"""Differentiable Euclidean norm."""

import math
from typing import Any, List

from ..dtype import float64
from ..tensor import Tensor
from ..math._reduction import Axis, immutable_axis, keepdims_shape, reduction_groups


class Norm:
    """Whole-tensor Euclidean norm with reverse-mode gradient rules."""

    @staticmethod
    def forward(
        value: Tensor,
        axis: Axis = None,
        keepdims: bool = False,
    ) -> Tensor:
        """Return Euclidean norms over one, several, or all axes."""
        dtype = value.dtype if value.dtype.typecode in {"f", "d"} else float64
        _, output_shape, groups = reduction_groups(value, axis, keepdims)
        results = [
            math.sqrt(sum(float(value._data[index]) ** 2 for index in group))
            for group in groups
        ]
        return Tensor(results, dtype=dtype, shape=output_shape)

    @staticmethod
    def backward(grad: Tensor, *inputs: Tensor, **kwargs: object) -> List[Tensor]:
        """Differentiate the Euclidean norm with respect to its input."""
        value = inputs[0]
        axis = kwargs.get("axis")
        keepdims = bool(kwargs.get("keepdims", False))
        _, output_shape, groups = reduction_groups(value, axis, keepdims)
        if grad.shape != output_shape:
            raise ValueError(
                f"Gradient shape {grad.shape} does not match output shape {output_shape}"
            )
        values = [0.0] * value.size
        for output_index, group in enumerate(groups):
            magnitude = math.sqrt(
                sum(float(value._data[index]) ** 2 for index in group)
            )
            if magnitude == 0:
                continue
            scale = grad._data[output_index] / magnitude
            for input_index in group:
                values[input_index] = scale * value._data[input_index]
        return [Tensor(values, dtype=grad.dtype, shape=value.shape)]

    @staticmethod
    def backward_graph(grad, *inputs, **kwargs: object):
        """Build a differentiable VJP for nonzero axis-aware norms."""
        from ..math.reshape import reshape

        value = inputs[0]
        axis = kwargs.get("axis")
        keepdims = bool(kwargs.get("keepdims", False))
        magnitudes = Norm.forward(value.data, axis=axis, keepdims=True)
        if any(item == 0 for item in magnitudes._data):
            raise ValueError(
                "Higher-order derivatives of norm are undefined at zero"
            )
        expanded_grad = grad if keepdims else reshape(
            grad, keepdims_shape(value.shape, axis)
        )
        return [expanded_grad * (value / norm(value, axis=axis, keepdims=True))]


def norm(value: Any, axis: Axis = None, keepdims: bool = False) -> Any:
    """Compute Euclidean norms over one, several, or all axes."""
    from ..variable import Variable

    axis = immutable_axis(axis)

    if isinstance(value, Variable):
        return Variable._from_operation(
            Norm.forward(value.data, axis=axis, keepdims=keepdims),
            "norm",
            Norm,
            [value],
            axis=axis,
            keepdims=keepdims,
        )
    if not isinstance(value, Tensor):
        value = Tensor(value)
    return Norm.forward(value, axis=axis, keepdims=keepdims)


__all__ = ["Norm", "norm"]
