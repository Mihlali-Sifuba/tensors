"""Differentiable stack operation."""

from array import array
from collections.abc import Sequence
from typing import Any, List

from ..dtype import result_dtype
from ..tensor import Tensor


class Stack:
    """Stack tensors along a new axis and split gradients back to inputs."""

    @staticmethod
    def forward(*tensors: Tensor | list[Any], axis: int = 0) -> Tensor:
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise TypeError("stack axis must be an integer")
        if len(tensors) == 1 and isinstance(tensors[0], list):
            tensors = tuple(tensors[0])
        if not tensors:
            raise ValueError("stack requires at least one tensor")

        converted = []
        for t in tensors:
            if isinstance(t, Tensor):
                converted.append(t)
            else:
                converted.append(Tensor(t))

        elem_shape = converted[0].shape
        n = len(converted)
        for t in converted[1:]:
            if t.shape != elem_shape:
                raise ValueError(
                    f"All tensors must have the same shape; got {elem_shape} and {t.shape}"
                )

        if axis < 0:
            axis += len(elem_shape) + 1
        if not (0 <= axis <= len(elem_shape)):
            raise ValueError(
                f"Axis {axis} out of bounds for {len(elem_shape)}D tensor stack"
            )

        out_shape = list(elem_shape)
        out_shape.insert(axis, n)

        before = 1
        for d in elem_shape[:axis]:
            before *= d

        axis_stride = 1
        for d in elem_shape[axis:]:
            axis_stride *= d

        dtype = converted[0].dtype
        for tensor in converted[1:]:
            dtype = result_dtype(dtype, tensor)
        result = array(dtype.typecode, [])
        for g in range(before):
            base = g * axis_stride
            for k in range(n):
                for t in range(axis_stride):
                    result.append(converted[k]._data[base + t])

        return Tensor(result, dtype=dtype, shape=tuple(out_shape))

    @staticmethod
    def backward(grad: Tensor, *inputs: Tensor, **kwargs: object) -> List[Tensor]:
        """Select each slice along the inserted axis."""
        axis = kwargs.get("axis", 0)
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise TypeError("stack axis must be an integer")
        if axis < 0:
            axis += grad.ndim
        gradients = []
        for index in range(len(inputs)):
            key = [slice(None)] * grad.ndim
            key[axis] = index
            gradients.append(grad[tuple(key)])
        return gradients

    @staticmethod
    def backward_graph(grad, *inputs, **kwargs: object):
        """Build differentiable selections for a stack VJP."""
        axis = kwargs.get("axis", 0)
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise TypeError("stack axis must be an integer")
        if axis < 0:
            axis += grad.ndim
        gradients = []
        for index in range(len(inputs)):
            key = [slice(None)] * grad.ndim
            key[axis] = index
            gradients.append(grad[tuple(key)])
        return gradients


def stack(tensors: Sequence[Any], axis: int = 0) -> Any:
    """Stack a sequence of Tensors or Variables along a new axis."""
    from ..variable import Variable

    if any(isinstance(value, Variable) for value in tensors):
        variables = [
            value if isinstance(value, Variable) else Variable(value, requires_grad=False)
            for value in tensors
        ]
        return Variable._from_operation(
            Stack.forward(*(variable.data for variable in variables), axis=axis),
            "stack",
            Stack,
            variables,
            axis=axis,
        )
    return Stack.forward(*tensors, axis=axis)


__all__ = ["Stack", "stack"]
