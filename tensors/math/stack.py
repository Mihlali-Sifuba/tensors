"""Stack operation."""

from array import array
from typing import List, Union

from ..tensor import Tensor


class Stack:
    """Stack operation."""

    @staticmethod
    def forward(tensors: List[Union[Tensor, List]], axis: int = 0) -> Tensor:
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
        result = array(dtype.typecode, [])
        for g in range(before):
            base = g * axis_stride
            for k in range(n):
                for t in range(axis_stride):
                    result.append(converted[k]._data[base + t])

        return Tensor(result, dtype=dtype, shape=tuple(out_shape))


def stack(tensors: List[Union[Tensor, List]], axis: int = 0) -> Tensor:
    """Stack a sequence of tensors along a new axis."""
    return Stack.forward(tensors, axis=axis)


__all__ = ["Stack", "stack"]
