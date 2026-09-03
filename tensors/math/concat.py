"""Tensor concatenation and its differentiation rule."""

from __future__ import annotations

from array import array
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, List, overload

from .._typing import TensorData, TensorLike, TensorResult
from ..backend import execute_concat
from ..dtype import result_dtype
from ..graph.operation import Operation
from ..tensor import Tensor

if TYPE_CHECKING:
    from ..variable import Variable


class Concat(Operation):
    """Concatenate tensors along an existing axis."""

    __slots__ = ("axis", "keepdims")
    name = "concat"

    def __init__(
        self,
        *,
        axis: int = 0,
        keepdims: bool = False,
    ) -> None:
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "keepdims", keepdims)

    def forward(self, *tensors: Tensor | list[Any]) -> Tensor:
        """Concatenate one or more tensors along ``axis``."""
        axis = self.axis
        keepdims = self.keepdims
        if not isinstance(keepdims, bool):
            raise TypeError("keepdims must be a bool")
        if keepdims:
            raise ValueError("concat does not support keepdims")
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise TypeError("concat axis must be an integer")
        if len(tensors) == 1 and isinstance(tensors[0], list):
            tensors = tuple(tensors[0])
        if not tensors:
            raise ValueError("concat requires at least one tensor")

        converted = [value if isinstance(value, Tensor) else Tensor(value) for value in tensors]
        reference = converted[0]
        if reference.ndim == 0:
            if axis < 0:
                axis += 1
            if axis != 0:
                raise ValueError("Axis out of bounds for scalar tensor concat")
            if any(tensor.ndim != 0 for tensor in converted[1:]):
                raise ValueError("All tensors must have the same rank")

            dtype = reference.dtype
            for tensor in converted[1:]:
                dtype = result_dtype(dtype, tensor)
            values = array(
                dtype.typecode,
                [tensor._data[0] for tensor in converted],
            )
            return Tensor(values, dtype=dtype, shape=(len(converted),))

        if axis < 0:
            axis += reference.ndim
        if not 0 <= axis < reference.ndim:
            raise ValueError(f"Axis {axis} out of bounds for {reference.ndim}D tensor")

        for tensor in converted[1:]:
            if tensor.ndim != reference.ndim:
                raise ValueError(
                    f"All tensors must have the same rank; "
                    f"got {reference.ndim} and {tensor.ndim}"
                )
            for dimension in range(reference.ndim):
                if dimension != axis and tensor.shape[dimension] != reference.shape[dimension]:
                    raise ValueError(
                        "Tensors must match on all non-concat axes; "
                        f"axis {dimension}: {reference.shape[dimension]} vs "
                        f"{tensor.shape[dimension]}"
                    )

        output_shape = list(reference.shape)
        output_shape[axis] = sum(tensor.shape[axis] for tensor in converted)

        trailing_size = 1
        for dimension in reference.shape[axis + 1:]:
            trailing_size *= dimension
        groups = 1
        for dimension in reference.shape[:axis]:
            groups *= dimension

        dtype = reference.dtype
        for tensor in converted[1:]:
            dtype = result_dtype(dtype, tensor)

        accelerated = execute_concat(
            converted,
            axis,
            dtype=dtype,
            output_shape=tuple(output_shape),
        )
        if accelerated is not None:
            return Tensor._from_owned_storage(
                accelerated,
                dtype=dtype,
                shape=tuple(output_shape),
            )

        promoted = [
            tensor if tensor.dtype == dtype else tensor.astype(dtype)
            for tensor in converted
        ]
        values = array(dtype.typecode, [])
        for group in range(groups):
            for tensor in promoted:
                count = tensor.shape[axis] * trailing_size
                start = group * count
                values.extend(tensor._data[start:start + count])

        return Tensor(values, dtype=dtype, shape=tuple(output_shape))

    def backward(self, grad: Tensor, *inputs: Tensor) -> List[Tensor]:
        """Split an upstream gradient back across every concatenated input."""
        axis = self.axis
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise TypeError("concat axis must be an integer")
        if axis < 0:
            axis += grad.ndim

        if inputs[0].ndim == 0:
            return [
                Tensor([grad._data[index]], dtype=grad.dtype, shape=())
                for index in range(len(inputs))
            ]

        offset = 0
        gradients = []
        for tensor in inputs:
            key = [slice(None)] * grad.ndim
            key[axis] = slice(offset, offset + tensor.shape[axis])
            gradients.append(grad[tuple(key)])
            offset += tensor.shape[axis]
        return gradients

    def backward_graph(self, grad, *inputs):
        """Differentiably split an upstream gradient along the concat axis."""
        axis = self.axis
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise TypeError("concat axis must be an integer")
        if axis < 0:
            axis += grad.ndim
        if inputs[0].ndim == 0:
            return [grad[index] for index in range(len(inputs))]
        offset = 0
        gradients = []
        for tensor in inputs:
            key = [slice(None)] * grad.ndim
            key[axis] = slice(offset, offset + tensor.shape[axis])
            gradients.append(grad[tuple(key)])
            offset += tensor.shape[axis]
        return gradients


@overload
def concat(tensors: Sequence[Variable], axis: int = 0) -> Variable: ...


@overload
def concat(tensors: Sequence[TensorData], axis: int = 0) -> Tensor: ...


@overload
def concat(tensors: Sequence[TensorLike], axis: int = 0) -> TensorResult: ...


def concat(tensors: Sequence[TensorLike], axis: int = 0) -> TensorResult:
    """Concatenate Tensors or Variables along an existing axis."""
    from ..variable import Variable

    if any(isinstance(value, Variable) for value in tensors):
        variables = [
            value if isinstance(value, Variable) else Variable(value, requires_grad=False)
            for value in tensors
        ]
        operation = Concat(axis=axis)
        return Variable._from_operation(
            operation.forward(*(variable.data for variable in variables)),
            operation,
            variables,
        )
    return Concat(axis=axis).forward(*tensors)


__all__ = ["Concat", "concat"]
