"""Tensor concatenation and its differentiation rule."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, List, Optional, overload

from tensors._typing import TensorData, TensorLike, TensorResult
from tensors.backend import execute_concat
from tensors.dtype import result_dtype
from tensors.graph.expression import as_tensor_operand
from tensors.operations.base import Operation
from tensors.tensor import Tensor

if TYPE_CHECKING:
    from tensors.graph.node import VariableNode
    from tensors.variable import Variable


def _operands(tensors: Sequence[Any]) -> tuple[Any, ...]:
    """Read a lone list argument as the operands it holds."""
    if len(tensors) == 1 and isinstance(tensors[0], list):
        return tuple(tensors[0])
    return tuple(tensors)


class Concat(Operation):
    """Concatenate tensors along an existing axis."""

    __slots__ = ("axis", "keepdims")
    name = "concat"

    def __init__(self, *, axis: int = 0, keepdims: bool = False) -> None:
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "keepdims", keepdims)

    def forward(self, *tensors: Tensor | list[Tensor]) -> Tensor:
        """Concatenate one or more tensors along the configured axis."""
        axis = self.axis
        keepdims = self.keepdims
        if not isinstance(keepdims, bool):
            raise TypeError("keepdims must be a bool")
        if keepdims:
            raise ValueError("concat does not support keepdims")
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise TypeError("concat axis must be an integer")
        converted: tuple[Tensor, ...] = _operands(tensors)
        if not converted:
            raise ValueError("concat requires at least one tensor")
        reference = converted[0]
        if reference.ndim == 0:
            if axis < 0:
                axis += 1
            if axis != 0:
                raise ValueError("Axis out of bounds for scalar tensor concat")
            if any(tensor.ndim != 0 for tensor in converted[1:]):
                raise ValueError("All tensors must have the same rank")
            output_shape = (len(converted),)
        else:
            if axis < 0:
                axis += reference.ndim
            if not 0 <= axis < reference.ndim:
                raise ValueError(
                    f"Axis {axis} out of bounds for {reference.ndim}D tensor"
                )
            for tensor in converted[1:]:
                if tensor.ndim != reference.ndim:
                    raise ValueError(
                        "All tensors must have the same rank; got "
                        f"{reference.ndim} and {tensor.ndim}"
                    )
                for dimension in range(reference.ndim):
                    if (
                        dimension != axis
                        and tensor.shape[dimension] != reference.shape[dimension]
                    ):
                        raise ValueError(
                            "Tensors must match on all non-concat axes; axis "
                            f"{dimension}: {reference.shape[dimension]} vs "
                            f"{tensor.shape[dimension]}"
                        )
            shape = list(reference.shape)
            shape[axis] = sum(tensor.shape[axis] for tensor in converted)
            output_shape = tuple(shape)
        dtype = reference.dtype
        for tensor in converted[1:]:
            dtype = result_dtype(dtype, tensor)
        promoted = tuple(
            tensor if tensor.dtype == dtype else tensor.astype(dtype)
            for tensor in converted
        )
        accelerated = execute_concat(
            promoted, axis, dtype=dtype, output_shape=output_shape
        )
        return Tensor._from_owned_storage(accelerated, dtype=dtype, shape=output_shape)

    def backward(
        self, grad: Tensor, *inputs: Tensor, needs_input_grad: tuple[bool, ...]
    ) -> List[Optional[Tensor]]:
        """Split an upstream gradient across the requested inputs."""
        axis = self.axis
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise TypeError("concat axis must be an integer")
        if axis < 0:
            axis += grad.ndim
        if inputs[0].ndim == 0:
            from tensors.variable import Variable

            if isinstance(grad, Variable):
                return [
                    grad[index] if wanted else None
                    for index, wanted in enumerate(needs_input_grad)
                ]
            return [
                (Tensor([grad[index]], dtype=grad.dtype, shape=()) if wanted else None)
                for index, wanted in enumerate(needs_input_grad)
            ]
        offset = 0
        gradients: List[Optional[Tensor]] = []
        for tensor, wanted in zip(inputs, needs_input_grad):
            if wanted:
                key = [slice(None)] * grad.ndim
                key[axis] = slice(offset, offset + tensor.shape[axis])
                gradients.append(grad[tuple(key)])
            else:
                gradients.append(None)
            offset += tensor.shape[axis]
        return gradients


@overload
def concat(tensors: Sequence[VariableNode], axis: int = 0) -> VariableNode: ...


@overload
def concat(tensors: Sequence[Variable], axis: int = 0) -> Variable: ...


@overload
def concat(tensors: Sequence[TensorData], axis: int = 0) -> Tensor: ...


@overload
def concat(tensors: Sequence[TensorLike], axis: int = 0) -> TensorResult: ...


@overload
def concat(
    tensors: Sequence[TensorLike | VariableNode], axis: int = 0
) -> TensorResult | VariableNode: ...


def concat(
    tensors: Sequence[TensorLike | VariableNode], axis: int = 0
) -> TensorResult | VariableNode:
    """Concatenate graph values, Tensors or Variables along an existing axis."""
    from tensors.graph.expression import (
        apply_operation,
        as_graph_operand,
        is_graph_operand,
    )

    if any(is_graph_operand(value) for value in tensors):
        return apply_operation(
            Concat(axis=axis), tuple(as_graph_operand(value) for value in tensors)
        )
    return Concat(axis=axis).forward(
        *(as_tensor_operand(value) for value in _operands(tensors))
    )


__all__ = ["Concat", "concat"]
