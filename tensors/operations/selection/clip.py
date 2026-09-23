"""Elementwise clipping to constant bounds."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional, overload

from tensors._typing import TensorData, TensorLike, TensorResult, TensorValue
from tensors.backend import execute_clip, execute_clip_gradient
from tensors.dtype import result_dtype
from tensors.graph.expression import as_tensor_operand
from tensors.operations.base import Operation
from tensors.tensor import Tensor

if TYPE_CHECKING:
    from tensors.graph.node import VariableNode


def _validate_bound(name: str, value: int | float | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number or None")
    if isinstance(value, float) and math.isnan(value):
        raise ValueError(f"{name} cannot be NaN")


def _validate_bounds(
    min_value: int | float | None, max_value: int | float | None
) -> None:
    _validate_bound("min_value", min_value)
    _validate_bound("max_value", max_value)
    if min_value is None and max_value is None:
        raise ValueError("clip requires min_value, max_value, or both")
    if min_value is not None and max_value is not None and min_value > max_value:
        raise ValueError("min_value cannot be greater than max_value")


class Clip(Operation):
    """Clip values using a zero subgradient at either finite boundary."""

    __slots__ = ("min_value", "max_value")
    name = "clip"

    def __init__(
        self, *, min_value: int | float | None, max_value: int | float | None
    ) -> None:
        object.__setattr__(self, "min_value", min_value)
        object.__setattr__(self, "max_value", max_value)

    def forward(self, value: Tensor) -> Tensor:
        min_value = self.min_value
        max_value = self.max_value
        _validate_bounds(min_value, max_value)
        dtype = value.dtype
        if min_value is not None:
            dtype = result_dtype(dtype, min_value)
        if max_value is not None:
            dtype = result_dtype(dtype, max_value)
        output_shape = value.shape
        storage = execute_clip(
            value,
            min_value,
            max_value,
            dtype=dtype,
            output_shape=output_shape,
        )
        return Tensor._from_owned_storage(storage, dtype=dtype, shape=output_shape)

    def backward(
        self, grad: Tensor, *inputs: Tensor, needs_input_grad: tuple[bool, ...]
    ) -> list[Optional[Tensor]]:
        """Pass ``G`` strictly between the bounds and zero it at boundaries."""
        if not needs_input_grad[0]:
            return [None]
        from tensors.graph.expression import apply_operation, is_graph_operand

        value = inputs[0]
        operation = ClipVJP(min_value=self.min_value, max_value=self.max_value)
        if is_graph_operand(grad):
            return [apply_operation(operation, (grad, value))]
        return [operation.forward(grad, value)]


class ClipVJP(Operation):
    """Internal graph node for the backend-native clipping VJP."""

    __slots__ = ("min_value", "max_value")
    name = "clip_vjp"

    def __init__(
        self, *, min_value: int | float | None, max_value: int | float | None
    ) -> None:
        object.__setattr__(self, "min_value", min_value)
        object.__setattr__(self, "max_value", max_value)

    def forward(self, grad: Tensor, value: Tensor) -> Tensor:
        min_value = self.min_value
        max_value = self.max_value
        _validate_bounds(min_value, max_value)
        if grad.shape != value.shape:
            raise ValueError(
                f"Gradient shape {grad.shape} does not match value shape {value.shape}"
            )
        dtype = value.dtype
        if min_value is not None:
            dtype = result_dtype(dtype, min_value)
        if max_value is not None:
            dtype = result_dtype(dtype, max_value)
        if grad.dtype is not dtype:
            raise ValueError(
                f"Gradient dtype {grad.dtype.name} does not match clip dtype "
                f"{dtype.name}"
            )
        output_shape = value.shape
        storage = execute_clip_gradient(
            grad,
            value,
            min_value,
            max_value,
            dtype=grad.dtype,
            output_shape=output_shape,
        )
        return Tensor._from_owned_storage(storage, dtype=grad.dtype, shape=output_shape)

    def backward(
        self, outer_grad: Tensor, *inputs: Tensor, needs_input_grad: tuple[bool, ...]
    ) -> list[Optional[Tensor]]:
        from tensors.graph.expression import apply_operation, is_graph_operand

        _, value = inputs
        grad_partial = None
        if needs_input_grad[0]:
            operation = ClipVJP(min_value=self.min_value, max_value=self.max_value)
            grad_partial = (
                apply_operation(operation, (outer_grad, value))
                if is_graph_operand(outer_grad)
                else operation.forward(outer_grad, value)
            )
        value_partial = outer_grad * 0.0 if needs_input_grad[1] else None
        return [grad_partial, value_partial]


@overload
def clip(
    value: VariableNode,
    min_value: int | float | None = None,
    max_value: int | float | None = None,
) -> VariableNode: ...


@overload
def clip(
    value: TensorValue,
    min_value: int | float | None = None,
    max_value: int | float | None = None,
) -> TensorValue: ...


@overload
def clip(
    value: TensorData,
    min_value: int | float | None = None,
    max_value: int | float | None = None,
) -> Tensor: ...


def clip(
    value: TensorLike | VariableNode,
    min_value: int | float | None = None,
    max_value: int | float | None = None,
) -> TensorResult | VariableNode:
    """Clip each value to the inclusive interval defined by the bounds.

    A graph value is applied through the graph: a Variable calculates the
    result now, and a vertex records the operation for a program that runs
    later. The bounds are configuration rather than operands, so they stay
    on the operation and a recorded clip replays the interval it was
    written with.
    """
    from tensors.graph.expression import apply_operation, is_graph_operand

    _validate_bounds(min_value, max_value)
    operation = Clip(min_value=min_value, max_value=max_value)
    if is_graph_operand(value):
        return apply_operation(operation, (value,))
    return operation.forward(as_tensor_operand(value))


__all__ = ["Clip", "clip"]
