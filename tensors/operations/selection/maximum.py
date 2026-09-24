"""Broadcasting elementwise maximum."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, overload

from tensors._typing import TensorData, TensorLike, TensorResult
from tensors.backend import execute_maximum, execute_maximum_gradient
from tensors.dtype import result_dtype
from tensors.operations.base import Operation
from tensors.operations.gradient_primitives import sum_to_shape
from tensors.operations.selection._extremum import (
    _ElementwiseExtremum,
    _extremum,
)
from tensors.tensor import Tensor

if TYPE_CHECKING:
    from tensors.graph.node import VariableNode
    from tensors.variable import Variable


class Maximum(_ElementwiseExtremum):
    """Elementwise maximum with broadcasting."""

    __slots__ = ()
    name = "maximum"

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        """Select the larger of each broadcast pair, propagating NaN."""
        shape = left.shape.broadcast_with(right.shape)
        dtype = result_dtype(left.dtype, right)
        storage = execute_maximum(left, right, dtype=dtype, output_shape=shape)
        return Tensor._from_owned_storage(storage, dtype=dtype, shape=shape)

    def backward(
        self, grad: Tensor, *inputs: Tensor, needs_input_grad: tuple[bool, ...]
    ) -> list[Optional[Tensor]]:
        """Route ``G`` to the selected operand, splitting ties equally."""
        if not any(needs_input_grad):
            return [None, None]
        from tensors.graph.expression import apply_operation, is_graph_operand

        left, right = inputs
        output_shape = left.shape.broadcast_with(right.shape)
        if grad.shape != output_shape:
            raise ValueError(
                f"Gradient shape {grad.shape} does not match maximum shape "
                f"{output_shape}"
            )
        dtype = result_dtype(left.dtype, right)
        if grad.dtype is not dtype:
            raise ValueError(
                f"Gradient dtype {grad.dtype.name} does not match maximum dtype "
                f"{dtype.name}"
            )
        if is_graph_operand(grad):
            gradients: list[Optional[Tensor]] = []
            for needed, operand, select_left in (
                (needs_input_grad[0], left, True),
                (needs_input_grad[1], right, False),
            ):
                if not needed:
                    gradients.append(None)
                    continue
                contribution = apply_operation(
                    MaximumVJP(select_left=select_left),
                    (grad, left, right),
                )
                gradients.append(sum_to_shape(contribution, operand.shape))
            return gradients
        storages = execute_maximum_gradient(
            grad,
            left,
            right,
            dtype=grad.dtype,
            output_shape=output_shape,
            needs_input_grad=needs_input_grad,
        )
        return self._gradients(grad, left, right, storages)


class MaximumVJP(Operation):
    """Internal graph node for one backend-native maximum VJP branch."""

    __slots__ = ("select_left",)
    name = "maximum_vjp"

    def __init__(self, *, select_left: bool) -> None:
        object.__setattr__(self, "select_left", select_left)

    def forward(self, grad: Tensor, left: Tensor, right: Tensor) -> Tensor:
        output_shape = left.shape.broadcast_with(right.shape)
        if grad.shape != output_shape:
            raise ValueError(
                f"Gradient shape {grad.shape} does not match maximum shape "
                f"{output_shape}"
            )
        dtype = result_dtype(left.dtype, right)
        if grad.dtype is not dtype:
            raise ValueError(
                f"Gradient dtype {grad.dtype.name} does not match maximum dtype "
                f"{dtype.name}"
            )
        left_storage, right_storage = execute_maximum_gradient(
            grad,
            left,
            right,
            dtype=grad.dtype,
            output_shape=output_shape,
            needs_input_grad=(self.select_left, not self.select_left),
        )
        storage = left_storage if self.select_left else right_storage
        if storage is None:
            raise RuntimeError("maximum VJP did not return its requested branch")
        return Tensor._from_owned_storage(storage, dtype=grad.dtype, shape=output_shape)

    def backward(
        self, outer_grad: Tensor, *inputs: Tensor, needs_input_grad: tuple[bool, ...]
    ) -> list[Optional[Tensor]]:
        from tensors.graph.expression import apply_operation, is_graph_operand

        _, left, right = inputs
        grad_partial = None
        if needs_input_grad[0]:
            operation = MaximumVJP(select_left=self.select_left)
            grad_partial = (
                apply_operation(operation, (outer_grad, left, right))
                if is_graph_operand(outer_grad)
                else operation.forward(outer_grad, left, right)
            )
        left_partial = (
            sum_to_shape(outer_grad * 0.0, left.shape) if needs_input_grad[1] else None
        )
        right_partial = (
            sum_to_shape(outer_grad * 0.0, right.shape) if needs_input_grad[2] else None
        )
        return [grad_partial, left_partial, right_partial]


@overload
def maximum(left: VariableNode, right: TensorLike | VariableNode) -> VariableNode: ...


@overload
def maximum(left: TensorLike, right: VariableNode) -> VariableNode: ...


@overload
def maximum(left: Variable, right: TensorLike) -> Variable: ...


@overload
def maximum(left: TensorLike, right: Variable) -> Variable: ...


@overload
def maximum(left: TensorData, right: TensorData) -> Tensor: ...


def maximum(
    left: TensorLike | VariableNode, right: TensorLike | VariableNode
) -> TensorResult | VariableNode:
    """Return the broadcasting elementwise maximum of two values."""
    return _extremum(Maximum(), left, right)


__all__ = ["Maximum", "maximum"]
