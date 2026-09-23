"""Differentiable selection with a constant condition mask."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, overload

from tensors._typing import TensorData, TensorLike, TensorResult
from tensors.backend import execute_where, execute_where_gradient
from tensors.dtype import result_dtype
from tensors.graph.expression import as_tensor_operand
from tensors.operations.base import Operation
from tensors.operations.gradient_primitives import sum_to_shape
from tensors.tensor import Tensor

if TYPE_CHECKING:
    from tensors.graph.node import VariableNode
    from tensors.variable import Variable


def _tensor(value: Any, *, dtype=None) -> Tensor:
    from tensors.variable import Variable

    if isinstance(value, Variable):
        return value.data
    if isinstance(value, Tensor):
        return value
    scalar_dtype = (
        result_dtype(dtype, value)
        if dtype is not None and isinstance(value, (int, float))
        else None
    )
    return as_tensor_operand(value, dtype=scalar_dtype)


class Where(Operation):
    """Choose values from two broadcastable inputs using a fixed condition."""

    __slots__ = ()
    name = "where"

    def forward(self, condition: Tensor, left: Tensor, right: Tensor) -> Tensor:
        shape = condition.shape.broadcast_with(left.shape).broadcast_with(right.shape)
        dtype = result_dtype(left.dtype, right)
        storage = execute_where(condition, left, right, dtype=dtype, output_shape=shape)
        return Tensor._from_owned_storage(storage, dtype=dtype, shape=shape)

    def backward(
        self, grad: Tensor, *inputs: Tensor, needs_input_grad: tuple[bool, ...]
    ) -> list[Optional[Tensor]]:
        """Route ``G`` to the selected data branch and reduce broadcast axes."""
        from tensors.graph.expression import apply_operation, is_graph_operand

        condition, left, right = inputs
        need_condition, need_left, need_right = needs_input_grad
        output_shape = condition.shape.broadcast_with(left.shape).broadcast_with(
            right.shape
        )
        if grad.shape != output_shape:
            raise ValueError(
                f"Gradient shape {grad.shape} does not match where shape "
                f"{output_shape}"
            )
        dtype = result_dtype(left.dtype, right)
        if grad.dtype is not dtype:
            raise ValueError(
                f"Gradient dtype {grad.dtype.name} does not match where dtype "
                f"{dtype.name}"
            )
        condition_gradient = (
            sum_to_shape(grad * 0.0, condition.shape) if need_condition else None
        )
        gradients: list[Optional[Tensor]] = [condition_gradient]
        for needed, branch, select_left in (
            (need_left, left, True),
            (need_right, right, False),
        ):
            if not needed:
                gradients.append(None)
                continue
            operation = WhereVJP(select_left=select_left, output_shape=output_shape)
            contribution = (
                apply_operation(operation, (grad, condition, branch))
                if is_graph_operand(grad)
                else operation.forward(grad, condition, branch)
            )
            gradients.append(sum_to_shape(contribution, branch.shape))
        return gradients


class WhereVJP(Operation):
    """Internal graph node for one backend-native branch of the where VJP."""

    __slots__ = ("select_left", "output_shape")
    name = "where_vjp"

    def __init__(self, *, select_left: bool, output_shape: tuple[int, ...]) -> None:
        object.__setattr__(self, "select_left", select_left)
        object.__setattr__(self, "output_shape", output_shape)

    def forward(self, grad: Tensor, condition: Tensor, branch: Tensor) -> Tensor:
        output_shape = self.output_shape
        condition.shape.broadcast_with(branch.shape).broadcast_with(output_shape)
        if grad.shape != output_shape:
            raise ValueError(
                f"Gradient shape {grad.shape} does not match where branch shape "
                f"{output_shape}"
            )
        left_storage, right_storage = execute_where_gradient(
            grad,
            condition,
            dtype=grad.dtype,
            output_shape=output_shape,
            needs_input_grad=(self.select_left, not self.select_left),
        )
        storage = left_storage if self.select_left else right_storage
        if storage is None:
            raise RuntimeError("where VJP did not return its requested branch")
        return Tensor._from_owned_storage(storage, dtype=grad.dtype, shape=output_shape)

    def backward(
        self, outer_grad: Tensor, *inputs: Tensor, needs_input_grad: tuple[bool, ...]
    ) -> list[Optional[Tensor]]:
        from tensors.graph.expression import apply_operation, is_graph_operand

        _, condition, branch = inputs
        grad_partial = None
        if needs_input_grad[0]:
            operation = WhereVJP(
                select_left=self.select_left, output_shape=self.output_shape
            )
            grad_partial = (
                apply_operation(operation, (outer_grad, condition, branch))
                if is_graph_operand(outer_grad)
                else operation.forward(outer_grad, condition, branch)
            )
        condition_partial = (
            sum_to_shape(outer_grad * 0.0, condition.shape)
            if needs_input_grad[1]
            else None
        )
        branch_partial = (
            sum_to_shape(outer_grad * 0.0, branch.shape)
            if needs_input_grad[2]
            else None
        )
        return [grad_partial, condition_partial, branch_partial]


@overload
def where(
    condition: VariableNode,
    left: TensorLike | VariableNode,
    right: TensorLike | VariableNode,
) -> VariableNode: ...


@overload
def where(
    condition: TensorLike, left: VariableNode, right: TensorLike | VariableNode
) -> VariableNode: ...


@overload
def where(
    condition: TensorLike, left: TensorLike, right: VariableNode
) -> VariableNode: ...


@overload
def where(condition: TensorLike, left: Variable, right: TensorLike) -> Variable: ...


@overload
def where(condition: TensorLike, left: TensorLike, right: Variable) -> Variable: ...


@overload
def where(condition: TensorLike, left: TensorData, right: TensorData) -> Tensor: ...


def where(
    condition: TensorLike | VariableNode,
    left: TensorLike | VariableNode,
    right: TensorLike | VariableNode,
) -> TensorResult | VariableNode:
    """Select elements from ``left`` or ``right`` using a nonzero mask.

    A graph value in any of the three positions applies the selection
    through the graph, recording the condition, the chosen branch and
    the rejected one as the first, second and third operands. A vertex
    names a value that does not exist, so the condition is not
    evaluated and neither branch can be read for the dtype a Python
    scalar is promoted against below.
    """
    from tensors.graph.expression import apply_operation, as_graph_operand
    from tensors.graph.node import VariableNode
    from tensors.variable import Variable

    if isinstance(condition, Variable) and condition.requires_grad:
        raise TypeError("where condition cannot require gradients")
    if any((isinstance(operand, VariableNode) for operand in (condition, left, right))):
        return apply_operation(
            Where(),
            (
                as_graph_operand(condition),
                as_graph_operand(left),
                as_graph_operand(right),
            ),
        )
    condition_tensor = _tensor(condition)
    left_is_variable = isinstance(left, Variable)
    right_is_variable = isinstance(right, Variable)
    reference_dtype = (
        left.dtype
        if left_is_variable or isinstance(left, Tensor)
        else right.dtype if right_is_variable or isinstance(right, Tensor) else None
    )
    left_tensor = _tensor(left, dtype=reference_dtype)
    right_tensor = _tensor(right, dtype=reference_dtype)
    if left_is_variable or right_is_variable:
        condition_variable = (
            condition
            if isinstance(condition, Variable)
            else Variable(condition_tensor, requires_grad=False)
        )
        left_variable = (
            left
            if isinstance(left, Variable)
            else Variable(left_tensor, requires_grad=False)
        )
        right_variable = (
            right
            if isinstance(right, Variable)
            else Variable(right_tensor, requires_grad=False)
        )
        operation = Where()
        return Variable._apply_operation(
            operation, (condition_variable, left_variable, right_variable)
        )
    return Where().forward(condition_tensor, left_tensor, right_tensor)


__all__ = ["Where", "where"]
