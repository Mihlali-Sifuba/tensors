"""Differentiable selection with a constant condition mask."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, overload

from .._typing import TensorData, TensorLike, TensorResult
from ..backend import execute_where, execute_where_gradient
from ..dtype import result_dtype
from ..ops._utils import sum_to_shape
from ..ops.operation import Operation
from ..tensor import Tensor
from ..utils.broadcasting import broadcast_to

if TYPE_CHECKING:
    from ..variable import Variable


def _tensor(value: Any, *, dtype=None) -> Tensor:
    from ..variable import Variable

    if isinstance(value, Variable):
        return value.data
    if isinstance(value, Tensor):
        return value
    scalar_dtype = (
        result_dtype(dtype, value)
        if dtype is not None and isinstance(value, (int, float))
        else None
    )
    return Tensor(value, dtype=scalar_dtype)


class Where(Operation):
    """Choose values from two broadcastable inputs using a fixed condition."""

    __slots__ = ()
    name = "where"

    def forward(self, condition: Tensor, left: Tensor, right: Tensor) -> Tensor:
        shape = condition.shape.broadcast_with(left.shape).broadcast_with(
            right.shape
        )
        dtype = result_dtype(left.dtype, right)
        accelerated = execute_where(
            condition,
            left,
            right,
            dtype=dtype,
            output_shape=shape,
        )
        if accelerated is not None:
            return Tensor._from_owned_storage(accelerated, dtype=dtype, shape=shape)
        expanded_condition = broadcast_to(condition, shape)
        expanded_left = broadcast_to(left, shape)
        expanded_right = broadcast_to(right, shape)
        return Tensor(
            [
                left_value if selected != 0 else right_value
                for selected, left_value, right_value in zip(
                    expanded_condition._data,
                    expanded_left._data,
                    expanded_right._data,
                )
            ],
            dtype=dtype,
            shape=shape,
        )

    def backward(
        self,
        grad: Tensor,
        *inputs: Tensor,
        needs_input_grad: tuple[bool, ...],
    ) -> list[Optional[Tensor]]:
        condition, left, right = inputs
        need_condition, need_left, need_right = needs_input_grad
        # ``where`` rejects a differentiable condition, so its derivative is
        # never requested; the branch remains for contract completeness.
        condition_gradient = (
            Tensor._from_values(
                [0.0] * condition.shape.size,
                grad.dtype,
                condition.shape,
            )
            if need_condition
            else None
        )
        accelerated = execute_where_gradient(
            grad,
            condition,
            needs_input_grad=(need_left, need_right),
        )
        if accelerated is not None:
            left_storage, right_storage = accelerated
            return [
                condition_gradient,
                sum_to_shape(
                    Tensor._from_owned_storage(
                        left_storage,
                        dtype=grad.dtype,
                        shape=grad.shape,
                    ),
                    left.shape,
                )
                if left_storage is not None
                else None,
                sum_to_shape(
                    Tensor._from_owned_storage(
                        right_storage,
                        dtype=grad.dtype,
                        shape=grad.shape,
                    ),
                    right.shape,
                )
                if right_storage is not None
                else None,
            ]
        expanded_condition = broadcast_to(condition, grad.shape)
        left_gradient = None
        if need_left:
            left_gradient = sum_to_shape(
                Tensor(
                    [
                        upstream if selected != 0 else 0.0
                        for upstream, selected in zip(
                            grad._data,
                            expanded_condition._data,
                        )
                    ],
                    dtype=grad.dtype,
                    shape=grad.shape,
                ),
                left.shape,
            )
        right_gradient = None
        if need_right:
            right_gradient = sum_to_shape(
                Tensor(
                    [
                        upstream if selected == 0 else 0.0
                        for upstream, selected in zip(
                            grad._data,
                            expanded_condition._data,
                        )
                    ],
                    dtype=grad.dtype,
                    shape=grad.shape,
                ),
                right.shape,
            )
        return [condition_gradient, left_gradient, right_gradient]

    def backward_graph(
        self,
        grad,
        *inputs,
        needs_input_grad: tuple[bool, ...],
    ):
        from ..ops._utils import (
            masked_value_graph,
            sum_to_shape_graph,
            zero_like_graph,
        )

        condition, left, right = inputs
        need_condition, need_left, need_right = needs_input_grad
        expanded = broadcast_to(condition.data, grad.shape)
        left_gradient = None
        if need_left:
            left_mask = Tensor(
                [1.0 if item != 0 else 0.0 for item in expanded._data],
                dtype=grad.dtype,
                shape=grad.shape,
            )
            left_gradient = sum_to_shape_graph(
                masked_value_graph(grad, left_mask),
                left.shape,
            ) + zero_like_graph(left)
        right_gradient = None
        if need_right:
            right_mask = Tensor(
                [1.0 if item == 0 else 0.0 for item in expanded._data],
                dtype=grad.dtype,
                shape=grad.shape,
            )
            right_gradient = sum_to_shape_graph(
                masked_value_graph(grad, right_mask),
                right.shape,
            ) + zero_like_graph(right)
        return [
            zero_like_graph(condition) if need_condition else None,
            left_gradient,
            right_gradient,
        ]


@overload
def where(
    condition: TensorLike,
    left: Variable,
    right: TensorLike,
) -> Variable: ...


@overload
def where(
    condition: TensorLike,
    left: TensorLike,
    right: Variable,
) -> Variable: ...


@overload
def where(
    condition: TensorLike,
    left: TensorData,
    right: TensorData,
) -> Tensor: ...


def where(
    condition: TensorLike,
    left: TensorLike,
    right: TensorLike,
) -> TensorResult:
    """Select elements from ``left`` or ``right`` using a nonzero mask."""
    from ..variable import Variable

    if isinstance(condition, Variable) and condition.requires_grad:
        raise TypeError("where condition cannot require gradients")
    condition_tensor = _tensor(condition)
    left_is_variable = isinstance(left, Variable)
    right_is_variable = isinstance(right, Variable)
    reference_dtype = (
        left.dtype if left_is_variable or isinstance(left, Tensor)
        else right.dtype if right_is_variable or isinstance(right, Tensor)
        else None
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
        return Variable._record_operation(
            operation.forward(condition_variable.data, left_variable.data, right_variable.data),
            operation,
            (condition_variable, left_variable, right_variable),
        )
    return Where().forward(condition_tensor, left_tensor, right_tensor)


__all__ = ["Where", "where"]
