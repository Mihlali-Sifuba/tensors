"""Differentiable selection with a constant condition mask."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

from .._typing import TensorData, TensorLike, TensorResult
from ..backend import execute_where, execute_where_gradient
from ..dtype import result_dtype
from ..ops._utils import sum_to_shape
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


class Where:
    """Choose values from two broadcastable inputs using a fixed condition."""

    @staticmethod
    def forward(condition: Tensor, left: Tensor, right: Tensor) -> Tensor:
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
            return Tensor(accelerated, dtype=dtype, shape=shape)
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

    @staticmethod
    def backward(
        grad: Tensor,
        *inputs: Tensor,
        **kwargs: object,
    ) -> list[Tensor]:
        condition, left, right = inputs
        accelerated = execute_where_gradient(grad, condition)
        if accelerated is not None:
            left_storage, right_storage = accelerated
            left_gradient = Tensor(
                left_storage,
                dtype=grad.dtype,
                shape=grad.shape,
            )
            right_gradient = Tensor(
                right_storage,
                dtype=grad.dtype,
                shape=grad.shape,
            )
            return [
                Tensor(
                    [0.0] * condition.size,
                    dtype=grad.dtype,
                    shape=condition.shape,
                ),
                sum_to_shape(left_gradient, left.shape),
                sum_to_shape(right_gradient, right.shape),
            ]
        expanded_condition = broadcast_to(condition, grad.shape)
        left_gradient = Tensor(
            [
                upstream if selected != 0 else 0.0
                for upstream, selected in zip(
                    grad._data,
                    expanded_condition._data,
                )
            ],
            dtype=grad.dtype,
            shape=grad.shape,
        )
        right_gradient = Tensor(
            [
                upstream if selected == 0 else 0.0
                for upstream, selected in zip(
                    grad._data,
                    expanded_condition._data,
                )
            ],
            dtype=grad.dtype,
            shape=grad.shape,
        )
        return [
            Tensor(
                [0.0] * condition.size,
                dtype=grad.dtype,
                shape=condition.shape,
            ),
            sum_to_shape(left_gradient, left.shape),
            sum_to_shape(right_gradient, right.shape),
        ]

    @staticmethod
    def backward_graph(grad, *inputs, **kwargs: object):
        from ..ops._utils import (
            masked_value_graph,
            sum_to_shape_graph,
            zero_like_graph,
        )

        condition, left, right = inputs
        expanded = broadcast_to(condition.data, grad.shape)
        left_mask = Tensor(
            [1.0 if item != 0 else 0.0 for item in expanded._data],
            dtype=grad.dtype,
            shape=grad.shape,
        )
        right_mask = Tensor(
            [1.0 if item == 0 else 0.0 for item in expanded._data],
            dtype=grad.dtype,
            shape=grad.shape,
        )
        return [
            zero_like_graph(condition),
            sum_to_shape_graph(
                masked_value_graph(grad, left_mask),
                left.shape,
            ) + zero_like_graph(left),
            sum_to_shape_graph(
                masked_value_graph(grad, right_mask),
                right.shape,
            ) + zero_like_graph(right),
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
        left_variable = left if left_is_variable else Variable(
            left_tensor,
            requires_grad=False,
        )
        right_variable = right if right_is_variable else Variable(
            right_tensor,
            requires_grad=False,
        )
        return Variable._from_operation(
            Where.forward(
                condition_variable.data,
                left_variable.data,
                right_variable.data,
            ),
            "where",
            Where,
            [condition_variable, left_variable, right_variable],
        )
    return Where.forward(condition_tensor, left_tensor, right_tensor)


__all__ = ["Where", "where"]
