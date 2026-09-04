"""Broadcasting elementwise minimum and maximum operations."""

from __future__ import annotations

import math
from typing import ClassVar, TYPE_CHECKING, Any, Optional, overload

from .._typing import TensorData, TensorLike, TensorResult
from ..backend import execute_extremum, execute_extremum_gradient
from ..dtype import result_dtype
from ..ops._utils import sum_to_shape
from ..ops.operation import Operation
from ..tensor import Tensor
from ..utils.broadcasting import broadcast_tensors

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


def _is_nan(value: int | float) -> bool:
    return isinstance(value, float) and math.isnan(value)


class _ElementwiseExtremum(Operation):
    """Shared elementwise extremum forward and gradient rules."""

    __slots__ = ()
    select_maximum: ClassVar[bool] = False

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        shape = left.shape.broadcast_with(right.shape)
        dtype = result_dtype(left.dtype, right)
        operation = "maximum" if self.select_maximum else "minimum"
        accelerated = execute_extremum(
            operation,
            left,
            right,
            dtype=dtype,
            output_shape=shape,
        )
        if accelerated is not None:
            return Tensor._from_owned_storage(accelerated, dtype=dtype, shape=shape)
        expanded_left, expanded_right = broadcast_tensors(left, right)
        values = []
        for left_value, right_value in zip(
            expanded_left._data,
            expanded_right._data,
        ):
            if _is_nan(left_value) or _is_nan(right_value):
                values.append(math.nan)
            elif self.select_maximum:
                values.append(
                    left_value if left_value >= right_value else right_value
                )
            else:
                values.append(
                    left_value if left_value <= right_value else right_value
                )
        return Tensor(
            values,
            dtype=dtype,
            shape=expanded_left.shape,
        )

    def _weights(
        self,
        left: Tensor,
        right: Tensor,
        *,
        higher_order: bool,
    ) -> tuple[list[float], list[float]]:
        expanded_left, expanded_right = broadcast_tensors(left, right)
        left_weights = []
        right_weights = []
        for left_value, right_value in zip(
            expanded_left._data,
            expanded_right._data,
        ):
            if _is_nan(left_value) or _is_nan(right_value):
                if higher_order:
                    raise ValueError(
                        "Higher-order derivatives of elementwise extrema "
                        "are undefined at NaN"
                    )
                left_weights.append(math.nan)
                right_weights.append(math.nan)
                continue
            if left_value == right_value:
                if higher_order:
                    raise ValueError(
                        "Higher-order derivatives of elementwise extrema "
                        "are undefined at ties"
                    )
                left_weights.append(0.5)
                right_weights.append(0.5)
                continue
            left_selected = (
                left_value > right_value
                if self.select_maximum
                else left_value < right_value
            )
            left_weights.append(1.0 if left_selected else 0.0)
            right_weights.append(0.0 if left_selected else 1.0)
        return left_weights, right_weights

    def backward(
        self,
        grad: Tensor,
        *inputs: Tensor,
        needs_input_grad: tuple[bool, ...],
    ) -> list[Optional[Tensor]]:
        left, right = inputs
        need_left, need_right = needs_input_grad
        operation = "maximum" if self.select_maximum else "minimum"
        accelerated = execute_extremum_gradient(
            operation,
            grad,
            left,
            right,
            needs_input_grad=needs_input_grad,
        )
        if accelerated is not None:
            left_storage, right_storage = accelerated
            return [
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
        left_weights, right_weights = self._weights(
            left,
            right,
            higher_order=False,
        )

        def weighted(weights: list[float], target: Tensor) -> Tensor:
            return sum_to_shape(
                Tensor(
                    [
                        upstream * weight
                        for upstream, weight in zip(grad._data, weights)
                    ],
                    dtype=grad.dtype,
                    shape=grad.shape,
                ),
                target.shape,
            )

        return [
            weighted(left_weights, left) if need_left else None,
            weighted(right_weights, right) if need_right else None,
        ]

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

        left, right = inputs
        need_left, need_right = needs_input_grad
        left_weights, right_weights = self._weights(
            left.data,
            right.data,
            higher_order=True,
        )

        def masked(weights: list[float], target: Any) -> Any:
            mask = Tensor(weights, dtype=grad.dtype, shape=grad.shape)
            return sum_to_shape_graph(
                masked_value_graph(grad, mask),
                target.shape,
            ) + zero_like_graph(target)

        return [
            masked(left_weights, left) if need_left else None,
            masked(right_weights, right) if need_right else None,
        ]


class Maximum(_ElementwiseExtremum):
    """Elementwise maximum with broadcasting."""

    __slots__ = ()
    name = "maximum"
    select_maximum = True


class Minimum(_ElementwiseExtremum):
    """Elementwise minimum with broadcasting."""

    __slots__ = ()
    name = "minimum"


def _extremum(operation: Operation, left: Any, right: Any) -> Any:
    from ..variable import Variable

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
        left_variable = left if left_is_variable else Variable(
            left_tensor,
            requires_grad=False,
        )
        right_variable = right if right_is_variable else Variable(
            right_tensor,
            requires_grad=False,
        )
        return Variable._from_operation(
            operation.forward(left_variable.data, right_variable.data),
            operation,
            (left_variable, right_variable),
        )
    return operation.forward(left_tensor, right_tensor)


@overload
def maximum(left: Variable, right: TensorLike) -> Variable: ...


@overload
def maximum(left: TensorLike, right: Variable) -> Variable: ...


@overload
def maximum(left: TensorData, right: TensorData) -> Tensor: ...


def maximum(left: TensorLike, right: TensorLike) -> TensorResult:
    """Return the broadcasting elementwise maximum of two values."""
    return _extremum(Maximum(), left, right)


@overload
def minimum(left: Variable, right: TensorLike) -> Variable: ...


@overload
def minimum(left: TensorLike, right: Variable) -> Variable: ...


@overload
def minimum(left: TensorData, right: TensorData) -> Tensor: ...


def minimum(left: TensorLike, right: TensorLike) -> TensorResult:
    """Return the broadcasting elementwise minimum of two values."""
    return _extremum(Minimum(), left, right)


__all__ = ["Maximum", "Minimum", "maximum", "minimum"]
