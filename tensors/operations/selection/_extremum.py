"""Shared structure for the elementwise extremum selections.

``minimum`` and ``maximum`` select between two broadcast operands by the same
rule, differing only in which comparison wins and which dispatch entry point
evaluates it. The selection semantics, the tie and NaN handling, and the
higher-order derivative construction live here; each operation owns its own
module.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar, Optional

from tensors.dtype import result_dtype
from tensors.graph.expression import as_tensor_operand
from tensors.operations.vjp import sum_to_shape
from tensors.operations.base import Operation
from tensors.tensor import Tensor
from tensors.utils.broadcasting import broadcast_tensors


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


def _is_nan(value: int | float) -> bool:
    return isinstance(value, float) and math.isnan(value)


class _ElementwiseExtremum(Operation):
    """Shared elementwise extremum forward and gradient rules."""

    __slots__ = ()
    select_maximum: ClassVar[bool] = False

    def _weights(
        self, left: Tensor, right: Tensor, *, higher_order: bool
    ) -> tuple[list[float], list[float]]:
        expanded_left, expanded_right = broadcast_tensors(left, right)
        left_weights = []
        right_weights = []
        for left_value, right_value in zip(expanded_left._data, expanded_right._data):
            if _is_nan(left_value) or _is_nan(right_value):
                if higher_order:
                    raise ValueError(
                        "Higher-order derivatives of elementwise extrema are undefined at NaN"
                    )
                left_weights.append(math.nan)
                right_weights.append(math.nan)
                continue
            if left_value == right_value:
                if higher_order:
                    raise ValueError(
                        "Higher-order derivatives of elementwise extrema are undefined at ties"
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
        return (left_weights, right_weights)

    @staticmethod
    def _gradients(
        grad: Tensor, left: Tensor, right: Tensor, storages: tuple[Any, Any]
    ) -> list[Optional[Tensor]]:
        """Shape each requested extremum VJP back onto its own operand."""
        left_storage, right_storage = storages
        return [
            (
                sum_to_shape(
                    Tensor._from_owned_storage(
                        left_storage, dtype=grad.dtype, shape=grad.shape
                    ),
                    left.shape,
                )
                if left_storage is not None
                else None
            ),
            (
                sum_to_shape(
                    Tensor._from_owned_storage(
                        right_storage, dtype=grad.dtype, shape=grad.shape
                    ),
                    right.shape,
                )
                if right_storage is not None
                else None
            ),
        ]

    def backward_graph(self, grad, *inputs, needs_input_grad: tuple[bool, ...]):
        from tensors.operations.vjp import (
            masked_value_graph,
            sum_to_shape,
            zero_like_graph,
        )

        left, right = inputs
        need_left, need_right = needs_input_grad
        left_weights, right_weights = self._weights(
            left.data, right.data, higher_order=True
        )

        def masked(weights: list[float], target: Any) -> Any:
            mask = Tensor(weights, dtype=grad.dtype, shape=grad.shape)
            return sum_to_shape(
                masked_value_graph(grad, mask), target.shape
            ) + zero_like_graph(target)

        return [
            masked(left_weights, left) if need_left else None,
            masked(right_weights, right) if need_right else None,
        ]


def _extremum(operation: Operation, left: Any, right: Any) -> Any:
    """Apply an extremum as the kinds of its two operands imply.

    A vertex is answered first and on its own terms: it names a value
    that does not exist, so neither operand can be read for the dtype a
    Python scalar is promoted against below, and the expression is
    recorded instead of calculated.
    """
    from tensors.graph.expression import apply_operation, as_graph_operand
    from tensors.graph.node import VariableNode
    from tensors.variable import Variable

    if isinstance(left, VariableNode) or isinstance(right, VariableNode):
        return apply_operation(
            operation, (as_graph_operand(left), as_graph_operand(right))
        )
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
        left_variable = (
            left if left_is_variable else Variable(left_tensor, requires_grad=False)
        )
        right_variable = (
            right if right_is_variable else Variable(right_tensor, requires_grad=False)
        )
        return Variable._apply_operation(operation, (left_variable, right_variable))
    return operation.forward(left_tensor, right_tensor)
