"""Numerically stable log-softmax and its differentiation rule."""

from __future__ import annotations
from tensors.backend import dispatch as backend_dispatch
from typing import TYPE_CHECKING, Any, List, overload
from tensors._typing import TensorData, TensorLike, TensorResult, TensorValue
from tensors.dtype import float64
from tensors.operations.base import Operation
from tensors.tensor import Tensor
from tensors.graph.expression import as_tensor_operand
from tensors.operations.normalization.softmax import (
    _axis_layout,
    _centered_softmax_tensor,
    _normalize_axis,
    _softmax_vjp,
    _softmax_vjp_tensor,
)

if TYPE_CHECKING:
    from tensors.graph.node import VariableNode


class LogSoftmax(Operation):
    """Normalize logits in log space along one axis."""

    __slots__ = ("axis",)
    name = "log_softmax"

    def __init__(self, *, axis: int = -1) -> None:
        object.__setattr__(self, "axis", axis)

    def forward(self, a: Tensor) -> Tensor:
        axis = self.axis
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise TypeError("log_softmax axis must be an integer")
        axis = _normalize_axis(a, axis)
        before, axis_size, trailing = _axis_layout(a, axis)
        dtype = a.dtype if a.dtype.typecode in {"f", "d"} else float64
        storage = backend_dispatch.execute_log_softmax(a, axis, dtype=dtype)
        return Tensor._from_owned_storage(storage, dtype=dtype, shape=a.shape)

    def backward(
        self, grad: Tensor, *inputs: Tensor, needs_input_grad: tuple[bool, ...]
    ) -> List[Tensor]:
        a = inputs[0]
        axis = self.axis
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise TypeError("log_softmax axis must be an integer")
        axis = _normalize_axis(a, axis)
        if not needs_input_grad[0]:
            return [None]
        from tensors.variable import Variable

        if isinstance(grad, Variable) or isinstance(a, Variable):
            return [_log_softmax_vjp(grad, a, axis)]
        return [_log_softmax_vjp_tensor(grad, a, axis)]


def _log_softmax_vjp_tensor(grad: Tensor, value: Tensor, axis: int) -> Tensor:
    """Return a cancellation-resistant log-softmax VJP."""
    storage = backend_dispatch.execute_log_softmax_gradient(grad, value, axis)
    return Tensor._from_owned_storage(storage, dtype=grad.dtype, shape=value.shape)


class LogSoftmaxGradient(Operation):
    """Differentiable, cancellation-resistant log-softmax VJP."""

    __slots__ = ("axis",)
    name = "log_softmax_gradient"

    def __init__(self, *, axis: int) -> None:
        object.__setattr__(self, "axis", axis)

    def forward(self, grad: Tensor, value: Tensor) -> Tensor:
        axis = self.axis
        return _log_softmax_vjp_tensor(grad, value, axis)

    def backward(
        self, outer_grad: Tensor, *inputs: Tensor, needs_input_grad: tuple[bool, ...]
    ) -> List[Tensor]:
        from tensors.operations.reductions.sum import Sum
        from tensors.utils.broadcasting import broadcast_to

        grad, value = inputs
        need_grad, need_value = needs_input_grad
        axis = self.axis
        assert isinstance(axis, int)
        from tensors.variable import Variable

        if isinstance(outer_grad, Variable):
            from tensors.operations.reductions.sum import sum as reduce_sum

            grad_gradient = None
            if need_grad:
                from tensors.operations.normalization.softmax import softmax

                probabilities = softmax(value, axis=axis)
                grad_gradient = outer_grad - reduce_sum(
                    outer_grad * probabilities, axis=axis, keepdims=True
                )
            value_gradient = None
            if need_value:
                total = reduce_sum(grad, axis=axis, keepdims=True)
                value_gradient = -total * _softmax_vjp(outer_grad, value, axis)
            return [grad_gradient, value_gradient]

        value_gradient = None
        if need_value:
            total = Sum(axis=axis, keepdims=True).forward(grad)
            expanded_total = broadcast_to(total, value.shape)
            value_vjp = _softmax_vjp_tensor(outer_grad, value, axis)
            value_gradient = Tensor(
                [
                    -scale * derivative
                    for scale, derivative in zip(expanded_total._data, value_vjp._data)
                ],
                dtype=outer_grad.dtype,
                shape=value.shape,
            )
        return [
            _centered_softmax_tensor(outer_grad, value, axis) if need_grad else None,
            value_gradient,
        ]


def _log_softmax_vjp(grad, value, axis: int):
    from tensors.variable import Variable

    operation = LogSoftmaxGradient(axis=axis)
    return Variable._apply_operation(operation, (grad, value))


@overload
def log_softmax(value: VariableNode, axis: int = -1) -> VariableNode: ...


@overload
def log_softmax(value: TensorValue, axis: int = -1) -> TensorValue: ...


@overload
def log_softmax(value: TensorData, axis: int = -1) -> Tensor: ...


def log_softmax(
    value: TensorLike | VariableNode, axis: int = -1
) -> TensorResult | VariableNode:
    """Return stable log probabilities along ``axis``.

    A graph value is applied through the graph: a Variable calculates the
    result now, and a vertex records the operation for a program that runs
    later. The configuration the operation is built with is the one every
    application uses, so a recorded call replays what it was written as.
    """
    from tensors.graph.expression import apply_operation, is_graph_operand

    operation = LogSoftmax(axis=axis)
    if is_graph_operand(value):
        return apply_operation(operation, (value,))
    return operation.forward(as_tensor_operand(value))


__all__ = ["LogSoftmax", "log_softmax"]
