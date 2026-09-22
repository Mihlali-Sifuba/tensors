"""Numerically stable softmax and its differentiation rule."""

from __future__ import annotations
from tensors.backend import dispatch as backend_dispatch
import math
from typing import TYPE_CHECKING, Any, List, overload
from tensors._typing import TensorData, TensorLike, TensorResult, TensorValue
from tensors.dtype import float64
from tensors.operations.base import Operation
from tensors.tensor import Tensor
from tensors.graph.expression import as_tensor_operand
from tensors.utils.normalization import shifted_normalization

if TYPE_CHECKING:
    from tensors.graph.node import VariableNode


def _normalize_axis(tensor: Tensor, axis: int) -> int:
    """Return a valid non-negative axis for ``tensor``."""
    if isinstance(axis, bool) or not isinstance(axis, int):
        raise TypeError("softmax axis must be an integer")
    if axis < 0:
        axis += tensor.ndim
    if not 0 <= axis < tensor.ndim:
        raise ValueError(f"Axis {axis} out of bounds for {tensor.ndim}D tensor")
    if tensor.shape[axis] == 0:
        raise ValueError("softmax is not defined along an empty axis")
    return axis


def _axis_layout(tensor: Tensor, axis: int) -> tuple[int, int, int]:
    """Return the row-major group sizes needed to traverse ``axis``."""
    before = 1
    for dimension in tensor.shape[:axis]:
        before *= dimension
    trailing = 1
    for dimension in tensor.shape[axis + 1 :]:
        trailing *= dimension
    return (before, tensor.shape[axis], trailing)


class Softmax(Operation):
    """Normalize values into probabilities along a chosen axis."""

    __slots__ = ("axis",)
    name = "softmax"

    def __init__(self, *, axis: int = -1) -> None:
        object.__setattr__(self, "axis", axis)

    def forward(self, a: Tensor, keepdims: bool = False) -> Tensor:
        """Compute numerically stable softmax values along ``axis``."""
        axis = self.axis
        if not isinstance(keepdims, bool):
            raise TypeError("keepdims must be a bool")
        if keepdims:
            raise ValueError("softmax does not support keepdims")
        axis = _normalize_axis(a, axis)
        before, axis_size, trailing = _axis_layout(a, axis)
        dtype = a.dtype if a.dtype.typecode in {"f", "d"} else float64
        storage = backend_dispatch.execute_softmax(a, axis, dtype=dtype)
        return Tensor._from_owned_storage(storage, dtype=dtype, shape=a.shape)

    def backward(
        self, grad: Tensor, *inputs: Tensor, needs_input_grad: tuple[bool, ...]
    ) -> List[Tensor]:
        """Apply the softmax Jacobian-vector product along ``axis``."""
        a = inputs[0]
        axis = self.axis
        if not isinstance(axis, int):
            raise TypeError("softmax axis must be an integer")
        axis = _normalize_axis(a, axis)
        return [_softmax_vjp_tensor(grad, a, axis)]


def _normalization_components(value: Tensor, axis: int) -> tuple[Tensor, list[float]]:
    """Return probabilities and accurately represented complements."""
    probabilities = Softmax(axis=axis).forward(value)
    before, axis_size, trailing = _axis_layout(value, axis)
    complements = [0.0] * value.size
    for group in range(before):
        group_start = group * axis_size * trailing
        for offset in range(trailing):
            positions = [
                group_start + offset + index * trailing for index in range(axis_size)
            ]
            group_values = [float(value._data[position]) for position in positions]
            if all((math.isfinite(item) for item in group_values)):
                _, _, _, group_complements = shifted_normalization(group_values)
            else:
                group_complements = [
                    1.0 - float(probabilities._data[position]) for position in positions
                ]
            for position, complement in zip(positions, group_complements):
                complements[position] = complement
    return (probabilities, complements)


def _centered_softmax_tensor(grad: Tensor, value: Tensor, axis: int) -> Tensor:
    """Return ``grad - E_softmax(grad)`` without dominant cancellation."""
    from tensors.utils.summation import stable_product_sum

    probabilities, complements = _normalization_components(value, axis)
    before, axis_size, trailing = _axis_layout(value, axis)
    values = [0.0] * value.size
    for group in range(before):
        group_start = group * axis_size * trailing
        for offset in range(trailing):
            positions = [
                group_start + offset + index * trailing for index in range(axis_size)
            ]
            for position in positions:
                terms = [(float(grad._data[position]), complements[position])]
                terms.extend(
                    (
                        (-float(grad._data[other]), float(probabilities._data[other]))
                        for other in positions
                        if other != position
                    )
                )
                values[position] = stable_product_sum(terms)
    return Tensor(values, dtype=grad.dtype, shape=value.shape)


def _softmax_vjp_tensor(grad: Tensor, value: Tensor, axis: int) -> Tensor:
    """Return a cancellation-resistant softmax Jacobian-vector product."""
    storage = backend_dispatch.execute_softmax_gradient(grad, value, axis)
    return Tensor._from_owned_storage(storage, dtype=grad.dtype, shape=value.shape)


def _softmax_expectation_tensor(grad: Tensor, value: Tensor, axis: int) -> Tensor:
    """Broadcast the softmax-weighted expectation of ``grad`` per group."""
    from tensors.utils.summation import stable_product_sum

    probabilities = Softmax(axis=axis).forward(value)
    before, axis_size, trailing = _axis_layout(value, axis)
    values = [0.0] * value.size
    for group in range(before):
        group_start = group * axis_size * trailing
        for offset in range(trailing):
            positions = [
                group_start + offset + index * trailing for index in range(axis_size)
            ]
            expectation = stable_product_sum(
                [
                    (float(grad._data[position]), float(probabilities._data[position]))
                    for position in positions
                ]
            )
            for position in positions:
                values[position] = expectation
    return Tensor(values, dtype=grad.dtype, shape=value.shape)


class SoftmaxCentered(Operation):
    """Differentiable softmax-expectation centering operation."""

    __slots__ = ("axis",)
    name = "softmax_centered"

    def __init__(self, *, axis: int) -> None:
        object.__setattr__(self, "axis", axis)

    def forward(self, grad: Tensor, value: Tensor) -> Tensor:
        axis = self.axis
        return _centered_softmax_tensor(grad, value, axis)

    def backward(
        self, outer_grad: Tensor, *inputs: Tensor, needs_input_grad: tuple[bool, ...]
    ) -> List[Tensor]:
        from tensors.operations.normalization.log_softmax import _log_softmax_vjp_tensor
        from tensors.operations.reductions.sum import Sum

        grad, value = inputs
        need_grad, need_value = needs_input_grad
        axis = self.axis
        assert isinstance(axis, int)
        value_gradient = None
        if need_value:
            total = Sum(axis=axis, keepdims=True).forward(outer_grad)
            expanded_total = _broadcast_reduction(total, value)
            value_vjp = _softmax_vjp_tensor(grad, value, axis)
            value_gradient = Tensor(
                [
                    -scale * derivative
                    for scale, derivative in zip(expanded_total._data, value_vjp._data)
                ],
                dtype=outer_grad.dtype,
                shape=value.shape,
            )
        return [
            _log_softmax_vjp_tensor(outer_grad, value, axis) if need_grad else None,
            value_gradient,
        ]


class SoftmaxGradient(Operation):
    """Differentiable, cancellation-resistant softmax VJP."""

    __slots__ = ("axis",)
    name = "softmax_gradient"

    def __init__(self, *, axis: int) -> None:
        object.__setattr__(self, "axis", axis)

    def forward(self, grad: Tensor, value: Tensor) -> Tensor:
        axis = self.axis
        return _softmax_vjp_tensor(grad, value, axis)

    def backward(
        self, outer_grad: Tensor, *inputs: Tensor, needs_input_grad: tuple[bool, ...]
    ) -> List[Tensor]:
        from tensors.utils.summation import stable_product_sum

        grad, value = inputs
        need_grad, need_value = needs_input_grad
        axis = self.axis
        assert isinstance(axis, int)
        value_gradient = None
        if need_value:
            centered = _centered_softmax_tensor(grad, value, axis)
            projections = _softmax_expectation_tensor(outer_grad, value, axis)
            vector = Tensor(
                [
                    stable_product_sum(
                        [(float(outer), float(difference)), (-float(item), projection)]
                    )
                    for outer, difference, item, projection in zip(
                        outer_grad._data, centered._data, grad._data, projections._data
                    )
                ],
                dtype=outer_grad.dtype,
                shape=value.shape,
            )
            value_gradient = _softmax_vjp_tensor(vector, value, axis)
        return [
            _softmax_vjp_tensor(outer_grad, value, axis) if need_grad else None,
            value_gradient,
        ]


def _broadcast_reduction(reduced: Tensor, value: Tensor) -> Tensor:
    from tensors.utils.broadcasting import broadcast_to

    return broadcast_to(reduced, value.shape)


def _softmax_centered(grad, value, axis: int):
    from tensors.variable import Variable

    operation = SoftmaxCentered(axis=axis)
    return Variable._apply_operation(operation, (grad, value))


def _softmax_vjp(grad, value, axis: int):
    from tensors.variable import Variable

    operation = SoftmaxGradient(axis=axis)
    return Variable._apply_operation(operation, (grad, value))


@overload
def softmax(value: VariableNode, axis: int = -1) -> VariableNode: ...


@overload
def softmax(value: TensorValue, axis: int = -1) -> TensorValue: ...


@overload
def softmax(value: TensorData, axis: int = -1) -> Tensor: ...


def softmax(
    value: TensorLike | VariableNode, axis: int = -1
) -> TensorResult | VariableNode:
    """Return softmax probabilities for a Tensor or differentiable Variable.

    A graph value is applied through the graph: a Variable calculates the
    result now, and a vertex records the operation for a program that runs
    later. The configuration the operation is built with is the one every
    application uses, so a recorded call replays what it was written as.
    """
    from tensors.graph.expression import apply_operation, is_graph_operand

    operation = Softmax(axis=axis)
    if is_graph_operand(value):
        return apply_operation(operation, (value,))
    return operation.forward(as_tensor_operand(value))


__all__ = ["Softmax", "softmax"]
