"""Elementwise inverse hyperbolic sine and its differentiation rule."""

from __future__ import annotations
from tensors.backend import dispatch as backend_dispatch
import math
from typing import TYPE_CHECKING, Any, overload
from tensors._typing import TensorData, TensorLike, TensorResult, TensorValue
from tensors.dtype import float64
from tensors.operations.base import Operation
from tensors.tensor import Tensor
from tensors.graph.expression import as_tensor_operand

if TYPE_CHECKING:
    from tensors.graph.node import VariableNode


class ArcSinh(Operation):
    """Elementwise inverse hyperbolic sine over the real numbers."""

    __slots__ = ()
    name = "arcsinh"

    def forward(self, value: Tensor) -> Tensor:
        dtype = value.dtype if value.dtype.typecode in {"f", "d"} else float64
        return Tensor._from_owned_storage(
            backend_dispatch.execute_arcsinh(value, dtype=dtype),
            dtype=dtype,
            shape=value.shape,
        )

    def backward(
        self, grad: Tensor, *inputs: Tensor, needs_input_grad: tuple[bool, ...]
    ) -> list[Tensor]:
        value = inputs[0]
        return [
            Tensor._from_owned_storage(
                backend_dispatch.execute_arcsinh_gradient(grad, value),
                dtype=grad.dtype,
                shape=value.shape,
            )
        ]

    def backward_graph(self, grad, *inputs, needs_input_grad: tuple[bool, ...]):
        """Build a stable differentiable VJP for inverse hyperbolic sine."""
        from tensors.operations.elementary.abs import abs
        from tensors.operations.elementary.sqrt import sqrt
        from tensors.operations.selection.where import where

        value = inputs[0]
        scale = 1.0 + abs(value)
        reciprocal_scale = 1.0 / scale
        normalized_value = value / scale
        stable_derivative = reciprocal_scale / sqrt(
            reciprocal_scale**2.0 + normalized_value**2.0
        )
        direct_derivative = 1.0 / sqrt(1.0 + value**2.0)
        large_mask = Tensor(
            [1.0 if math.fabs(float(item)) > 1.0 else 0.0 for item in value.data._data],
            dtype=grad.dtype,
            shape=value.shape,
        )
        return [grad * where(large_mask, stable_derivative, direct_derivative)]


@overload
def arcsinh(value: VariableNode) -> VariableNode: ...


@overload
def arcsinh(value: TensorValue) -> TensorValue: ...


@overload
def arcsinh(value: TensorData) -> Tensor: ...


def arcsinh(value: TensorLike | VariableNode) -> TensorResult | VariableNode:
    """Return the elementwise inverse hyperbolic sine.

    A graph value is applied through the graph: a Variable calculates the
    result now, and a vertex records the operation for a program that runs
    later.
    """
    from tensors.graph.expression import apply_operation, is_graph_operand

    if is_graph_operand(value):
        return apply_operation(ArcSinh(), (value,))
    value = as_tensor_operand(value)
    return ArcSinh().forward(value)


__all__ = ["ArcSinh", "arcsinh"]


def _gradient(upstream, value):
    value = float(value)
    magnitude = abs(value)
    if math.isinf(magnitude):
        derivative = 0.0
    elif magnitude <= 1.0:
        derivative = 1.0 / math.sqrt(1.0 + value * value)
    else:
        reciprocal = 1.0 / magnitude
        derivative = reciprocal / math.sqrt(1.0 + reciprocal * reciprocal)
    return upstream * derivative
