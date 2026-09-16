"""Elementwise inverse tangent and its differentiation rule."""

from __future__ import annotations
from tensors.backend import dispatch as backend_dispatch
import math
from typing import TYPE_CHECKING, Any, overload
from .._typing import TensorData, TensorLike, TensorResult, TensorValue
from ..dtype import float64
from ..ops.operation import Operation
from ..tensor import Tensor
from ..graph.expression import as_tensor_operand

if TYPE_CHECKING:
    from ..graph.node import VariableNode


class ArcTan(Operation):
    """Elementwise inverse tangent over the real numbers."""

    __slots__ = ()
    name = "arctan"

    def forward(self, value: Tensor) -> Tensor:
        dtype = value.dtype if value.dtype.typecode in {"f", "d"} else float64
        return Tensor._from_owned_storage(
            backend_dispatch.execute_arctan(value, dtype=dtype),
            dtype=dtype,
            shape=value.shape,
        )

    def backward(
        self, grad: Tensor, *inputs: Tensor, needs_input_grad: tuple[bool, ...]
    ) -> list[Tensor]:
        value = inputs[0]
        return [
            Tensor._from_owned_storage(
                backend_dispatch.execute_arctan_gradient(grad, value),
                dtype=grad.dtype,
                shape=value.shape,
            )
        ]

    def backward_graph(self, grad, *inputs, needs_input_grad: tuple[bool, ...]):
        """Build a differentiable VJP for inverse tangent."""
        from .abs import abs

        value = inputs[0]
        scale = 1.0 + abs(value)
        reciprocal_scale = 1.0 / scale
        normalized_value = value / scale
        reciprocal_square = reciprocal_scale**2.0
        derivative = reciprocal_square / (reciprocal_square + normalized_value**2.0)
        return [grad * derivative]


@overload
def arctan(value: VariableNode) -> VariableNode: ...


@overload
def arctan(value: TensorValue) -> TensorValue: ...


@overload
def arctan(value: TensorData) -> Tensor: ...


def arctan(value: TensorLike | VariableNode) -> TensorResult | VariableNode:
    """Return the elementwise inverse tangent in radians.

    A graph value is applied through the graph: a Variable calculates the
    result now, and a vertex records the operation for a program that runs
    later.
    """
    from ..graph.expression import apply_operation, is_graph_operand

    if is_graph_operand(value):
        return apply_operation(ArcTan(), (value,))
    value = as_tensor_operand(value)
    return ArcTan().forward(value)


__all__ = ["ArcTan", "arctan"]


def _gradient(upstream, value):
    value = float(value)
    if math.isinf(value):
        derivative = 0.0
    elif abs(value) <= 1.0:
        derivative = 1.0 / (1.0 + value * value)
    else:
        reciprocal = 1.0 / value
        square = reciprocal * reciprocal
        derivative = square / (1.0 + square)
    return upstream * derivative
