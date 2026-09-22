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
