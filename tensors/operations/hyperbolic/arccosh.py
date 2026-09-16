"""Elementwise inverse hyperbolic cosine and its differentiation rule."""

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


class ArcCosh(Operation):
    """Elementwise inverse hyperbolic cosine on the real interval [1, infinity)."""

    __slots__ = ()
    name = "arccosh"

    def forward(self, value: Tensor) -> Tensor:
        dtype = value.dtype if value.dtype.typecode in {"f", "d"} else float64
        return Tensor._from_owned_storage(
            backend_dispatch.execute_arccosh(value, dtype=dtype),
            dtype=dtype,
            shape=value.shape,
        )

    def backward(
        self, grad: Tensor, *inputs: Tensor, needs_input_grad: tuple[bool, ...]
    ) -> list[Tensor]:
        value = inputs[0]
        return [
            Tensor._from_owned_storage(
                backend_dispatch.execute_arccosh_gradient(grad, value),
                dtype=grad.dtype,
                shape=value.shape,
            )
        ]

    def backward_graph(self, grad, *inputs, needs_input_grad: tuple[bool, ...]):
        """Build a differentiable VJP for inverse hyperbolic cosine."""
        from tensors.operations.elementary.sqrt import sqrt

        value = inputs[0]
        if any((item == 1.0 for item in value.data._data)):
            raise ValueError("arccosh derivative is undefined at 1")
        denominator = sqrt(value - 1.0) * sqrt(value + 1.0)
        return [grad / denominator]


@overload
def arccosh(value: VariableNode) -> VariableNode: ...


@overload
def arccosh(value: TensorValue) -> TensorValue: ...


@overload
def arccosh(value: TensorData) -> Tensor: ...


def arccosh(value: TensorLike | VariableNode) -> TensorResult | VariableNode:
    """Return the elementwise inverse hyperbolic cosine.

    A graph value is applied through the graph: a Variable calculates the
    result now, and a vertex records the operation for a program that runs
    later.
    """
    from tensors.graph.expression import apply_operation, is_graph_operand

    if is_graph_operand(value):
        return apply_operation(ArcCosh(), (value,))
    value = as_tensor_operand(value)
    return ArcCosh().forward(value)


__all__ = ["ArcCosh", "arccosh"]


def _arccosh(value):
    if value < 1.0:
        raise ValueError(
            "arccosh is only defined for values greater than or equal to 1"
        )
    return math.acosh(float(value))


def _gradient(upstream, value):
    if value == 1.0:
        raise ValueError("arccosh derivative is undefined at 1")
    value = float(value)
    if math.isinf(value):
        derivative = 0.0
    else:
        derivative = 1.0 / (math.sqrt(value - 1.0) * math.sqrt(value + 1.0))
    return upstream * derivative
