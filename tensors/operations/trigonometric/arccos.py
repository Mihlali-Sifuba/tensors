"""Elementwise inverse cosine and its differentiation rule."""

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


class ArcCos(Operation):
    """Elementwise inverse cosine on the closed real interval [-1, 1]."""

    __slots__ = ()
    name = "arccos"

    def forward(self, value: Tensor) -> Tensor:
        dtype = value.dtype if value.dtype.typecode in {"f", "d"} else float64
        return Tensor._from_owned_storage(
            backend_dispatch.execute_arccos(value, dtype=dtype),
            dtype=dtype,
            shape=value.shape,
        )

    def backward(
        self, grad: Tensor, *inputs: Tensor, needs_input_grad: tuple[bool, ...]
    ) -> list[Tensor]:
        value = inputs[0]
        return [
            Tensor._from_owned_storage(
                backend_dispatch.execute_arccos_gradient(grad, value),
                dtype=grad.dtype,
                shape=value.shape,
            )
        ]

    def backward_graph(self, grad, *inputs, needs_input_grad: tuple[bool, ...]):
        """Build a differentiable VJP for inverse cosine."""
        from tensors.operations.elementary.sqrt import sqrt

        value = inputs[0]
        if any((item == -1.0 or item == 1.0 for item in value.data._data)):
            raise ValueError("arccos derivative is undefined at -1 and 1")
        return [-(grad / sqrt(1.0 - value**2.0))]


@overload
def arccos(value: VariableNode) -> VariableNode: ...


@overload
def arccos(value: TensorValue) -> TensorValue: ...


@overload
def arccos(value: TensorData) -> Tensor: ...


def arccos(value: TensorLike | VariableNode) -> TensorResult | VariableNode:
    """Return the elementwise inverse cosine in radians.

    A graph value is applied through the graph: a Variable calculates the
    result now, and a vertex records the operation for a program that runs
    later.
    """
    from tensors.graph.expression import apply_operation, is_graph_operand

    if is_graph_operand(value):
        return apply_operation(ArcCos(), (value,))
    value = as_tensor_operand(value)
    return ArcCos().forward(value)


__all__ = ["ArcCos", "arccos"]


def _arccos(value):
    if value < -1.0 or value > 1.0:
        raise ValueError("arccos is only defined for values between -1 and 1")
    return math.acos(float(value))


def _arccos_gradient(upstream, value):
    if value == -1.0 or value == 1.0:
        raise ValueError("arccos derivative is undefined at -1 and 1")
    return -upstream / math.sqrt(1.0 - float(value) ** 2.0)
