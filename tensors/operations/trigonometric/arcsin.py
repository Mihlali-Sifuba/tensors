"""Elementwise inverse sine and its differentiation rule."""

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


class ArcSin(Operation):
    """Elementwise inverse sine on the closed real interval [-1, 1]."""

    __slots__ = ()
    name = "arcsin"

    def forward(self, value: Tensor) -> Tensor:
        dtype = value.dtype if value.dtype.typecode in {"f", "d"} else float64
        return Tensor._from_owned_storage(
            backend_dispatch.execute_arcsin(value, dtype=dtype),
            dtype=dtype,
            shape=value.shape,
        )

    def backward(
        self, grad: Tensor, *inputs: Tensor, needs_input_grad: tuple[bool, ...]
    ) -> list[Tensor]:
        value = inputs[0]
        return [
            Tensor._from_owned_storage(
                backend_dispatch.execute_arcsin_gradient(grad, value),
                dtype=grad.dtype,
                shape=value.shape,
            )
        ]


@overload
def arcsin(value: VariableNode) -> VariableNode: ...


@overload
def arcsin(value: TensorValue) -> TensorValue: ...


@overload
def arcsin(value: TensorData) -> Tensor: ...


def arcsin(value: TensorLike | VariableNode) -> TensorResult | VariableNode:
    """Return the elementwise inverse sine in radians.

    A graph value is applied through the graph: a Variable calculates the
    result now, and a vertex records the operation for a program that runs
    later.
    """
    from tensors.graph.expression import apply_operation, is_graph_operand

    if is_graph_operand(value):
        return apply_operation(ArcSin(), (value,))
    value = as_tensor_operand(value)
    return ArcSin().forward(value)


__all__ = ["ArcSin", "arcsin"]


def _arcsin(value):
    if value < -1.0 or value > 1.0:
        raise ValueError("arcsin is only defined for values between -1 and 1")
    return math.asin(float(value))


def _arcsin_gradient(upstream, value):
    if value == -1.0 or value == 1.0:
        raise ValueError("arcsin derivative is undefined at -1 and 1")
    return upstream / math.sqrt(1.0 - float(value) ** 2.0)
