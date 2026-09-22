"""Elementwise hyperbolic cosine and its differentiation rule."""

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


class Cosh(Operation):
    """Elementwise hyperbolic cosine with a reverse-mode gradient rule."""

    __slots__ = ()
    name = "cosh"

    def forward(self, value: Tensor) -> Tensor:
        dtype = value.dtype if value.dtype.typecode in {"f", "d"} else float64
        return Tensor._from_owned_storage(
            backend_dispatch.execute_cosh(value, dtype=dtype),
            dtype=dtype,
            shape=value.shape,
        )

    def backward(
        self, grad: Tensor, *inputs: Tensor, needs_input_grad: tuple[bool, ...]
    ) -> list[Tensor]:
        value = inputs[0]
        return [
            Tensor._from_owned_storage(
                backend_dispatch.execute_cosh_gradient(grad, value),
                dtype=grad.dtype,
                shape=value.shape,
            )
        ]


@overload
def cosh(value: VariableNode) -> VariableNode: ...


@overload
def cosh(value: TensorValue) -> TensorValue: ...


@overload
def cosh(value: TensorData) -> Tensor: ...


def cosh(value: TensorLike | VariableNode) -> TensorResult | VariableNode:
    """Return the elementwise hyperbolic cosine.

    A graph value is applied through the graph: a Variable calculates the
    result now, and a vertex records the operation for a program that runs
    later.
    """
    from tensors.graph.expression import apply_operation, is_graph_operand

    if is_graph_operand(value):
        return apply_operation(Cosh(), (value,))
    value = as_tensor_operand(value)
    return Cosh().forward(value)


__all__ = ["Cosh", "cosh"]


def _cosh(value):
    try:
        return math.cosh(float(value))
    except OverflowError:
        return math.inf


def _sinh(value):
    try:
        return math.sinh(float(value))
    except OverflowError:
        return math.copysign(math.inf, value)
