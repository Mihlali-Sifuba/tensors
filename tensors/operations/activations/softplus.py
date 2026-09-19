"""Elementwise softplus and its differentiation rule."""

from __future__ import annotations
from tensors.backend import dispatch as backend_dispatch
import math as _math
from typing import TYPE_CHECKING, Any, List, overload
from tensors._typing import TensorData, TensorLike, TensorResult, TensorValue
from tensors.dtype import float64
from tensors.operations.base import Operation
from tensors.tensor import Tensor
from tensors.graph.expression import as_tensor_operand

if TYPE_CHECKING:
    from tensors.graph.node import VariableNode


class Softplus(Operation):
    """Elementwise softplus with a reverse-mode gradient rule."""

    __slots__ = ()
    name = "softplus"

    def forward(self, a: Tensor) -> Tensor:
        dtype = a.dtype if a.dtype.typecode in {"f", "d"} else float64
        return Tensor._from_owned_storage(
            backend_dispatch.execute_softplus(a, dtype=dtype),
            dtype=dtype,
            shape=a.shape,
        )

    def backward(
        self, grad: Tensor, *inputs: Tensor, needs_input_grad: tuple[bool, ...]
    ) -> List[Tensor]:
        a = inputs[0]
        return [
            Tensor._from_owned_storage(
                backend_dispatch.execute_softplus_gradient(grad, a),
                dtype=grad.dtype,
                shape=a.shape,
            )
        ]

    def backward_graph(self, grad, *inputs, needs_input_grad: tuple[bool, ...]):
        """Build a differentiable VJP for softplus."""
        from tensors.operations.activations.sigmoid import sigmoid

        return [grad * sigmoid(inputs[0])]


@overload
def softplus(value: VariableNode) -> VariableNode: ...


@overload
def softplus(value: TensorValue) -> TensorValue: ...


@overload
def softplus(value: TensorData) -> Tensor: ...


def softplus(value: TensorLike | VariableNode) -> TensorResult | VariableNode:
    """Return the elementwise softplus of a graph value or Tensor.

    A graph value is applied through the graph: a Variable calculates the
    result now, and a vertex records the operation for a program that runs
    later.
    """
    from tensors.graph.expression import apply_operation, is_graph_operand

    if is_graph_operand(value):
        return apply_operation(Softplus(), (value,))
    value = as_tensor_operand(value)
    return Softplus().forward(value)


__all__ = ["Softplus", "softplus"]


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = _math.exp(-value)
        return 1.0 / (1.0 + z)
    z = _math.exp(value)
    return z / (1.0 + z)


def _softplus(value):
    value = float(value)
    return _math.log1p(_math.exp(-abs(value))) + max(value, 0.0)
