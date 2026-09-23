"""Elementwise exponential and its differentiation rule."""

from __future__ import annotations
from tensors.backend import dispatch as backend_dispatch
from typing import TYPE_CHECKING, List, Optional, overload
from tensors._typing import TensorData, TensorLike, TensorResult, TensorValue
from tensors.dtype import float64
from tensors.operations.base import Operation
from tensors.tensor import Tensor
from tensors.graph.expression import as_tensor_operand

if TYPE_CHECKING:
    from tensors.graph.node import VariableNode


class Exp(Operation):
    """Elementwise exponential with a reverse-mode gradient rule."""

    __slots__ = ()
    name = "exp"

    def forward(self, a: Tensor) -> Tensor:
        """Exponentiate every element, promoting an integer operand.

        The Tensor semantics are settled here — the shape is the operand's
        and an integer operand promotes to ``float64``, because ``exp`` has
        no integral values to return — and `execute_exp` owns where the
        evaluation runs. See docs/exp-semantics.md.
        """
        dtype = a.dtype if a.dtype.typecode in {"f", "d"} else float64
        output_shape = a.shape
        return Tensor._from_owned_storage(
            backend_dispatch.execute_exp(a, dtype=dtype, output_shape=output_shape),
            dtype=dtype,
            shape=output_shape,
        )

    def backward(
        self, grad: Tensor, *inputs: Tensor, needs_input_grad: tuple[bool, ...]
    ) -> List[Optional[Tensor]]:
        """Scale the upstream gradient by ``exp(x)``.

        Shape and dtype agreement are Tensor semantics and are settled here;
        `execute_exp_gradient` owns where the evaluation runs. Only a
        floating operand reaches this method — an integer Tensor cannot
        require a gradient — so the promotion in ``forward`` has nothing to
        undo and the gradient carries the operand's own dtype.

        An unrequested gradient costs nothing: no kernel runs and ``None``
        is returned, rather than a value the reverse pass would discard.
        See docs/exp-semantics.md section 6.
        """
        if not needs_input_grad[0]:
            return [None]
        a = inputs[0]
        if grad.shape != a.shape:
            raise ValueError(
                f"Gradient shape {grad.shape} does not match value shape {a.shape}"
            )
        if grad.dtype is not a.dtype:
            raise ValueError(
                f"Gradient dtype {grad.dtype.name} does not match value dtype "
                f"{a.dtype.name}"
            )
        dtype = a.dtype
        output_shape = a.shape
        return [
            Tensor._from_owned_storage(
                backend_dispatch.execute_exp_gradient(
                    grad, a, dtype=dtype, output_shape=output_shape
                ),
                dtype=dtype,
                shape=output_shape,
            )
        ]


@overload
def exp(value: VariableNode) -> VariableNode: ...


@overload
def exp(value: TensorValue) -> TensorValue: ...


@overload
def exp(value: TensorData) -> Tensor: ...


def exp(value: TensorLike | VariableNode) -> TensorResult | VariableNode:
    """Return the elementwise exponential of a graph value or Tensor.

    A graph value is applied through the graph: a Variable calculates the
    result now, and a vertex records the operation for a program that runs
    later.
    """
    from tensors.graph.expression import apply_operation, is_graph_operand

    if is_graph_operand(value):
        return apply_operation(Exp(), (value,))
    value = as_tensor_operand(value)
    return Exp().forward(value)


__all__ = ["Exp", "exp"]
