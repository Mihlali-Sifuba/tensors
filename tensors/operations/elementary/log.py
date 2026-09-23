"""Elementwise natural logarithm and its differentiation rule."""

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


class Log(Operation):
    """Elementwise natural logarithm with a reverse-mode gradient rule."""

    __slots__ = ()
    name = "log"

    def forward(self, a: Tensor) -> Tensor:
        """Take the logarithm of every element, promoting an integer operand.

        The Tensor semantics are settled here — the shape is the operand's
        and an integer operand promotes to ``float64``, because ``log`` has
        no integral values to return — and `execute_log` owns where the
        evaluation runs and where the domain is enforced. See
        docs/log-semantics.md.
        """
        dtype = a.dtype if a.dtype.typecode in {"f", "d"} else float64
        output_shape = a.shape
        return Tensor._from_owned_storage(
            backend_dispatch.execute_log(a, dtype=dtype, output_shape=output_shape),
            dtype=dtype,
            shape=output_shape,
        )

    def backward(
        self, grad: Tensor, *inputs: Tensor, needs_input_grad: tuple[bool, ...]
    ) -> List[Optional[Tensor]]:
        """Divide the upstream gradient by the primal.

        Shape and dtype agreement are Tensor semantics and are settled here;
        `execute_log_gradient` owns where the evaluation runs. Only a
        floating operand reaches this method — an integer Tensor cannot
        require a gradient — so the promotion in ``forward`` has nothing to
        undo and the gradient carries the operand's own dtype.

        The domain is not re-checked. The forward refuses a non-positive
        operand, so a primal arriving here from a completed forward pass is
        positive; repeating the test would put a reduction, and on CUDA a
        host round trip, on every reverse pass. See docs/log-semantics.md
        section 6.

        An unrequested gradient costs nothing: no kernel runs and ``None``
        is returned, rather than a value the reverse pass would discard.
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
                backend_dispatch.execute_log_gradient(
                    grad, a, dtype=dtype, output_shape=output_shape
                ),
                dtype=dtype,
                shape=output_shape,
            )
        ]


@overload
def log(value: VariableNode) -> VariableNode: ...


@overload
def log(value: TensorValue) -> TensorValue: ...


@overload
def log(value: TensorData) -> Tensor: ...


def log(value: TensorLike | VariableNode) -> TensorResult | VariableNode:
    """Return the elementwise natural logarithm of a graph value or Tensor.

    A graph value is applied through the graph: a Variable calculates the
    result now, and a vertex records the operation for a program that runs
    later.
    """
    from tensors.graph.expression import apply_operation, is_graph_operand

    if is_graph_operand(value):
        return apply_operation(Log(), (value,))
    value = as_tensor_operand(value)
    return Log().forward(value)


__all__ = ["Log", "log"]
