"""Elementwise rectified linear unit and its differentiation rule."""

from __future__ import annotations
from tensors.backend import dispatch as backend_dispatch
import math
from typing import TYPE_CHECKING, Any, List, overload
from tensors._typing import TensorData, TensorLike, TensorResult, TensorValue
from tensors.operations.base import Operation, UNARY_DEMAND
from tensors.tensor import Tensor
from tensors.graph.expression import as_tensor_operand

if TYPE_CHECKING:
    from tensors.graph.node import VariableNode


class ReLU(Operation):
    """Elementwise rectified linear unit with a reverse-mode gradient rule."""

    __slots__ = ()
    name = "relu"

    def forward(self, value: Tensor) -> Tensor:
        """Rectify each element, preserving the operand's dtype and shape.

        The Tensor semantics are settled here — ReLU changes neither — and
        `execute_relu` owns where the rectification runs. See
        docs/relu-semantics.md.
        """
        dtype = value.dtype
        output_shape = value.shape
        return Tensor._from_owned_storage(
            backend_dispatch.execute_relu(
                value, dtype=dtype, output_shape=output_shape
            ),
            dtype=dtype,
            shape=output_shape,
        )

    def backward(
        self, grad: Tensor, *inputs: Tensor, needs_input_grad: tuple[bool, ...]
    ) -> List[Tensor]:
        """Pass the upstream gradient where the primal is positive.

        Shape and dtype agreement are Tensor semantics and are settled here;
        `execute_relu_gradient` owns where the routing runs. See
        docs/relu-semantics.md section 6.
        """
        value = inputs[0]
        _validate_vjp_operands(grad, value)
        dtype = value.dtype
        output_shape = value.shape
        return [
            Tensor._from_owned_storage(
                backend_dispatch.execute_relu_gradient(
                    grad, value, dtype=dtype, output_shape=output_shape
                ),
                dtype=dtype,
                shape=output_shape,
            )
        ]

    def backward_graph(self, grad, *inputs, needs_input_grad: tuple[bool, ...]):
        """Build the same VJP as a graph vertex, not a frozen host mask.

        The previous implementation built a Python mask from the
        materialised host values and multiplied the upstream gradient by it.
        That pulled device values to Python, froze the branch decision at
        the values the graph was built with, and let the multiplication
        carry an infinity or a NaN into the inactive side.
        :class:`ReLUVJP` records the routing instead and re-executes it.
        """
        from tensors.variable import Variable

        value = inputs[0]
        return [Variable._apply_operation(ReLUVJP(), (grad, value))]


class ReLUVJP(Operation):
    """Internal graph-building first-order VJP for :class:`ReLU`.

    This is not public API. No facade re-exports it, and it exists so that
    `ReLU.backward_graph` can record the VJP as a graph vertex rather than
    materialise a mask from host values.
    """

    __slots__ = ()
    name = "relu_vjp"

    def forward(self, grad: Tensor, value: Tensor) -> Tensor:
        """Evaluate the first-order VJP through the ordinary eager path."""
        return ReLU().backward(grad, value, needs_input_grad=UNARY_DEMAND)[0]

    def backward(
        self, outer_grad: Tensor, *inputs: Tensor, needs_input_grad: tuple[bool, ...]
    ) -> List[Tensor]:
        """Differentiate the routed VJP.

        With respect to the **upstream gradient** the VJP is the same
        routing again — it passes the outer gradient through where the
        primal is positive — so it is evaluated with this same operation.

        With respect to the **primal** the VJP is locally constant away from
        the kink, so that partial is zero, NaN where the primal is NaN, and
        undefined exactly at zero. That is the shape of the sign VJP, so it
        is evaluated by it and its zero error is restated in this
        operation's terms.

        The kink check belongs only to the primal partial, so when that
        partial is not requested it is not computed and a gradient with
        respect to the upstream alone is still returned at a zero primal.
        """
        from tensors.operations.elementary.sign import Sign

        need_grad, need_value = needs_input_grad
        value = inputs[1]

        upstream_partial = None
        if need_grad:
            upstream_partial = ReLU().backward(
                outer_grad, value, needs_input_grad=UNARY_DEMAND
            )[0]

        primal_partial = None
        if need_value:
            try:
                primal_partial = Sign().backward(
                    outer_grad, value, needs_input_grad=UNARY_DEMAND
                )[0]
            except ValueError as error:
                raise ValueError(
                    "relu second derivative is undefined at zero"
                ) from error

        return [upstream_partial, primal_partial]

    def backward_graph(self, outer_grad, *inputs, needs_input_grad: tuple[bool, ...]):
        """Build that higher-order rule as graph vertices."""
        from tensors.operations.elementary.sign import SignVJP
        from tensors.variable import Variable

        need_grad, need_value = needs_input_grad
        value = inputs[1]

        upstream_partial = None
        if need_grad:
            upstream_partial = Variable._apply_operation(ReLUVJP(), (outer_grad, value))

        primal_partial = None
        if need_value:
            try:
                primal_partial = Variable._apply_operation(
                    SignVJP(), (outer_grad, value)
                )
            except ValueError as error:
                raise ValueError(
                    "relu second derivative is undefined at zero"
                ) from error

        return [upstream_partial, primal_partial]


def _validate_vjp_operands(grad: Tensor, value: Tensor) -> None:
    """Hold the upstream gradient to the primal's shape and dtype."""
    if grad.shape != value.shape:
        raise ValueError(
            f"Gradient shape {grad.shape} does not match value shape {value.shape}"
        )
    if grad.dtype is not value.dtype:
        raise ValueError(
            f"Gradient dtype {grad.dtype.name} does not match value dtype "
            f"{value.dtype.name}"
        )


@overload
def relu(value: VariableNode) -> VariableNode: ...


@overload
def relu(value: TensorValue) -> TensorValue: ...


@overload
def relu(value: TensorData) -> Tensor: ...


def relu(value: TensorLike | VariableNode) -> TensorResult | VariableNode:
    """Return the elementwise rectified linear unit of a graph value or Tensor.

    A graph value is applied through the graph: a Variable calculates the
    result now, and a vertex records the operation for a program that runs
    later.
    """
    from tensors.graph.expression import apply_operation, is_graph_operand

    if is_graph_operand(value):
        return apply_operation(ReLU(), (value,))
    value = as_tensor_operand(value)
    return ReLU().forward(value)


__all__ = ["ReLU", "relu"]


def _relu(value):
    if isinstance(value, float) and math.isnan(value):
        return math.nan
    return value if value > 0 else 0


def _gradient(upstream, value):
    if isinstance(value, float) and math.isnan(value):
        return math.nan
    return upstream if value > 0 else 0
