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
        undefined exactly at zero. That is the shape of the sign VJP, so
        :class:`ReLUPrimalVJP` evaluates it by that while restating the zero
        error in this operation's terms.

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
            primal_partial = ReLUPrimalVJP().forward(outer_grad, value)

        return [upstream_partial, primal_partial]

    def backward_graph(self, outer_grad, *inputs, needs_input_grad: tuple[bool, ...]):
        """Build that higher-order rule as graph vertices.

        The primal partial is recorded as :class:`ReLUPrimalVJP` rather than
        as the sign VJP it borrows its numbers from. The distinction only
        shows up on replay: a graph built here runs again later, and an
        error raised then comes from the recorded operation's own
        ``forward``, long after any ``try`` around the construction has
        returned. Recording the sign VJP directly would therefore report the
        sign function's error for a second derivative of ReLU.
        """
        from tensors.variable import Variable

        need_grad, need_value = needs_input_grad
        value = inputs[1]

        upstream_partial = None
        if need_grad:
            upstream_partial = Variable._apply_operation(ReLUVJP(), (outer_grad, value))

        primal_partial = None
        if need_value:
            primal_partial = Variable._apply_operation(
                ReLUPrimalVJP(), (outer_grad, value)
            )

        return [upstream_partial, primal_partial]


class ReLUPrimalVJP(Operation):
    """Internal node for the primal partial of :class:`ReLUVJP`.

    This is not public API. No facade re-exports it.

    Numerically it is the sign VJP: zero away from the kink, NaN where the
    primal is NaN, undefined exactly at zero. It exists so that the
    *identity* of the operation survives into a compiled graph. A recorded
    vertex is executed again on every replay, and an error raised then is
    raised by the recorded operation's own ``forward`` — not inside the
    ``backward_graph`` call that recorded it. Recording the sign VJP
    directly would therefore make a replayed second derivative of ReLU
    report the sign function's error, so the translation has to live here,
    where replay will run it.
    """

    __slots__ = ()
    name = "relu_primal_vjp"

    def forward(self, grad: Tensor, value: Tensor) -> Tensor:
        """Evaluate the sign VJP, restating its zero error for ReLU."""
        from tensors.operations.elementary.sign import Sign

        try:
            return Sign().backward(grad, value, needs_input_grad=UNARY_DEMAND)[0]
        except ValueError as error:
            raise ValueError("relu second derivative is undefined at zero") from error

    def backward(
        self, outer_grad: Tensor, *inputs: Tensor, needs_input_grad: tuple[bool, ...]
    ) -> List[Tensor]:
        """Differentiate a partial that is itself locally constant."""
        value = inputs[1]
        pattern = ReLUPrimalVJP().forward(outer_grad, value)
        return [pattern if needed else None for needed in needs_input_grad]

    def backward_graph(self, outer_grad, *inputs, needs_input_grad: tuple[bool, ...]):
        """Build that rule as graph vertices, keeping this operation's error."""
        from tensors.variable import Variable

        value = inputs[1]
        pattern = Variable._apply_operation(ReLUPrimalVJP(), (outer_grad, value))
        return [pattern if needed else None for needed in needs_input_grad]


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
