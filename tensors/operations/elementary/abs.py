"""Elementwise absolute value and its differentiation rule."""

from __future__ import annotations
from tensors.backend import dispatch as backend_dispatch
import math
from typing import TYPE_CHECKING, Any, overload
from tensors._typing import TensorData, TensorLike, TensorResult, TensorValue
from tensors.operations.base import Operation, UNARY_DEMAND
from tensors.tensor import Tensor
from tensors.graph.expression import as_tensor_operand

if TYPE_CHECKING:
    from tensors.graph.node import VariableNode


class Abs(Operation):
    """Elementwise absolute value with a zero subgradient at zero."""

    __slots__ = ()
    name = "abs"

    def forward(self, value: Tensor) -> Tensor:
        """Take each element's magnitude, preserving dtype and shape.

        The Tensor semantics are settled here — abs changes neither — and
        `execute_abs` owns where the magnitude is taken and which inputs it
        refuses. See docs/abs-semantics.md.
        """
        dtype = value.dtype
        output_shape = value.shape
        return Tensor._from_owned_storage(
            backend_dispatch.execute_abs(value, dtype=dtype, output_shape=output_shape),
            dtype=dtype,
            shape=output_shape,
        )

    def backward(
        self, grad: Tensor, *inputs: Tensor, needs_input_grad: tuple[bool, ...]
    ) -> list[Tensor]:
        """Route the upstream gradient by the sign of the primal.

        Shape and dtype agreement are Tensor semantics and are settled here;
        `execute_abs_gradient` owns where the routing runs. See
        docs/abs-semantics.md section 6.
        """
        value = inputs[0]
        _validate_vjp_operands(grad, value)
        dtype = value.dtype
        output_shape = value.shape
        return [
            Tensor._from_owned_storage(
                backend_dispatch.execute_abs_gradient(
                    grad, value, dtype=dtype, output_shape=output_shape
                ),
                dtype=dtype,
                shape=output_shape,
            )
        ]


class AbsVJP(Operation):
    """Internal graph-building first-order VJP for :class:`Abs`.

    This is not public API. No facade re-exports it, and it exists so that
    `Abs.backward` can record the VJP as a graph vertex rather than
    materialise masks from host values.
    """

    __slots__ = ()
    name = "abs_vjp"

    def forward(self, grad: Tensor, value: Tensor) -> Tensor:
        """Evaluate the first-order VJP through the ordinary eager path."""
        return Abs().backward(grad, value, needs_input_grad=UNARY_DEMAND)[0]

    def backward(
        self, outer_grad: Tensor, *inputs: Tensor, needs_input_grad: tuple[bool, ...]
    ) -> list[Tensor]:
        """Differentiate the routed VJP.

        With respect to the **upstream gradient** the VJP is the same
        routing again: it passes the outer gradient through where the primal
        is positive and negates it where the primal is negative. So it is
        evaluated with this same operation.

        With respect to the **primal** the VJP is locally constant away from
        the kink, so that partial is zero, NaN where the primal is NaN, and
        undefined exactly at zero. That is the shape of the sign VJP, so
        :class:`AbsPrimalVJP` evaluates it by that rather than reasoning
        about it a second time, while restating the zero error in this
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
            upstream_partial = Abs().backward(
                outer_grad, value, needs_input_grad=UNARY_DEMAND
            )[0]

        primal_partial = None
        if need_value:
            primal_partial = AbsPrimalVJP().forward(outer_grad, value)

        return [upstream_partial, primal_partial]


class AbsPrimalVJP(Operation):
    """Internal node for the primal partial of :class:`AbsVJP`.

    This is not public API. No facade re-exports it.

    Numerically it is the sign VJP: zero away from the kink, NaN where the
    primal is NaN, undefined exactly at zero. It exists so that the
    *identity* of the operation survives into a compiled graph. A recorded
    vertex is executed again on every replay, and an error raised then is
    raised by the recorded operation's own ``forward`` — not inside the
    call that recorded it. Recording the sign VJP
    directly would therefore make a replayed second derivative of abs report
    the sign function's error, so the translation has to live here, where
    replay will run it.
    """

    __slots__ = ()
    name = "abs_primal_vjp"

    def forward(self, grad: Tensor, value: Tensor) -> Tensor:
        """Evaluate the sign VJP, restating its zero error for abs."""
        from tensors.operations.elementary.sign import Sign

        try:
            return Sign().backward(grad, value, needs_input_grad=UNARY_DEMAND)[0]
        except ValueError as error:
            raise ValueError("abs second derivative is undefined at zero") from error

    def backward(
        self, outer_grad: Tensor, *inputs: Tensor, needs_input_grad: tuple[bool, ...]
    ) -> list[Tensor]:
        """Differentiate a partial that is itself locally constant."""
        value = inputs[1]
        pattern = AbsPrimalVJP().forward(outer_grad, value)
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
def abs(value: VariableNode) -> VariableNode: ...


@overload
def abs(value: TensorValue) -> TensorValue: ...


@overload
def abs(value: TensorData) -> Tensor: ...


def abs(value: TensorLike | VariableNode) -> TensorResult | VariableNode:
    """Return the elementwise absolute value of a graph value or Tensor.

    A graph value is applied through the graph: a Variable calculates the
    result now, and a vertex records the operation for a program that runs
    later.
    """
    from tensors.graph.expression import apply_operation, is_graph_operand

    if is_graph_operand(value):
        return apply_operation(Abs(), (value,))
    value = as_tensor_operand(value)
    return Abs().forward(value)


__all__ = ["Abs", "abs"]


def _abs_gradient(upstream, value):
    if isinstance(value, float) and math.isnan(value):
        return math.nan
    if value > 0:
        return upstream
    if value < 0:
        return -upstream
    return 0.0
