"""Elementwise sign and its differentiation rule."""

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


class Sign(Operation):
    """Elementwise sign with a zero derivative away from zero."""

    __slots__ = ()
    name = "sign"

    def forward(self, value: Tensor) -> Tensor:
        """Classify each element, preserving the operand's dtype and shape.

        The Tensor semantics are settled here — sign changes neither — and
        `execute_sign` owns where the classification runs. See
        docs/sign-semantics.md.
        """
        dtype = value.dtype
        output_shape = value.shape
        return Tensor._from_owned_storage(
            backend_dispatch.execute_sign(
                value, dtype=dtype, output_shape=output_shape
            ),
            dtype=dtype,
            shape=output_shape,
        )

    def backward(
        self, grad: Tensor, *inputs: Tensor, needs_input_grad: tuple[bool, ...]
    ) -> list[Tensor]:
        """Route the upstream gradient to zero away from zero.

        Shape and dtype agreement are Tensor semantics and are settled here;
        `execute_sign_gradient` owns where the routing runs. See
        docs/sign-semantics.md §6.
        """
        value = inputs[0]
        _validate_vjp_operands(grad, value)
        dtype = value.dtype
        output_shape = value.shape
        return [
            Tensor._from_owned_storage(
                backend_dispatch.execute_sign_gradient(
                    grad, value, dtype=dtype, output_shape=output_shape
                ),
                dtype=dtype,
                shape=output_shape,
            )
        ]


class SignVJP(Operation):
    """Internal graph-building first-order VJP for :class:`Sign`.

    This is not public API. No facade re-exports it, and it exists so that
    `Sign.backward` can record the VJP as a graph vertex rather than
    materialise a constant from host values. Its own ``backward`` supplies
    the higher-order rule.
    """

    __slots__ = ()
    name = "sign_vjp"

    def forward(self, grad: Tensor, value: Tensor) -> Tensor:
        """Evaluate the first-order VJP through the ordinary eager path."""
        return Sign().backward(grad, value, needs_input_grad=UNARY_DEMAND)[0]

    def backward(
        self, outer_grad: Tensor, *inputs: Tensor, needs_input_grad: tuple[bool, ...]
    ) -> list[Tensor]:
        """Differentiate a locally constant VJP.

        Away from zero the first VJP is constant in both operands, so both
        partial derivatives are zero; where the primal is NaN the VJP is NaN
        and so is its derivative. That is the same routing the first VJP
        performs, so it is evaluated the same way rather than reasoned about
        twice. A zero primal cannot reach here: the forward VJP raised.
        """
        grad, value = inputs
        pattern = Sign().backward(outer_grad, value, needs_input_grad=UNARY_DEMAND)[0]
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
def sign(value: VariableNode) -> VariableNode: ...


@overload
def sign(value: TensorValue) -> TensorValue: ...


@overload
def sign(value: TensorData) -> Tensor: ...


def sign(value: TensorLike | VariableNode) -> TensorResult | VariableNode:
    """Return -1, 0, or 1 according to each element's sign.

    A graph value is applied through the graph: a Variable calculates the
    result now, and a vertex records the operation for a program that runs
    later.
    """
    from tensors.graph.expression import apply_operation, is_graph_operand

    if is_graph_operand(value):
        return apply_operation(Sign(), (value,))
    value = as_tensor_operand(value)
    return Sign().forward(value)


__all__ = ["Sign", "sign"]


def _sign(value):
    if isinstance(value, float) and math.isnan(value):
        return math.nan
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _gradient(upstream, value):
    if value == 0:
        raise ValueError("sign derivative is undefined at zero")
    if isinstance(value, float) and math.isnan(value):
        return math.nan
    return 0.0
