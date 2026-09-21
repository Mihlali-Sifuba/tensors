"""Elementwise square root and its differentiation rule."""

from __future__ import annotations
from tensors.backend import dispatch as backend_dispatch
import math as _math
from typing import TYPE_CHECKING, Any, List, overload
from tensors._typing import TensorData, TensorLike, TensorResult, TensorValue
from tensors.dtype import float64
from tensors.operations.base import Operation, UNARY_DEMAND
from tensors.tensor import Tensor
from tensors.graph.expression import as_tensor_operand

if TYPE_CHECKING:
    from tensors.graph.node import VariableNode


class Sqrt(Operation):
    """Elementwise square root with a reverse-mode gradient rule."""

    __slots__ = ()
    name = "sqrt"

    def forward(self, a: Tensor) -> Tensor:
        """Take each element's square root, converting an integer operand.

        Sqrt keeps the shape and, alone among the migrated elementary
        operations so far, can change the dtype: a floating operand keeps its
        format and an integer operand is answered in ``float64``. Those are
        the Tensor semantics this operation owns; `execute_sqrt` owns where
        the root is taken. See docs/sqrt-semantics.md.
        """
        dtype = a.dtype if a.dtype.typecode in {"f", "d"} else float64
        output_shape = a.shape
        return Tensor._from_owned_storage(
            backend_dispatch.execute_sqrt(a, dtype=dtype, output_shape=output_shape),
            dtype=dtype,
            shape=output_shape,
        )

    def backward(
        self, grad: Tensor, *inputs: Tensor, needs_input_grad: tuple[bool, ...]
    ) -> List[Tensor]:
        """Scale the upstream gradient by ``1 / (2 * sqrt(x))``.

        Shape and dtype agreement are Tensor semantics and are settled here;
        `execute_sqrt_gradient` owns where the three specified steps run.
        See docs/sqrt-semantics.md section 6.
        """
        value = inputs[0]
        _validate_vjp_operands(grad, value)
        dtype = value.dtype
        output_shape = value.shape
        return [
            Tensor._from_owned_storage(
                backend_dispatch.execute_sqrt_gradient(
                    grad, value, dtype=dtype, output_shape=output_shape
                ),
                dtype=dtype,
                shape=output_shape,
            )
        ]

    def backward_graph(self, grad, *inputs, needs_input_grad: tuple[bool, ...]):
        """Build the same VJP as a graph vertex, not a frozen host check.

        The previous implementation read the materialised host values to
        decide the zero domain and then built ``grad / (2 * sqrt(x))`` from
        public operations. That pulled device values to Python and froze the
        domain decision at the values the graph was built with, so a replay
        onto a zero primal would not have raised. :class:`SqrtVJP` records
        the VJP instead and re-executes it, domain check included.
        """
        from tensors.variable import Variable

        value = inputs[0]
        return [Variable._apply_operation(SqrtVJP(), (grad, value))]


class SqrtVJP(Operation):
    """Internal graph-building first-order VJP for :class:`Sqrt`.

    This is not public API. No facade re-exports it, and it exists so that
    `Sqrt.backward_graph` can record the VJP as a graph vertex — carrying
    its zero domain check with it — rather than decide the domain from host
    values once, at graph-build time.
    """

    __slots__ = ()
    name = "sqrt_vjp"

    def forward(self, grad: Tensor, value: Tensor) -> Tensor:
        """Evaluate the first-order VJP through the ordinary eager path."""
        return Sqrt().backward(grad, value, needs_input_grad=UNARY_DEMAND)[0]

    def backward(
        self, outer_grad: Tensor, *inputs: Tensor, needs_input_grad: tuple[bool, ...]
    ) -> List[Tensor]:
        """Differentiate the genuine VJP.

        With respect to the **upstream gradient** the VJP is linear, so the
        partial is ``1 / (2 * sqrt(x))`` scaled by the outer gradient, which
        is this operation applied again.

        With respect to the **primal** it is
        ``-g / (4 * x * sqrt(x))``, built from the governed operations so it
        follows the same arithmetic contract as the first VJP. A zero primal
        cannot reach here: the forward VJP raised.

        The minus sign is carried by the scalar rather than by negating the
        numerator, deliberately. ``negate`` has not been migrated yet, so it
        still applies a workload threshold and answers a small tensor with
        Python storage — which the division would then reject as a residency
        mismatch. Every operation used here is one whose execution is
        already governed.
        """
        grad, value = inputs
        need_grad, need_value = needs_input_grad

        upstream_partial = None
        if need_grad:
            upstream_partial = Sqrt().backward(
                outer_grad, value, needs_input_grad=UNARY_DEMAND
            )[0]

        primal_partial = None
        if need_value:
            root = Sqrt().forward(value)
            primal_partial = (outer_grad * grad) / (-4.0 * value * root)

        return [upstream_partial, primal_partial]

    def backward_graph(self, outer_grad, *inputs, needs_input_grad: tuple[bool, ...]):
        """Build that higher-order rule as graph vertices.

        The primal partial is written with public graph operations, so any
        further derivative follows from the graph rather than from another
        hand-written rule.
        """
        from tensors.variable import Variable

        grad, value = inputs
        need_grad, need_value = needs_input_grad

        upstream_partial = None
        if need_grad:
            upstream_partial = Variable._apply_operation(SqrtVJP(), (outer_grad, value))

        primal_partial = None
        if need_value:
            primal_partial = (outer_grad * grad) / (-4.0 * value * sqrt(value))

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
def sqrt(value: VariableNode) -> VariableNode: ...


@overload
def sqrt(value: TensorValue) -> TensorValue: ...


@overload
def sqrt(value: TensorData) -> Tensor: ...


def sqrt(value: TensorLike | VariableNode) -> TensorResult | VariableNode:
    """Return the elementwise square root of a graph value or Tensor.

    A graph value is applied through the graph: a Variable calculates the
    result now, and a vertex records the operation for a program that runs
    later.
    """
    from tensors.graph.expression import apply_operation, is_graph_operand

    if is_graph_operand(value):
        return apply_operation(Sqrt(), (value,))
    value = as_tensor_operand(value)
    return Sqrt().forward(value)


__all__ = ["Sqrt", "sqrt"]


def _sqrt(value):
    if value < 0:
        raise ValueError("sqrt is only defined for non-negative values")
    return _math.sqrt(float(value))


def _sqrt_gradient(upstream, value):
    if value == 0:
        raise ValueError("sqrt derivative is undefined at zero")
    return upstream / (2.0 * _sqrt(value))
