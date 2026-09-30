"""The fused product reduction, ``ReduceToShape(A ⊙ B, S)``."""

from __future__ import annotations
from tensors.operations.base import Operation
from tensors.tensor import Tensor


class ProductSumToShape(Operation):
    """Multiply two operands and reduce the product to a target shape.

    ``F(A, B; S) = ReduceToShape(A ⊙ B, S)``: an elementwise product over the
    operands' broadcast shape, summed along the axes that ``S`` does not have
    or has as singletons. The target shape is the operation's own state, so a
    recorded invocation reduces to the shape it was written with.

    It is a contraction and follows the package's contraction contract
    (docs/summation-semantics.md): each product is formed once, rounded to
    the dtype, and the products that reduce to one target position are
    summed in logical order with the pairwise tree. A product that overflows
    is therefore an infinity in its group, exactly as it would be for
    ``sum_to_shape(left * right, shape)``; keeping it one operation keeps the
    derivative rule and the recorded graph together.

    Multiplication's vector-Jacobian product is its best-known caller — the
    gradient with respect to one operand is the upstream gradient times the
    other, reduced back — but the computation is meaningful on its own terms,
    and it is an :class:`~tensors.operations.base.Operation` because it
    defines a forward and a derivative rule, not because of where it is
    currently used.
    """

    __slots__ = ("target_shape",)
    name = "product_sum_to_shape"

    def __init__(self, *, target_shape: tuple[int, ...]) -> None:
        object.__setattr__(self, "target_shape", target_shape)

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        """Multiply in the dtype, then reduce the products pairwise.

        Each product is rounded once to the dtype and the products that reach
        one target position are summed in logical order with the package's
        pairwise tree, the contraction contract of
        docs/summation-semantics.md.
        """
        from tensors.backend import execute_sum_products_to_shape

        shape = self.target_shape
        accelerated = execute_sum_products_to_shape(left, right, shape)
        return Tensor.from_backend_storage(accelerated, dtype=left.dtype, shape=shape)

    def backward(self, grad, *inputs, needs_input_grad: tuple[bool, ...]):
        """Differentiate the product reduction, which is itself one.

        Each operand's derivative is the same shape of computation as the
        multiplication VJP this operation implements: the upstream gradient
        times the other operand, summed back to that operand's shape. So it
        is expressed as this operation again, and every derivative level
        follows the same products-then-pairwise-sum contract.

        The gradient arrives at the reduced shape and the reduction needs it
        at the shape the products had. Multiplying by ones is that expansion,
        exactly — ``x * 1`` is ``x`` for every value a gradient can hold —
        and it keeps the work on the selected backend, where the host-side
        broadcast it replaces did not.
        """
        from tensors.creation import ones
        from tensors.graph.expression import apply_operation, is_graph_operand

        left, right = inputs
        common_shape = left.shape.broadcast_with(right.shape)
        expanded = (
            grad
            if grad.shape == common_shape
            else grad * ones(common_shape, dtype=grad.dtype)
        )
        gradients = []
        for operand, factor, requested in (
            (left, right, needs_input_grad[0]),
            (right, left, needs_input_grad[1]),
        ):
            if not requested:
                gradients.append(None)
                continue
            reduction = ProductSumToShape(target_shape=operand.shape)
            gradients.append(
                apply_operation(reduction, (expanded, factor))
                if is_graph_operand(expanded)
                else reduction.forward(expanded, factor)
            )
        return gradients


__all__ = ["ProductSumToShape"]
