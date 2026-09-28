"""Multiplication operation."""

from typing import Union
from tensors.backend import execute_multiply
from tensors.dtype import convert_scalar, resolve_result_dtype
from tensors.operations.base import Operation
from tensors.tensor import Tensor
from tensors.operations.reductions.product_sum_to_shape import (
    ProductSumToShape,
)

Scalar = Union[int, float]


class Mul(Operation):
    """Element-wise multiplication — forward and backward."""

    __slots__ = ()
    name = "mul"

    def forward(self, a: Tensor, b: Union[Tensor, Scalar]) -> Tensor:
        """Element-wise multiplication of two tensors or a tensor and a scalar."""
        if not isinstance(b, (int, float, Tensor)):
            raise TypeError(f"Unsupported: {type(b)}")
        # Two declared dtypes promote; a scalar converts to the tensor's
        # dtype and never widens the result. See section 6.2 and 6.5.
        if isinstance(b, Tensor):
            other = b
            dtype = resolve_result_dtype(a.dtype, b.dtype)
            output_shape = a.shape.broadcast_with(b.shape)
        else:
            other = convert_scalar(b, a.dtype)
            dtype = a.dtype
            output_shape = a.shape
        accelerated = execute_multiply(
            a, other, dtype=dtype, output_shape=output_shape
        )
        return Tensor._from_owned_storage(accelerated, dtype=dtype, shape=output_shape)

    def backward(self, grad, *inputs, needs_input_grad: tuple[bool, ...]):
        """Weight the upstream gradient by the other operand, then reduce it.

        The derivative of ``a * b`` with respect to ``a`` is ``b``, so each
        operand's VJP is the upstream gradient times the other operand,
        summed back over the axes the forward broadcast stretched.

        Those two steps stay one operation. Multiplying first can overflow to
        infinities that then cancel to NaN, or underflow to zero, where the
        exact reduced result is representable, so the products are grouped
        before they are rounded. That is the specified vector-Jacobian
        product of multiplication, and expressing it as a multiply followed
        by a separate reduction would quietly change it.

        This is the only derivative multiplication defines. The fused
        reduction is applied as an operation rather than called as a
        function, so the operands decide what the statement means: given
        Tensors it calculates, and given Variables it records a
        differentiable graph that can be differentiated again.

        An unrequested operand costs nothing: its product is never formed.
        """
        from tensors.graph.expression import apply_operation, is_graph_operand

        left, right = inputs
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
                apply_operation(reduction, (grad, factor))
                if is_graph_operand(grad)
                else reduction.forward(grad, factor)
            )
        return gradients


multiply = Mul().forward
