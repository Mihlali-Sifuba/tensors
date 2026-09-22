"""Subtraction operation."""

from typing import Union
from tensors.backend import execute_subtract
from tensors.dtype import convert_scalar, resolve_result_dtype
from tensors.operations.base import Operation
from tensors.operations.vjp import sum_to_shape
from tensors.tensor import Tensor

Scalar = Union[int, float]


class Sub(Operation):
    """Element-wise subtraction — forward and backward."""

    __slots__ = ()
    name = "sub"

    def forward(self, a: Tensor, b: Union[Tensor, Scalar]) -> Tensor:
        """Element-wise subtraction of two tensors or a tensor and a scalar."""
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
        accelerated = execute_subtract(
            a, other, dtype=dtype, output_shape=output_shape
        )
        return Tensor._from_owned_storage(accelerated, dtype=dtype, shape=output_shape)

    def backward(self, grad, *inputs, needs_input_grad: tuple[bool, ...]):
        """Reduce the upstream gradient, negated for the right operand.

        Subtraction's derivative is one with respect to its left operand and
        minus one with respect to its right, so each VJP is the upstream
        gradient — negated for the right — reduced over the axes the forward
        broadcast stretched. The negation comes first, as it always has: it
        is exact at every value, so the order changes nothing numerically,
        and keeping it means the reduction sees the same operand it did.

        This is the only derivative subtraction defines. It is written
        against operations rather than against Tensors, so the operands
        decide what the statements mean: given Tensors they calculate, and
        given Variables the same statements record a differentiable graph.

        Both steps run on the selected backend, which is the contract ``-``
        has always answered under: negation is one operation however it is
        reached, and the reduction asks for the same execution the addition
        VJP does.

        An unrequested operand costs nothing: neither its negation nor its
        reduction is formed.
        """
        from tensors.graph.expression import apply_operation, is_graph_operand

        gradients = []
        for operand, negated, requested in (
            (inputs[0], False, needs_input_grad[0]),
            (inputs[1], True, needs_input_grad[1]),
        ):
            if not requested:
                gradients.append(None)
                continue
            contribution = -grad if negated else grad
            gradients.append(
                sum_to_shape(contribution, operand.shape)
            )
        return gradients


subtract = Sub().forward


def subtract_scalar(left: Scalar, right: Tensor) -> Tensor:
    """Return ``left - right`` for a scalar left operand.

    Reflected subtraction converts the scalar against the tensor's declared
    dtype and keeps that dtype, exactly as the forward form does. Evaluating
    it as ``-right + left`` would not: negating an unsigned tensor widens it,
    and the scalar would then be measured against the wider dtype.
    """
    converted = convert_scalar(left, right.dtype)
    dtype = right.dtype
    accelerated = execute_subtract(
        converted, right, dtype=dtype, output_shape=right.shape
    )
    return Tensor._from_owned_storage(accelerated, dtype=dtype, shape=right.shape)
