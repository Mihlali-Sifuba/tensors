"""Multiplication operation."""

from typing import List, Optional, Union
from tensors.backend import execute_multiply
from tensors.dtype import resolve_binary
from tensors.operations.base import Operation
from tensors.tensor import Tensor
from tensors.operations._gradient_shaping import sum_products_to_shape

Scalar = Union[int, float]


class Mul(Operation):
    """Element-wise multiplication — forward and backward."""

    __slots__ = ()
    name = "mul"

    def forward(self, a: Tensor, b: Union[Tensor, Scalar]) -> Tensor:
        """Element-wise multiplication of two tensors or a tensor and a scalar."""
        if not isinstance(b, (int, float, Tensor)):
            raise TypeError(f"Unsupported: {type(b)}")
        # Promotion for a tensor operand, conversion for a scalar.
        dtype, other = resolve_binary(a.dtype, b)
        output_shape = (
            a.shape.broadcast_with(b.shape) if isinstance(b, Tensor) else a.shape
        )
        accelerated = execute_multiply(a, other, dtype=dtype, output_shape=output_shape)
        return Tensor._from_owned_storage(accelerated, dtype=dtype, shape=output_shape)

    def backward(
        self, grad: Tensor, *inputs: Tensor, needs_input_grad: tuple[bool, ...]
    ) -> List[Optional[Tensor]]:
        a, b = inputs
        need_left, need_right = needs_input_grad
        return [
            sum_products_to_shape(grad, b, a.shape) if need_left else None,
            sum_products_to_shape(grad, a, b.shape) if need_right else None,
        ]

    def backward_graph(self, grad, *inputs, needs_input_grad: tuple[bool, ...]):
        """Build a differentiable VJP for multiplication."""
        left, right = inputs
        need_left, need_right = needs_input_grad
        from tensors.operations._gradient_shaping import sum_products_to_shape_graph

        return [
            sum_products_to_shape_graph(grad, right, left.shape) if need_left else None,
            (
                sum_products_to_shape_graph(grad, left, right.shape)
                if need_right
                else None
            ),
        ]


multiply = Mul().forward
