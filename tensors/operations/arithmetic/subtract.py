"""Subtraction operation."""

from typing import List, Optional, Union
from tensors.backend import execute_subtract
from tensors.dtype import result_dtype
from tensors.operations.base import Operation
from tensors.tensor import Tensor
from tensors.operations._gradient_shaping import sum_to_shape

Scalar = Union[int, float]


class Sub(Operation):
    """Element-wise subtraction — forward and backward."""

    __slots__ = ()
    name = "sub"

    def forward(self, a: Tensor, b: Union[Tensor, Scalar]) -> Tensor:
        """Element-wise subtraction of two tensors or a tensor and a scalar."""
        if not isinstance(b, (int, float, Tensor)):
            raise TypeError(f"Unsupported: {type(b)}")
        dtype = result_dtype(a.dtype, b)
        output_shape = (
            a.shape.broadcast_with(b.shape) if isinstance(b, Tensor) else a.shape
        )
        accelerated = execute_subtract(a, b, dtype=dtype, output_shape=output_shape)
        return Tensor._from_owned_storage(accelerated, dtype=dtype, shape=output_shape)

    def backward(
        self, grad: Tensor, *inputs: Tensor, needs_input_grad: tuple[bool, ...]
    ) -> List[Optional[Tensor]]:
        left, right = inputs
        need_left, need_right = needs_input_grad
        from tensors.operations.arithmetic.negate import negate

        return [
            sum_to_shape(grad, left.shape) if need_left else None,
            sum_to_shape(negate(grad), right.shape) if need_right else None,
        ]

    def backward_graph(self, grad, *inputs, needs_input_grad: tuple[bool, ...]):
        """Build a differentiable VJP for subtraction."""
        left, right = inputs
        need_left, need_right = needs_input_grad
        from tensors.operations._gradient_shaping import sum_to_shape_graph

        return [
            sum_to_shape_graph(grad, left.shape) if need_left else None,
            sum_to_shape_graph(-grad, right.shape) if need_right else None,
        ]


subtract = Sub().forward
