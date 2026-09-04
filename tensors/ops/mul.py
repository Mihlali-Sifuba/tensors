"""Multiplication operation."""

from typing import List, Optional, Union

from ..backend import execute_binary
from ..dtype import result_dtype
from ..graph.operation import Operation
from ..tensor import Tensor
from ..utils.broadcasting import broadcast_binary_values
from ._utils import sum_products_to_shape


Scalar = Union[int, float]


class Mul(Operation):
    """Element-wise multiplication — forward and backward."""

    __slots__ = ()
    name = "mul"

    def forward(self, a: Tensor, b: Union[Tensor, Scalar]) -> Tensor:
        """Element-wise multiplication of two tensors or a tensor and a scalar."""
        if not isinstance(b, (int, float, Tensor)):
            raise TypeError(f"Unsupported: {type(b)}")
        dtype = result_dtype(a.dtype, b)
        output_shape = (
            a.shape.broadcast_with(b.shape)
            if isinstance(b, Tensor)
            else a.shape
        )
        accelerated = execute_binary(
            "multiply",
            a,
            b,
            dtype=dtype,
            output_shape=output_shape,
        )
        if accelerated is not None:
            return Tensor._from_owned_storage(accelerated, dtype=dtype, shape=output_shape)
        if isinstance(b, (int, float)):
            data = [x * b for x in a._data]
            return Tensor._from_values(data, dtype, a.shape)
        if isinstance(b, Tensor):
            data = broadcast_binary_values(a, b, output_shape, lambda x, y: x * y)
            return Tensor._from_values(data, dtype, output_shape)
        raise TypeError(f"Unsupported: {type(b)}")

    def backward(
        self,
        grad: Tensor,
        *inputs: Tensor,
        needs_input_grad: tuple[bool, ...],
    ) -> List[Optional[Tensor]]:
        a, b = inputs
        need_left, need_right = needs_input_grad
        return [
            sum_products_to_shape(grad, b, a.shape) if need_left else None,
            sum_products_to_shape(grad, a, b.shape) if need_right else None,
        ]

    def backward_graph(
        self,
        grad,
        *inputs,
        needs_input_grad: tuple[bool, ...],
    ):
        """Build a differentiable VJP for multiplication."""
        left, right = inputs
        need_left, need_right = needs_input_grad
        from ._utils import sum_products_to_shape_graph
        return [
            sum_products_to_shape_graph(grad, right, left.shape)
            if need_left
            else None,
            sum_products_to_shape_graph(grad, left, right.shape)
            if need_right
            else None,
        ]


multiply = Mul().forward
