"""Multiplication operation."""

from typing import List, Union

from ..backend import execute_binary
from ..dtype import result_dtype
from ..graph.operation import Operation
from ..tensor import Tensor
from ..utils.broadcasting import broadcast_binary_values
from ._utils import sum_products_to_shape, zeros_like


Scalar = Union[int, float]


class Mul(Operation):
    """Element-wise multiplication — forward and backward."""

    __slots__ = ("differentiate_left", "differentiate_right")
    name = "mul"

    def __init__(
        self,
        *,
        differentiate_left: bool = True,
        differentiate_right: bool = True,
    ) -> None:
        object.__setattr__(self, "differentiate_left", bool(differentiate_left))
        object.__setattr__(
            self,
            "differentiate_right",
            bool(differentiate_right),
        )

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

    def backward(self, grad: Tensor, *inputs: Tensor) -> List[Tensor]:
        a, b = inputs
        return [
            sum_products_to_shape(grad, b, a.shape)
            if self.differentiate_left
            else zeros_like(a, grad.dtype),
            sum_products_to_shape(grad, a, b.shape)
            if self.differentiate_right
            else zeros_like(b, grad.dtype),
        ]

    def backward_graph(self, grad, *inputs):
        """Build a differentiable VJP for multiplication."""
        left, right = inputs
        from ._utils import sum_products_to_shape_graph
        return [
            sum_products_to_shape_graph(grad, right, left.shape),
            sum_products_to_shape_graph(grad, left, right.shape),
        ]


multiply = Mul().forward
