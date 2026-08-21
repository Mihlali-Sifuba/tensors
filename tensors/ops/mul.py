"""Multiplication operation."""

from typing import List, Union

from ..dtype import result_dtype
from ..tensor import Tensor
from ..utils.broadcasting import broadcast_tensors
from ._utils import sum_products_to_shape


Scalar = Union[int, float]


class Mul:
    """Element-wise multiplication — forward and backward."""

    @staticmethod
    def forward(a: Tensor, b: Union[Tensor, Scalar]) -> Tensor:
        """Element-wise multiplication of two tensors or a tensor and a scalar."""
        dtype = result_dtype(a.dtype, b)
        if isinstance(b, (int, float)):
            data = [x * b for x in a._data]
            return Tensor(data, dtype=dtype, shape=a.shape)
        if isinstance(b, Tensor):
            a, b = broadcast_tensors(a, b)
            data = [x * y for x, y in zip(a._data, b._data)]
            return Tensor(data, dtype=dtype, shape=a.shape)
        raise TypeError(f"Unsupported: {type(b)}")

    @staticmethod
    def backward(grad: Tensor, *inputs: Tensor, **kwargs: object) -> List[Tensor]:
        if len(inputs) == 2:
            a, b = inputs
            expanded_a, expanded_b = broadcast_tensors(a, b)
            return [
                sum_products_to_shape(grad, expanded_b, a.shape),
                sum_products_to_shape(grad, expanded_a, b.shape),
            ]
        scalar = kwargs.get("scalar", 1.0)
        assert isinstance(scalar, (int, float))
        return [Tensor([g * scalar for g in grad._data], dtype=grad.dtype, shape=grad.shape)]

    @staticmethod
    def backward_graph(grad, *inputs, **kwargs: object):
        """Build a differentiable VJP for multiplication."""
        if len(inputs) == 1:
            scalar = kwargs.get("scalar", 1.0)
            return [grad * scalar]
        left, right = inputs
        from ._utils import sum_products_to_shape_graph
        return [
            sum_products_to_shape_graph(grad, right, left.shape),
            sum_products_to_shape_graph(grad, left, right.shape),
        ]
