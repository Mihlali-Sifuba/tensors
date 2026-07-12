"""Subtraction operation."""

from typing import List, Union

from ..tensor import Tensor, _broadcast_tensors
from ..dtype import result_dtype
from ._utils import unbroadcast


Scalar = Union[int, float]


class Sub:
    """Element-wise subtraction — forward and backward."""

    @staticmethod
    def forward(a: Tensor, b: Union[Tensor, Scalar]) -> Tensor:
        """Element-wise subtraction of two tensors or a tensor and a scalar."""
        dtype = result_dtype(a.dtype, b)
        if isinstance(b, (int, float)):
            data = [x - b for x in a._data]
            return Tensor(data, dtype=dtype, shape=a.shape)
        if isinstance(b, Tensor):
            a, b = _broadcast_tensors(a, b)
            data = [x - y for x, y in zip(a._data, b._data)]
            return Tensor(data, dtype=dtype, shape=a.shape)
        raise TypeError(f"Unsupported: {type(b)}")

    @staticmethod
    def backward(grad: Tensor, *inputs: Tensor, **kwargs: object) -> List[Tensor]:
        if len(inputs) == 1:
            return [grad]
        left, right = inputs
        neg = Tensor([-x for x in grad._data], dtype=grad.dtype.typecode, shape=grad.shape)
        return [unbroadcast(grad, left.shape), unbroadcast(neg, right.shape)]

    @staticmethod
    def backward_graph(grad, *inputs, **kwargs: object):
        """Build a differentiable VJP for subtraction."""
        if len(inputs) == 1:
            return [grad]
        left, right = inputs
        if left.shape != right.shape:
            raise NotImplementedError("Higher-order derivatives through broadcast sub are not implemented")
        return [grad, -grad]
