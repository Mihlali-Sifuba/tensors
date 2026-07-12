"""Negation operation."""

from typing import List

from ..tensor import Tensor
from ..dtype import negation_dtype


class Neg:
    """Negation — forward and backward."""

    @staticmethod
    def forward(a: Tensor) -> Tensor:
        """Negate all elements of a tensor."""
        dtype = negation_dtype(a.dtype)
        data = [-x for x in a._data]
        return Tensor(data, dtype=dtype, shape=a.shape)

    @staticmethod
    def backward(grad: Tensor, *inputs: Tensor, **kwargs: object) -> List[Tensor]:
        neg = Tensor([-x for x in grad._data], dtype=grad.dtype.typecode, shape=grad.shape)
        return [neg]

    @staticmethod
    def backward_graph(grad, *inputs, **kwargs: object):
        """Build a differentiable VJP for negation."""
        return [-grad]
