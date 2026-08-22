"""Negation operation."""

from typing import List

from ..backend import execute_negate
from ..tensor import Tensor
from ..dtype import negation_dtype


class Neg:
    """Negation — forward and backward."""

    @staticmethod
    def forward(a: Tensor) -> Tensor:
        """Negate all elements of a tensor."""
        dtype = negation_dtype(a.dtype)
        accelerated = execute_negate(a, dtype=dtype)
        if accelerated is not None:
            return Tensor(accelerated, dtype=dtype, shape=a.shape)
        data = [-x for x in a._data]
        return Tensor(data, dtype=dtype, shape=a.shape)

    @staticmethod
    def backward(grad: Tensor, *inputs: Tensor, **kwargs: object) -> List[Tensor]:
        return [Neg.forward(grad)]

    @staticmethod
    def backward_graph(grad, *inputs, **kwargs: object):
        """Build a differentiable VJP for negation."""
        return [-grad]
