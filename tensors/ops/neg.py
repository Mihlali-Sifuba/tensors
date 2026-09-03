"""Negation operation."""

from typing import List

from ..backend import execute_negate
from ..graph.operation import Operation
from ..tensor import Tensor
from ..dtype import negation_dtype


class Neg(Operation):
    """Negation — forward and backward."""

    __slots__ = ()
    name = "neg"

    def forward(self, a: Tensor) -> Tensor:
        """Negate all elements of a tensor."""
        dtype = negation_dtype(a.dtype)
        accelerated = execute_negate(a, dtype=dtype)
        if accelerated is not None:
            return Tensor._from_owned_storage(accelerated, dtype=dtype, shape=a.shape)
        data = [-x for x in a._data]
        return Tensor(data, dtype=dtype, shape=a.shape)

    def backward(self, grad: Tensor, *inputs: Tensor) -> List[Tensor]:
        return [self.forward(grad)]

    def backward_graph(self, grad, *inputs):
        """Build a differentiable VJP for negation."""
        return [-grad]


negate = Neg().forward
