"""Negation operation."""

from typing import List
from tensors.backend import execute_negate
from tensors.ops.operation import Operation
from tensors.tensor import Tensor
from tensors.dtype import negation_dtype


class Neg(Operation):
    """Negation — forward and backward."""

    __slots__ = ()
    name = "neg"

    def forward(self, a: Tensor) -> Tensor:
        """Negate all elements of a tensor."""
        dtype = negation_dtype(a.dtype)
        accelerated = execute_negate(a, dtype=dtype)
        return Tensor._from_owned_storage(accelerated, dtype=dtype, shape=a.shape)

    def backward(
        self, grad: Tensor, *inputs: Tensor, needs_input_grad: tuple[bool, ...]
    ) -> List[Tensor]:
        return [self.forward(grad)]

    def backward_graph(self, grad, *inputs, needs_input_grad: tuple[bool, ...]):
        """Build a differentiable VJP for negation."""
        return [-grad]


negate = Neg().forward
