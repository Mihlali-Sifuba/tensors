"""Negation operation."""

from tensors.backend import execute_negate
from tensors.operations.base import Operation
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

    def backward(self, grad, *inputs, needs_input_grad: tuple[bool, ...]):
        """Negate the upstream gradient.

        The derivative of ``-a`` with respect to ``a`` is minus one, so the
        VJP is the upstream gradient negated. Written as the negation
        operator rather than as a call on a Tensor, the operands decide what
        it means: a Tensor negates now, and a Variable records a negation
        that can be differentiated again.
        """
        return [-grad]


negate = Neg().forward
