"""Negation operation."""

from typing import List

from ..tensor import Tensor
from ._utils import negation_dtype


def _neg_impl(a: Tensor) -> Tensor:
    """Actual negation computation."""
    dtype = negation_dtype(a.dtype)
    data = [-x for x in a._data]
    return Tensor(data, dtype=dtype, shape=a.shape)


class Neg:
    """Negation — forward and backward."""

    forward = staticmethod(_neg_impl)

    @staticmethod
    def backward(grad: Tensor, *inputs: Tensor, **kwargs: object) -> List[Tensor]:
        neg = Tensor([-x for x in grad._data], dtype=grad.dtype.typecode, shape=grad.shape)
        return [neg]
