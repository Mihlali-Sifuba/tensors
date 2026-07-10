"""Sum operation."""

import builtins
from typing import List

from ..tensor import Tensor


def _sum_impl(a: Tensor) -> float:
    """Actual sum computation (returns float)."""
    return builtins.sum(a._data)


class Sum:
    """Sum of all elements — forward and backward."""

    forward = staticmethod(_sum_impl)

    @staticmethod
    def backward(grad: Tensor, *inputs: Tensor, **kwargs: object) -> List[Tensor]:
        a = inputs[0]
        grad_value = float(next(iter(grad._data)))
        return [
            Tensor([grad_value] * a.size, dtype=grad.dtype, shape=a.shape)
        ]
