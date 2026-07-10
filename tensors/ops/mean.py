"""Mean operation."""

import builtins
from typing import List

from ..tensor import Tensor


def _mean_impl(a: Tensor) -> float:
    """Actual mean computation (returns float)."""
    if a.size == 0:
        return 0.0
    return builtins.sum(a._data) / a.size


class Mean:
    """Mean of all elements — forward and backward."""

    forward = staticmethod(_mean_impl)

    @staticmethod
    def backward(grad: Tensor, *inputs: Tensor, **kwargs: object) -> List[Tensor]:
        a = inputs[0]
        if a.size == 0:
            return [Tensor([], dtype=grad.dtype, shape=a.shape)]
        scale = 1.0 / a.size
        grad_val = float(next(iter(grad._data)))
        return [
            Tensor([grad_val * scale] * a.size, dtype=grad.dtype, shape=a.shape)
        ]
