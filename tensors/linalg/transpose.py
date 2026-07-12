"""Differentiable tensor transpose."""

from typing import Any, List

from ..tensor import Tensor
from .dot import _transpose_impl


class Transpose:
    """Transpose final matrix axes with a reverse-mode gradient rule."""

    forward = staticmethod(_transpose_impl)

    @staticmethod
    def backward(grad: Tensor, *inputs: Tensor, **kwargs: object) -> List[Tensor]:
        """Transpose the upstream gradient back to the input layout."""
        return [_transpose_impl(grad)]

    @staticmethod
    def backward_graph(grad, *inputs, **kwargs: object):
        """Build a differentiable VJP for transpose."""
        return [transpose(grad)]


def transpose(value: Any) -> Any:
    """Transpose the final two axes of a Tensor or Variable."""
    from ..variable import Variable

    if isinstance(value, Variable):
        return Variable._from_operation(
            Transpose.forward(value.data),
            "transpose",
            Transpose,
            [value],
        )
    return Transpose.forward(value if isinstance(value, Tensor) else Tensor(value))


__all__ = ["Transpose", "transpose"]
