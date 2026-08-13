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
        axes = kwargs.get("axes")
        if axes is None:
            return [_transpose_impl(grad)]
        normalized = tuple(axis + grad.ndim if axis < 0 else axis for axis in axes)
        inverse = tuple(normalized.index(axis) for axis in range(grad.ndim))
        return [_transpose_impl(grad, inverse)]

    @staticmethod
    def backward_graph(grad, *inputs, **kwargs: object):
        """Build a differentiable VJP for transpose."""
        axes = kwargs.get("axes")
        if axes is None:
            return [transpose(grad)]
        normalized = tuple(axis + grad.ndim if axis < 0 else axis for axis in axes)
        inverse = tuple(normalized.index(axis) for axis in range(grad.ndim))
        return [transpose(grad, axes=inverse)]


def transpose(
    value: Any,
    axes: tuple[int, ...] | list[int] | None = None,
) -> Any:
    """Permute axes, or transpose the final two matrix axes by default."""
    from ..variable import Variable

    if isinstance(value, Variable):
        return Variable._from_operation(
            Transpose.forward(value.data, axes=axes),
            "transpose",
            Transpose,
            [value],
            axes=axes,
        )
    return Transpose.forward(
        value if isinstance(value, Tensor) else Tensor(value),
        axes=axes,
    )


__all__ = ["Transpose", "transpose"]
