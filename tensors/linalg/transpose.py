"""Differentiable tensor transpose."""

from typing import Any, List, overload

from .._typing import TensorData, TensorLike, TensorResult, TensorValue
from ..ops.operation import Operation
from ..tensor import Tensor
from .dot import _transpose_impl


class Transpose(Operation):
    """Transpose final matrix axes with a reverse-mode gradient rule."""

    __slots__ = ("axes",)
    name = "transpose"

    def __init__(self, *, axes: tuple[int, ...] | None = None) -> None:
        object.__setattr__(self, "axes", axes)

    def forward(self, value: Tensor) -> Tensor:
        """Permute the tensor axes described by this invocation."""
        return _transpose_impl(value, self.axes)

    def backward(
        self,
        grad: Tensor,
        *inputs: Tensor,
        needs_input_grad: tuple[bool, ...],
    ) -> List[Tensor]:
        """Transpose the upstream gradient back to the input layout."""
        axes = self.axes
        if axes is None:
            return [_transpose_impl(grad)]
        normalized = tuple(axis + grad.ndim if axis < 0 else axis for axis in axes)
        inverse = tuple(normalized.index(axis) for axis in range(grad.ndim))
        return [_transpose_impl(grad, inverse)]

    def backward_graph(
        self,
        grad,
        *inputs,
        needs_input_grad: tuple[bool, ...],
    ):
        """Build a differentiable VJP for transpose."""
        axes = self.axes
        if axes is None:
            return [transpose(grad)]
        normalized = tuple(axis + grad.ndim if axis < 0 else axis for axis in axes)
        inverse = tuple(normalized.index(axis) for axis in range(grad.ndim))
        return [transpose(grad, axes=inverse)]


@overload
def transpose(
    value: TensorValue,
    axes: tuple[int, ...] | list[int] | None = None,
) -> TensorValue: ...


@overload
def transpose(
    value: TensorData,
    axes: tuple[int, ...] | list[int] | None = None,
) -> Tensor: ...


def transpose(
    value: TensorLike,
    axes: tuple[int, ...] | list[int] | None = None,
) -> TensorResult:
    """Permute axes, or transpose the final two matrix axes by default."""
    from ..variable import Variable

    axes = tuple(axes) if isinstance(axes, list) else axes

    if isinstance(value, Variable):
        operation = Transpose(axes=axes)
        return Variable._from_operation(
            operation.forward(value.data),
            operation,
            (value,),
        )
    return Transpose(axes=axes).forward(
        value if isinstance(value, Tensor) else Tensor(value)
    )


__all__ = ["Transpose", "transpose"]
