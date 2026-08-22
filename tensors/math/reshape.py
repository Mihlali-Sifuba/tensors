"""Reshape operation."""

from typing import Any, List, Tuple, overload

from .._typing import TensorData, TensorLike, TensorResult, TensorValue
from ..tensor import Tensor
from ..utils.shape import shape_size


class Reshape:
    """Reshape operation."""

    @staticmethod
    def forward(tensor: Tensor, shape: Tuple[int, ...]) -> Tensor:
        current_element_count = shape_size(tensor.shape)
        requested_element_count = shape_size(shape)
        if current_element_count != requested_element_count:
            raise ValueError(
                f"Cannot reshape tensor of size {current_element_count} "
                f"to shape {shape}"
            )
        return Tensor(tensor._data, dtype=tensor.dtype, shape=shape)

    @staticmethod
    def backward(grad: Tensor, *inputs: Tensor, **kwargs: object) -> List[Tensor]:
        """Restore the input shape without changing gradient values."""
        return [Reshape.forward(grad, inputs[0].shape)]

    @staticmethod
    def backward_graph(grad, *inputs, **kwargs: object):
        """Build a differentiable reshape VJP."""
        return [reshape(grad, inputs[0].shape)]


@overload
def reshape(tensor: TensorValue, shape: tuple[int, ...]) -> TensorValue: ...


@overload
def reshape(tensor: TensorData, shape: tuple[int, ...]) -> Tensor: ...


def reshape(tensor: TensorLike, shape: tuple[int, ...]) -> TensorResult:
    """Reshape a Tensor or Variable without changing its values."""
    from ..variable import Variable

    shape = tuple(shape)
    if isinstance(tensor, Variable):
        return Variable._from_operation(
            Reshape.forward(tensor.data, shape),
            "reshape",
            Reshape,
            [tensor],
            shape=shape,
        )
    if not isinstance(tensor, Tensor):
        tensor = Tensor(tensor)
    return Reshape.forward(tensor, shape)


__all__ = ["Reshape", "reshape"]
