"""Reshape operation."""

from typing import Any, List, Tuple, overload

from .._typing import TensorData, TensorLike, TensorResult, TensorValue
from ..shape import Shape
from ..ops.operation import Operation
from ..tensor import Tensor


class Reshape(Operation):
    """Reshape operation."""

    __slots__ = ("shape",)
    name = "reshape"

    def __init__(self, *, shape: Tuple[int, ...]) -> None:
        object.__setattr__(self, "shape", tuple(shape))

    def forward(self, tensor: Tensor) -> Tensor:
        shape = self.shape
        current_element_count = tensor.shape.size
        requested_element_count = Shape.from_iterable(shape).size
        if current_element_count != requested_element_count:
            raise ValueError(
                f"Cannot reshape tensor of size {current_element_count} "
                f"to shape {shape}"
            )
        return Tensor(tensor._data, dtype=tensor.dtype, shape=shape)

    def backward(
        self,
        grad: Tensor,
        *inputs: Tensor,
        needs_input_grad: tuple[bool, ...],
    ) -> List[Tensor]:
        """Restore the input shape without changing gradient values."""
        return [Reshape(shape=inputs[0].shape).forward(grad)]

    def backward_graph(
        self,
        grad,
        *inputs,
        needs_input_grad: tuple[bool, ...],
    ):
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
        operation = Reshape(shape=shape)
        return Variable._from_operation(
            operation.forward(tensor.data),
            operation,
            (tensor,),
        )
    if not isinstance(tensor, Tensor):
        tensor = Tensor(tensor)
    return Reshape(shape=shape).forward(tensor)


__all__ = ["Reshape", "reshape"]
