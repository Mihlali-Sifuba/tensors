"""Reshape operation."""

from typing import Any, List, Tuple

from ..tensor import Tensor


class Reshape:
    """Reshape operation."""

    @staticmethod
    def forward(tensor: Tensor, shape: Tuple[int, ...]) -> Tensor:
        total = tensor._get_total_elements()
        new_total = 1
        for dim in shape:
            new_total *= dim
        if total != new_total:
            raise ValueError(
                f"Cannot reshape tensor of size {total} to shape {shape}"
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


def reshape(tensor: Any, shape: Tuple[int, ...]) -> Any:
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
