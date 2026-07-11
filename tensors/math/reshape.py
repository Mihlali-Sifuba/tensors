"""Reshape operation."""

from typing import Tuple

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
        return Tensor(tensor._data, shape=shape)


def reshape(tensor: Tensor, shape: Tuple[int, ...]) -> Tensor:
    """Reshape a tensor (total elements must match)."""
    return Reshape.forward(tensor, shape)


__all__ = ["Reshape", "reshape"]
