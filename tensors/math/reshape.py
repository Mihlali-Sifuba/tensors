"""Reshape operation."""

from typing import Tuple

from ..tensor import Tensor


def reshape(tensor: Tensor, shape: Tuple[int, ...]) -> Tensor:
    """Reshape a tensor (total elements must match).

    Args:
        tensor: The tensor to reshape.
        shape: The new dimensions as a tuple.

    Returns:
        A new tensor with the same data but different shape.
    """
    total = tensor._get_total_elements()
    new_total = 1
    for dim in shape:
        new_total *= dim
    if total != new_total:
        raise ValueError(
            f"Cannot reshape tensor of size {total} to shape {shape}"
        )
    return Tensor(tensor._data, shape=shape)


__all__ = ["reshape"]
