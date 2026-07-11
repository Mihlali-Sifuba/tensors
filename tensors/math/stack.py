"""Stack operation."""

from array import array
from typing import List, Union

from ..tensor import Tensor


def stack(tensors: List[Union[Tensor, List]], axis: int = 0) -> Tensor:
    """Stack a sequence of tensors along a new axis.

    Each tensor must have the same shape.  A new dimension of size
    ``len(tensors)`` is inserted at *axis*.

    Args:
        tensors: Sequence of tensors to stack.
        axis: Position of the new dimension.

    Returns:
        A new tensor with ``ndim + 1`` dimensions.

    Example::

        a = ts.Tensor([1, 2, 3])
        b = ts.Tensor([4, 5, 6])
        ts.stack([a, b])        # shape (2, 3)
        ts.stack([a, b], 1)     # shape (3, 2)
    """
    if not tensors:
        raise ValueError("stack requires at least one tensor")

    converted = []
    for t in tensors:
        if isinstance(t, Tensor):
            converted.append(t)
        else:
            converted.append(Tensor(t))

    elem_shape = converted[0].shape
    n = len(converted)
    for t in converted[1:]:
        if t.shape != elem_shape:
            raise ValueError(
                f"All tensors must have the same shape; got {elem_shape} and {t.shape}"
            )

    if axis < 0:
        axis += len(elem_shape) + 1
    if not (0 <= axis <= len(elem_shape)):
        raise ValueError(
            f"Axis {axis} out of bounds for {len(elem_shape)}D tensor stack"
        )

    out_shape = list(elem_shape)
    out_shape.insert(axis, n)

    # Elements before the stacked axis (outer groups)
    before = 1
    for d in elem_shape[:axis]:
        before *= d

    # Elements in one source tensor per group at the axis position
    axis_stride = 1
    for d in elem_shape[axis:]:
        axis_stride *= d

    dtype = converted[0].dtype
    result = array(dtype.typecode, [])
    for g in range(before):
        base = g * axis_stride
        for k in range(n):
            for t in range(axis_stride):
                result.append(converted[k]._data[base + t])

    return Tensor(result, dtype=dtype, shape=tuple(out_shape))


__all__ = ["stack"]
