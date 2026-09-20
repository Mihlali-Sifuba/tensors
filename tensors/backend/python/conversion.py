"""The boundary between tensors and the Python values its kernels operate on.

A kernel in this backend receives values, not tensors. Turning an operand into
those values — reading a Tensor's logical elements, expanding a broadcast
operand to the agreed output shape, and pairing a scalar against the other
operand — happens here, once, rather than in each kernel.

The Python backend has no array library to broadcast on its behalf, so the
expansion is explicit. A scalar is paired lazily with
:func:`itertools.repeat`, which costs nothing per element: the tensor operand
bounds the pairing, exactly as the per-kernel scalar branches used to.
"""

from __future__ import annotations

from collections.abc import Iterable
from itertools import repeat
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tensors._typing import Scalar
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


def prepare_binary_operands(
    left: Tensor | Scalar,
    right: Tensor | Scalar,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> tuple[Iterable[Any], Iterable[Any]]:
    """Return two operands already paired for elementwise evaluation.

    The result is a pair of iterables a kernel can ``zip`` straight through.
    Tensor operands are expanded to ``output_shape``; a scalar is repeated
    against the tensor beside it and never materialized.

    ``dtype`` is the declared *result* dtype and does not convert the operands
    here: Python integers are exact, so a kernel evaluates in Python's own
    arithmetic and the declared width is applied once, by
    :meth:`PythonStorage.from_arithmetic`, when the result is stored. The
    parameter is part of the shared preparation contract, which the array
    backends do use to place their operands in the declared dtype.
    """
    from tensors.tensor import Tensor
    from tensors.utils.broadcasting import broadcast_to

    left_is_tensor = isinstance(left, Tensor)
    right_is_tensor = isinstance(right, Tensor)

    if left_is_tensor and right_is_tensor:
        return (
            broadcast_to(left, output_shape)._data,
            broadcast_to(right, output_shape)._data,
        )
    if left_is_tensor:
        return left._data, repeat(right)
    if right_is_tensor:
        return repeat(left), right._data
    # Two scalars: one pair, so the operation still evaluates exactly once.
    return (left,), (right,)


__all__ = ["prepare_binary_operands"]
