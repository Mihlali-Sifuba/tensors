"""Matrix multiplication (dot) operation."""

from typing import List

from ..tensor import Tensor
from ._utils import result_dtype


def _dot_impl(a: Tensor, b: Tensor) -> Tensor:
    """Actual dot product computation (2D matrices only)."""
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("Dot product only supported for 2D tensors")
    if a.shape[1] != b.shape[0]:
        raise ValueError(
            f"Cannot multiply {a.shape} with {b.shape}: "
            f"inner dimensions must match"
        )

    dtype = result_dtype(a.dtype, b)
    result = dtype.make_array([])
    a_cols = a.shape[1]
    b_cols = b.shape[1]

    for i in range(a.shape[0]):
        for j in range(b_cols):
            total = 0
            for k in range(a_cols):
                total += a._data[i * a_cols + k] * b._data[k * b_cols + j]
            result.append(total)

    return Tensor(result, dtype=dtype, shape=(a.shape[0], b.shape[1]))


def _transpose_impl(t: Tensor) -> Tensor:
    """Transpose a 2D tensor (helper for backward)."""
    if t.ndim != 2:
        raise ValueError("Transpose only supported for 2D tensors")
    new_data = t.dtype.make_array([])
    for j in range(t.shape[1]):
        for i in range(t.shape[0]):
            new_data.append(t._data[i * t.shape[1] + j])
    return Tensor(new_data, dtype=t.dtype, shape=(t.shape[1], t.shape[0]))


class Dot:
    """Matrix multiplication for 2D tensors — forward and backward."""

    forward = staticmethod(_dot_impl)

    @staticmethod
    def backward(grad: Tensor, *inputs: Tensor, **kwargs: object) -> List[Tensor]:
        a, b = inputs
        og = grad
        if og.ndim == 1:
            og = Tensor(og._data, shape=(1, og.shape[0]))
        da = _dot_impl(og, _transpose_impl(b))
        db = _dot_impl(_transpose_impl(a), og)
        return [da, db]
