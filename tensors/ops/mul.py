"""Multiplication operation."""

from typing import List, Union

from ..tensor import Tensor
from ._utils import result_dtype


Scalar = Union[int, float]


def _mul_impl(a: Tensor, b: Union[Tensor, Scalar]) -> Tensor:
    """Actual multiplication computation."""
    dtype = result_dtype(a.dtype, b)
    if isinstance(b, (int, float)):
        data = dtype.make_array(x * b for x in a._data)
        return Tensor(data, dtype=dtype, shape=a.shape)
    if isinstance(b, Tensor):
        if a.shape != b.shape:
            raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
        data = dtype.make_array([])
        for x, y in zip(a._data, b._data):
            data.append(x * y)
        return Tensor(data, dtype=dtype, shape=a.shape)
    raise TypeError(f"Unsupported: {type(b)}")


class Mul:
    """Element-wise multiplication — forward and backward."""

    forward = staticmethod(_mul_impl)

    @staticmethod
    def backward(grad: Tensor, *inputs: Tensor, **kwargs: object) -> List[Tensor]:
        dt = grad.dtype.typecode
        sh = grad.shape
        if len(inputs) == 2:
            a, b = inputs
            da = Tensor([g * y for g, y in zip(grad._data, b._data)], dtype=dt, shape=sh)
            db = Tensor([g * x for g, x in zip(grad._data, a._data)], dtype=dt, shape=sh)
            return [da, db]
        scalar = kwargs.get("scalar", 1.0)
        assert isinstance(scalar, (int, float))
        return [Tensor([g * scalar for g in grad._data], dtype=dt, shape=sh)]
