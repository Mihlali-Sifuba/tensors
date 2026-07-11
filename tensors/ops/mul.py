"""Multiplication operation."""

from typing import List, Union

from ..tensor import Tensor, _broadcast_tensors
from ..dtype import result_dtype


Scalar = Union[int, float]


class Mul:
    """Element-wise multiplication — forward and backward."""

    @staticmethod
    def forward(a: Tensor, b: Union[Tensor, Scalar]) -> Tensor:
        """Element-wise multiplication of two tensors or a tensor and a scalar."""
        dtype = result_dtype(a.dtype, b)
        if isinstance(b, (int, float)):
            data = [x * b for x in a._data]
            return Tensor(data, dtype=dtype, shape=a.shape)
        if isinstance(b, Tensor):
            a, b = _broadcast_tensors(a, b)
            data = [x * y for x, y in zip(a._data, b._data)]
            return Tensor(data, dtype=dtype, shape=a.shape)
        raise TypeError(f"Unsupported: {type(b)}")

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
