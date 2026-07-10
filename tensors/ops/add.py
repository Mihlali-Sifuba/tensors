"""Addition operation."""

from typing import List, Union

from ..tensor import Tensor
from ._utils import result_dtype


Scalar = Union[int, float]


def _add_impl(a: Tensor, b: Union[Tensor, Scalar]) -> Tensor:
    """Actual addition computation."""
    dtype = result_dtype(a.dtype, b)
    if isinstance(b, (int, float)):
        data = dtype.make_array(x + b for x in a._data)
        return Tensor(data, dtype=dtype, shape=a.shape)
    if isinstance(b, Tensor):
        if a.shape != b.shape:
            raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
        data = dtype.make_array([])
        for x, y in zip(a._data, b._data):
            data.append(x + y)
        return Tensor(data, dtype=dtype, shape=a.shape)
    raise TypeError(f"Unsupported: {type(b)}")


class Add:
    """Element-wise addition — forward and backward."""

    forward = staticmethod(_add_impl)

    @staticmethod
    def backward(grad: Tensor, *inputs: Tensor, **kwargs: object) -> List[Tensor]:
        return [grad] * len(inputs)
