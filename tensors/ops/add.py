"""Addition operation."""

from typing import List, Union

from ..tensor import Tensor
from ..dtype import result_dtype


Scalar = Union[int, float]


class Add:
    """Element-wise addition — forward and backward."""

    @staticmethod
    def forward(a: Tensor, b: Union[Tensor, Scalar]) -> Tensor:
        """Element-wise addition of two tensors or a tensor and a scalar."""
        dtype = result_dtype(a.dtype, b)
        if isinstance(b, (int, float)):
            data = [x + b for x in a._data]
            return Tensor(data, dtype=dtype, shape=a.shape)
        if isinstance(b, Tensor):
            if a.shape != b.shape:
                raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
            data = [x + y for x, y in zip(a._data, b._data)]
            return Tensor(data, dtype=dtype, shape=a.shape)
        raise TypeError(f"Unsupported: {type(b)}")

    @staticmethod
    def backward(grad: Tensor, *inputs: Tensor, **kwargs: object) -> List[Tensor]:
        return [grad] * len(inputs)
