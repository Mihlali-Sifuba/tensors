"""Addition operation."""

from typing import List, Union

from ..backend import execute_binary
from ..dtype import result_dtype
from ..tensor import Tensor
from ..utils.broadcasting import broadcast_shape, broadcast_tensors
from ._utils import sum_to_shape


Scalar = Union[int, float]


class Add:
    """Element-wise addition — forward and backward."""

    @staticmethod
    def forward(a: Tensor, b: Union[Tensor, Scalar]) -> Tensor:
        """Element-wise addition of two tensors or a tensor and a scalar."""
        if not isinstance(b, (int, float, Tensor)):
            raise TypeError(f"Unsupported: {type(b)}")
        dtype = result_dtype(a.dtype, b)
        output_shape = (
            broadcast_shape(a.shape, b.shape)
            if isinstance(b, Tensor)
            else a.shape
        )
        accelerated = execute_binary(
            "add",
            a,
            b,
            dtype=dtype,
            output_shape=output_shape,
        )
        if accelerated is not None:
            return Tensor(accelerated, dtype=dtype, shape=output_shape)
        if isinstance(b, (int, float)):
            data = [x + b for x in a._data]
            return Tensor(data, dtype=dtype, shape=a.shape)
        if isinstance(b, Tensor):
            a, b = broadcast_tensors(a, b)
            data = [x + y for x, y in zip(a._data, b._data)]
            return Tensor(data, dtype=dtype, shape=a.shape)
        raise TypeError(f"Unsupported: {type(b)}")

    @staticmethod
    def backward(grad: Tensor, *inputs: Tensor, **kwargs: object) -> List[Tensor]:
        if len(inputs) == 1:
            return [grad]
        left, right = inputs
        return [sum_to_shape(grad, left.shape), sum_to_shape(grad, right.shape)]

    @staticmethod
    def backward_graph(grad, *inputs, **kwargs: object):
        """Build a differentiable VJP for addition."""
        if len(inputs) == 1:
            return [grad]
        left, right = inputs
        from ._utils import sum_to_shape_graph
        return [
            sum_to_shape_graph(grad, left.shape),
            sum_to_shape_graph(grad, right.shape),
        ]
