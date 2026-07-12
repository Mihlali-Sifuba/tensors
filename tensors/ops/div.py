"""Division operation."""

from typing import List, Union

from ..tensor import Tensor, _broadcast_tensors
from ..dtype import result_dtype
from ._utils import unbroadcast


Scalar = Union[int, float]


class Div:
    """Element-wise division — forward and backward."""

    @staticmethod
    def forward(a: Tensor, b: Union[Tensor, Scalar]) -> Tensor:
        """Element-wise division."""
        dtype = result_dtype(a.dtype, b, division=True)
        if isinstance(b, (int, float)):
            if b == 0:
                raise ZeroDivisionError("Division by zero")
            data = [x / b for x in a._data]
            return Tensor(data, dtype=dtype, shape=a.shape)
        if isinstance(b, Tensor):
            a, b = _broadcast_tensors(a, b)
            data = []
            for x, y in zip(a._data, b._data):
                if y == 0:
                    raise ZeroDivisionError("Division by zero")
                data.append(x / y)
            return Tensor(data, dtype=dtype, shape=a.shape)
        raise TypeError(f"Unsupported: {type(b)}")

    @staticmethod
    def _mul_tensors(a: Tensor, b: Tensor) -> Tensor:
        """Element-wise multiply two Tensors (helper for backward)."""
        data = [x * y for x, y in zip(a._data, b._data)]
        return Tensor(data, dtype=a.dtype, shape=a.shape)

    @staticmethod
    def _neg_tensor(t: Tensor) -> Tensor:
        """Negate a Tensor (helper for backward)."""
        return Tensor([-x for x in t._data], dtype=t.dtype.typecode, shape=t.shape)

    @staticmethod
    def forward_reverse(a: Tensor, scalar: Scalar) -> Tensor:
        """Forward for scalar / tensor (reverse division)."""
        numerator = Tensor([scalar] * a.size, shape=a.shape)
        return Div.forward(numerator, a)

    @staticmethod
    def backward(grad: Tensor, *inputs: Tensor, **kwargs: object) -> List[Tensor]:
        if len(inputs) == 2:
            a, b = inputs
            expanded_a, expanded_b = _broadcast_tensors(a, b)
            da = Tensor(
                [g / y for g, y in zip(grad._data, expanded_b._data)],
                dtype=grad.dtype,
                shape=grad.shape,
            )
            db = Tensor(
                [
                    -g * x / (y * y)
                    for g, x, y in zip(
                        grad._data,
                        expanded_a._data,
                        expanded_b._data,
                    )
                ],
                dtype=grad.dtype,
                shape=grad.shape,
            )
            return [unbroadcast(da, a.shape), unbroadcast(db, b.shape)]
        scalar = kwargs.get("scalar", 1.0)
        assert isinstance(scalar, (int, float))
        if kwargs.get("reverse", False):
            a = inputs[0]
            dtype = result_dtype(grad.dtype, a, division=True)
            values = (
                -g * scalar / (x * x) for g, x in zip(grad._data, a._data)
            )
            return [Tensor(list(values), dtype=dtype, shape=a.shape)]
        return [Tensor([g / scalar for g in grad._data], dtype=grad.dtype.typecode, shape=grad.shape)]
