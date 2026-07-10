"""Division operation."""

from typing import List, Union

from ..tensor import Tensor
from ._utils import result_dtype


Scalar = Union[int, float]


def _div_impl(a: Tensor, b: Union[Tensor, Scalar]) -> Tensor:
    """Actual division computation."""
    dtype = result_dtype(a.dtype, b, division=True)
    if isinstance(b, (int, float)):
        if b == 0:
            raise ZeroDivisionError("Division by zero")
        data = dtype.make_array(x / b for x in a._data)
        return Tensor(data, dtype=dtype, shape=a.shape)
    if isinstance(b, Tensor):
        if a.shape != b.shape:
            raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
        data = dtype.make_array([])
        for x, y in zip(a._data, b._data):
            if y == 0:
                raise ZeroDivisionError("Division by zero")
            data.append(x / y)
        return Tensor(data, dtype=dtype, shape=a.shape)
    raise TypeError(f"Unsupported: {type(b)}")


def _mul_tensors(a: Tensor, b: Tensor) -> Tensor:
    """Element-wise multiply two Tensors (helper for backward)."""
    data = a.dtype.make_array(x * y for x, y in zip(a._data, b._data))
    return Tensor(data, dtype=a.dtype, shape=a.shape)


def _neg_tensor(t: Tensor) -> Tensor:
    """Negate a Tensor (helper for backward)."""
    return Tensor([-x for x in t._data], dtype=t.dtype.typecode, shape=t.shape)


class Div:
    """Element-wise division — forward and backward."""

    forward = staticmethod(_div_impl)

    @staticmethod
    def forward_reverse(a: Tensor, scalar: Scalar) -> Tensor:
        numerator = Tensor([scalar] * a.size, shape=a.shape)
        return _div_impl(numerator, a)

    @staticmethod
    def backward(grad: Tensor, *inputs: Tensor, **kwargs: object) -> List[Tensor]:
        if len(inputs) == 2:
            a, b = inputs
            # da = grad / b
            da = _div_impl(grad, b)
            # db = -(grad * a) / (b * b)
            num = _mul_tensors(grad, a)
            den = _mul_tensors(b, b)
            db = _neg_tensor(_div_impl(num, den))
            return [da, db]
        scalar = kwargs.get("scalar", 1.0)
        assert isinstance(scalar, (int, float))
        if kwargs.get("reverse", False):
            a = inputs[0]
            dtype = result_dtype(grad.dtype, a, division=True)
            values = (
                -g * scalar / (x * x) for g, x in zip(grad._data, a._data)
            )
            return [Tensor(dtype.make_array(values), dtype=dtype, shape=a.shape)]
        return [Tensor([g / scalar for g in grad._data], dtype=grad.dtype.typecode, shape=grad.shape)]
