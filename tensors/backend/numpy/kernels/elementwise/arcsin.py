"""NumPy implementation of arcsine."""

from __future__ import annotations
import numpy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.conversion import _storage
from tensors.backend.numpy.conversion import tensor_to_logical_array

if TYPE_CHECKING:
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


def arcsin(value: Tensor, *, dtype: DataType) -> Storage | None:
    """Run an elementwise unary kernel while preserving public domains."""
    if dtype.kind == "integer":
        return None
    try:
        values = tensor_to_logical_array(value).astype(numpy.float64, copy=False)
    except (TypeError, ValueError):
        return None
    if bool(numpy.any((values < -1.0) | (values > 1.0))):
        raise ValueError(f"{'arcsin'} is only defined for values between -1 and 1")
    functions = {
        "abs": numpy.abs,
        "sqrt": numpy.sqrt,
        "exp": numpy.exp,
        "log": numpy.log,
        "sin": numpy.sin,
        "cos": numpy.cos,
        "tan": numpy.tan,
        "arcsin": numpy.arcsin,
        "arccos": numpy.arccos,
        "arctan": numpy.arctan,
        "sinh": numpy.sinh,
        "cosh": numpy.cosh,
        "arcsinh": numpy.arcsinh,
        "arccosh": numpy.arccosh,
        "arctanh": numpy.arctanh,
        "sign": numpy.sign,
        "tanh": numpy.tanh,
    }
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        result = functions["arcsin"](values)
    return _storage(result, dtype=dtype, output_shape=value.shape)
