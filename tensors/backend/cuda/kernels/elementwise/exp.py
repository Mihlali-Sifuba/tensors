"""CuPy implementation of the exponential."""

from __future__ import annotations
import cupy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.cuda.conversion import _errstate
from tensors.backend.cuda.conversion import _storage
from tensors.backend.cuda.conversion import _working_values

if TYPE_CHECKING:
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


def exp(value: Tensor, *, dtype: DataType) -> Storage | None:
    """Run an elementwise unary kernel while preserving public domains."""
    if dtype.kind == "integer":
        return None
    try:
        values = _working_values(value)
    except (TypeError, ValueError):
        return None
    functions = {
        "abs": cupy.abs,
        "sqrt": cupy.sqrt,
        "exp": cupy.exp,
        "log": cupy.log,
        "sin": cupy.sin,
        "cos": cupy.cos,
        "tan": cupy.tan,
        "arcsin": cupy.arcsin,
        "arccos": cupy.arccos,
        "arctan": cupy.arctan,
        "sinh": cupy.sinh,
        "cosh": cupy.cosh,
        "arcsinh": cupy.arcsinh,
        "arccosh": cupy.arccosh,
        "arctanh": cupy.arctanh,
        "sign": cupy.sign,
        "tanh": cupy.tanh,
    }
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        result = functions["exp"](values)
    return _storage(result, dtype=dtype, output_shape=value.shape)
