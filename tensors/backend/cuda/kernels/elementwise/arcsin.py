"""CuPy implementation of arcsine."""

from __future__ import annotations
import cupy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.cuda.conversion import _errstate
from tensors.backend.cuda.conversion import _storage
from tensors.backend.cuda.conversion import _view

if TYPE_CHECKING:
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


def arcsin(value: Tensor, *, dtype: DataType) -> Storage | None:
    """Run an elementwise unary kernel while preserving public domains."""
    if dtype.kind == "integer":
        return None
    try:
        values = _view(value).astype(cupy.float64, copy=False)
    except (TypeError, ValueError):
        return None
    if bool(cupy.any((values < -1.0) | (values > 1.0))):
        raise ValueError(f"{'arcsin'} is only defined for values between -1 and 1")
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
        result = functions["arcsin"](values)
    return _storage(result, dtype=dtype, output_shape=value.shape)
