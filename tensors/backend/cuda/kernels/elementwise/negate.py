"""CuPy implementation of negation."""

from __future__ import annotations
import cupy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.cuda.conversion import _arithmetic_storage
from tensors.backend.cuda.conversion import _errstate
from tensors.backend.cuda.conversion import _operand
from tensors.backend.cuda.conversion import tensor_to_logical_array
from tensors.backend.cuda.conversion import _storage

if TYPE_CHECKING:
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


def negate(value: Tensor, *, dtype: DataType) -> Storage | None:
    """Run elementwise negation on the device.

    An integer does not go through ``_operand``. That helper refuses integer
    dtypes because the Python reference computes integer arithmetic in
    arbitrary-precision Python integers that no fixed-width device type can
    match — a reason that does not reach negation, which needs no
    intermediate precision at all: flipping a sign is exact in the operand's
    own width. Section 10.1 rule B12 asks for integer arithmetic to execute
    natively here, and this is one operation where nothing stands in the way.

    The declared output width is not always the operand's, because
    ``negation_dtype`` widens an unsigned dtype so every negated value is
    representable. The result is retained as arithmetic, so a signed
    minimum — the one integer overflow negation can produce — wraps under
    rule B1 rather than declining.
    """
    if dtype.kind == "integer":
        native = cupy.dtype(dtype.name)
        operand = tensor_to_logical_array(value).astype(native, copy=False)
        with _errstate(over="ignore"):
            result = cupy.negative(operand)
        return _arithmetic_storage(result, dtype=dtype, output_shape=value.shape)
    try:
        operand = _operand(value, dtype)
    except (TypeError, ValueError):
        return None
    result = cupy.negative(operand)
    return _storage(result, dtype=dtype, output_shape=value.shape)
