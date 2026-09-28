"""NumPy implementation of negation."""

from __future__ import annotations
import numpy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.numpy.conversion import _arithmetic_storage
from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.conversion import _operand
from tensors.backend.numpy.conversion import tensor_to_logical_array
from tensors.backend.numpy.conversion import _storage

if TYPE_CHECKING:
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


def negate(value: Tensor, *, dtype: DataType) -> Storage | None:
    """Run elementwise NumPy negation.

    An integer negates in the declared output width, which is not always the
    operand's: ``negation_dtype`` widens an unsigned dtype so that every
    negated value is representable. The result is retained as arithmetic, so
    the one integer overflow negation can produce — a signed dtype's minimum,
    which has no positive counterpart — wraps under section 10.1 rule B1
    rather than declining.
    """
    if dtype.kind == "integer":
        native = numpy.dtype(dtype.name)
        operand = tensor_to_logical_array(value).astype(native, copy=False)
        with _errstate(over="ignore"):
            result = numpy.negative(operand)
        return _arithmetic_storage(result, dtype=dtype, output_shape=value.shape)
    try:
        operand = _operand(value, dtype)
    except (TypeError, ValueError):
        return None
    result = numpy.negative(operand)
    return _storage(result, dtype=dtype, output_shape=value.shape)
