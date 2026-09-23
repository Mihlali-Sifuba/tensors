"""CuPy implementation of the natural logarithm."""

from __future__ import annotations
import math
import cupy
from typing import TYPE_CHECKING, Any
from tensors.backend.cuda.conversion import _errstate, _narrow, _widen
from tensors.backend.cuda.storage import CudaStorage
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def log(
    values: Any,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Return device storage at the declared dtype.

    The domain is settled before this runs, in the dispatcher, which is also
    where the one host synchronisation the refusal needs is paid. Nothing
    here inspects a value, so this kernel adds no second reduction.

    Evaluated in binary64 and narrowed once. Both format crossings go
    through the PTX conversions, because a binary32 subnormal *operand*
    would otherwise arrive as zero — and zero is precisely the value the
    dispatcher just established was not there, so flushing it would turn a
    legitimate operand into one with no logarithm. ``_widen`` and
    ``_narrow`` are no-ops for any dtype that is not binary32.
    """
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        if values.dtype.kind in "iu":
            values = values.astype(cupy.float64, copy=False)
        working = _widen(values)
        result = cupy.log(working)
        narrowed = _narrow(result, cupy.dtype(dtype.name))
    storage = CudaStorage(narrowed, dtype)
    if storage.size != math.prod(output_shape):
        raise RuntimeError("Log kernel returned an unexpected result size")
    return storage
