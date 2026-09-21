"""CuPy implementation of the rectified linear unit."""

from __future__ import annotations
import math
import cupy
from typing import TYPE_CHECKING, Any
from tensors.backend.cuda.conversion import _errstate
from tensors.backend.cuda.kernels.arithmetic import ieee32
from tensors.backend.cuda.storage import CudaStorage
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def relu(
    values: Any,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Return device storage at the declared dtype.

    binary32 crosses the format boundary twice, and both crossings flush a
    subnormal on this toolchain. ``cupy.maximum`` on a native binary32 array
    was measured returning zero for the smallest *and* the largest positive
    subnormal, both of which the contract requires unchanged, and a CuPy
    conversion of a binary64 result back down would flush them again. ReLU is
    like abs and unlike sqrt here: its result can itself be subnormal, so the
    PTX conversion is needed in both directions.

    ``numpy.maximum``'s two useful properties hold for ``cupy.maximum`` as
    well, and were measured rather than assumed: NaN propagates instead of
    being sent to the zero branch, and ``-0.0`` yields canonical positive
    zero. ``float64`` already underflows gradually on the device, and an
    integer dtype computes in its own width so a large ``int64`` keeps its
    value.
    """
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        if dtype.typecode == "f":
            result = ieee32.narrow(cupy.maximum(ieee32.widen(values), 0.0))
        else:
            result = cupy.maximum(values, 0)
    storage = CudaStorage(result, dtype)
    if storage.size != math.prod(output_shape):
        raise RuntimeError("ReLU kernel returned an unexpected result size")
    return storage
