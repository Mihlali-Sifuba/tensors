"""CuPy implementation of in-place indexed writing."""

from __future__ import annotations
import cupy
from collections.abc import Sequence
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors._typing import Scalar


def assign_indices(
    destination: Storage,
    indices: Sequence[int],
    values: Storage | Scalar,
    *,
    source_indices: Sequence[int] | None = None,
) -> bool | None:
    """Scatter the values into the addressed positions of the device array itself.

    One indexed store per call, into the tensor's own array. Where a
    broadcast decides which source position each written position reads, the
    gather is a second indexed read on the same array rather than a host
    loop, so neither side of the write leaves the backend.
    """
    buffer = destination.buffer
    if not len(indices):
        return True
    positions = cupy.asarray(indices, dtype=cupy.intp)
    if not isinstance(values, Storage):
        buffer[positions] = values
        return True
    source = values.buffer
    if source_indices is not None:
        source = source[cupy.asarray(source_indices, dtype=cupy.intp)]
    buffer[positions] = source
    return True
