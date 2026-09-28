"""Workspace reuse and parameter batching shared by the optimizers."""

from __future__ import annotations
import numpy
import threading
from collections.abc import Sequence
from typing import Any
from typing import TYPE_CHECKING
from tensors.backend.numpy.storage import NumPyStorage
from tensors.backend.storage import Storage
from tensors.backend.numpy.conversion import _storage
from tensors.shape import Shape

if TYPE_CHECKING:
    from tensors.dtype import DataType
_optimizer_workspace = threading.local()


def _optimizer_workspace_buffer(*, slot: str, size: int, dtype: Any) -> Any:
    """Return one thread-local temporary buffer for optimizer execution."""
    buffers = getattr(_optimizer_workspace, "buffers", None)
    if buffers is None:
        buffers = {}
        _optimizer_workspace.buffers = buffers
    provider_dtype = numpy.dtype(dtype)
    key = ("numpy", slot, provider_dtype.str)
    buffer = buffers.get(key)
    if buffer is None or buffer.size != size:
        buffer = numpy.empty((size,), dtype=provider_dtype)
        buffers[key] = buffer
    return buffer


def _optimizer_batch_values(values: Sequence[Any], *, slot: str) -> Any | None:
    """Pack compatible native optimizer arrays into reusable storage."""
    if not values:
        return None
    arrays = tuple(
        numpy.asarray(value).astype(numpy.float64, copy=False).reshape(-1)
        for value in values
    )
    buffer = _optimizer_workspace_buffer(
        slot=slot, size=sum(array.size for array in arrays), dtype=numpy.float64
    )
    numpy.concatenate(arrays, out=buffer)
    return buffer


def _optimizer_batch_partitions(
    dtypes: Sequence[DataType],
    shapes: Sequence[tuple[int, ...]],
    *groups: Sequence[Any],
) -> tuple[tuple[int, ...], ...] | None:
    """Group structurally compatible optimizer records by dtype."""
    if not dtypes:
        return None
    count = len(dtypes)
    if len(shapes) != count or any(len(group) != count for group in groups):
        return None
    partitions: dict[Any, list[int]] = {}
    for index, arrays in enumerate(zip(*groups)):
        shape = shapes[index]
        dtype = dtypes[index]
        native_dtype = numpy.dtype(dtype.name)
        if any(
            numpy.asarray(array).shape != shape
            or numpy.asarray(array).dtype != native_dtype
            for array in arrays
        ):
            return None
        partitions.setdefault(dtype, []).append(index)
    return tuple((tuple(indices) for indices in partitions.values()))


def _optimizer_partition(
    values: Sequence[Any], indices: tuple[int, ...]
) -> tuple[Any, ...]:
    """Select one optimizer batch partition while preserving record order."""
    return tuple((values[index] for index in indices))


def _optimizer_invalid_flag() -> Any:
    """Return a cleared scalar device flag for fused optimizer validation."""
    flag = _optimizer_workspace_buffer(slot="invalid", size=1, dtype=numpy.uint32)
    flag.fill(0)
    return flag


def _split_optimizer_storage(
    result: Any,
    dtypes: Sequence[DataType],
    shapes: Sequence[tuple[int, ...]],
) -> tuple[Storage, ...] | None:
    """Retain slices of one batched result without copying them again."""
    if not dtypes:
        return ()
    dtype = dtypes[0]
    sizes = tuple(Shape.from_iterable(shape).size for shape in shapes)
    total = sum(sizes)
    storage = _storage(result, dtype=dtype, output_shape=(total,))
    if storage is None:
        return None
    storages: list[Storage] = []
    offset = 0
    for size in sizes:
        end = offset + size
        storages.append(NumPyStorage(storage.buffer[offset:end], dtype))
        offset = end
    return tuple(storages)


def _optimizer_scalar_batch(
    values: Sequence[float], shapes: Sequence[tuple[int, ...]]
) -> Any:
    """Expand one scalar per parameter into its batched element layout."""
    if values and all((value == values[0] for value in values[1:])):
        return float(values[0])
    scalars = numpy.asarray(tuple(values), dtype=numpy.float64)
    counts = numpy.asarray(
        tuple(Shape.from_iterable(shape).size for shape in shapes), dtype=numpy.int64
    )
    return numpy.repeat(scalars, counts)
