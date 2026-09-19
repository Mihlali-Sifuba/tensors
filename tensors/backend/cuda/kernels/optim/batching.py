"""Workspace reuse and parameter batching shared by the optimizers."""

from __future__ import annotations
import cupy
import threading
from collections.abc import Sequence
from typing import Any
from typing import TYPE_CHECKING
from tensors.backend.cuda.storage import CudaStorage
from tensors.backend.storage import Storage
from tensors.backend.cuda.conversion import _storage
from tensors.backend.cuda.conversion import _working_values

if TYPE_CHECKING:
    from tensors.tensor import Tensor
_optimizer_workspace = threading.local()


def _optimizer_workspace_buffer(*, slot: str, size: int, dtype: Any) -> Any:
    """Return one thread-local temporary buffer for optimizer execution."""
    buffers = getattr(_optimizer_workspace, "buffers", None)
    if buffers is None:
        buffers = {}
        _optimizer_workspace.buffers = buffers
    provider_dtype = cupy.dtype(dtype)
    key = ("cuda", slot, provider_dtype.str)
    buffer = buffers.get(key)
    if buffer is None or buffer.size != size:
        buffer = cupy.empty((size,), dtype=provider_dtype)
        buffers[key] = buffer
    return buffer


def _optimizer_batch_values(tensors: Sequence[Tensor], *, slot: str) -> Any | None:
    """Pack compatible optimizer tensors into reusable native storage."""
    if not tensors:
        return None
    dtype = tensors[0].dtype
    if any((tensor.dtype != dtype for tensor in tensors)):
        return None
    arrays = tuple((_working_values(tensor).reshape(-1) for tensor in tensors))
    buffer = _optimizer_workspace_buffer(
        slot=slot, size=sum((tensor.size for tensor in tensors)), dtype=cupy.float64
    )
    cupy.concatenate(arrays, out=buffer)
    return buffer


def _optimizer_batch_partitions(
    *groups: Sequence[Tensor],
) -> tuple[tuple[int, ...], ...] | None:
    """Group structurally compatible optimizer records by dtype."""
    if not groups or not groups[0]:
        return None
    count = len(groups[0])
    if any((len(group) != count for group in groups)):
        return None
    partitions: dict[Any, list[int]] = {}
    for index, tensors in enumerate(zip(*groups)):
        shape = tensors[0].shape
        dtype = tensors[0].dtype
        if any((tensor.shape != shape or tensor.dtype != dtype for tensor in tensors)):
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
    flag = _optimizer_workspace_buffer(slot="invalid", size=1, dtype=cupy.uint32)
    flag.fill(0)
    return flag


def _split_optimizer_storage(
    result: Any, references: Sequence[Tensor]
) -> tuple[Storage, ...] | None:
    """Retain slices of one batched result without copying them again."""
    if not references:
        return ()
    dtype = references[0].dtype
    total = sum((reference.size for reference in references))
    storage = _storage(result, dtype=dtype, output_shape=(total,))
    if storage is None:
        return None
    storages: list[Storage] = []
    offset = 0
    for reference in references:
        end = offset + reference.size
        storages.append(CudaStorage(storage.buffer[offset:end], dtype))
        offset = end
    return tuple(storages)


def _optimizer_scalar_batch(
    values: Sequence[float], references: Sequence[Tensor]
) -> Any:
    """Expand one scalar per parameter into its batched element layout."""
    if values and all((value == values[0] for value in values[1:])):
        return float(values[0])
    scalars = cupy.asarray(tuple(values), dtype=cupy.float64)
    counts = cupy.asarray(
        tuple((reference.size for reference in references)), dtype=cupy.int64
    )
    return cupy.repeat(scalars, counts)
