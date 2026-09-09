"""Workspace reuse and parameter batching shared by the optimizers."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any, TYPE_CHECKING

from ...storage import CudaStorage, NumPyStorage, Storage
from ..core import _array_kind, _storage, _view

if TYPE_CHECKING:
    from ....tensor import Tensor

_optimizer_workspace = threading.local()

def _optimizer_workspace_buffer(
    numpy: Any,
    *,
    slot: str,
    size: int,
    dtype: Any,
) -> Any:
    """Return one thread-local temporary buffer for optimizer execution."""
    buffers = getattr(_optimizer_workspace, "buffers", None)
    if buffers is None:
        buffers = {}
        _optimizer_workspace.buffers = buffers
    provider_dtype = numpy.dtype(dtype)
    key = (_array_kind(numpy), slot, provider_dtype.str)
    buffer = buffers.get(key)
    if buffer is None or buffer.size != size:
        buffer = numpy.empty((size,), dtype=provider_dtype)
        buffers[key] = buffer
    return buffer

def _optimizer_batch_values(
    tensors: Sequence[Tensor],
    numpy: Any,
    *,
    slot: str,
) -> Any | None:
    """Pack compatible optimizer tensors into reusable native storage."""
    if not tensors:
        return None
    dtype = tensors[0].dtype
    if any(tensor.dtype != dtype for tensor in tensors):
        return None
    arrays = tuple(
        _view(tensor, numpy).astype(numpy.float64, copy=False).reshape(-1)
        for tensor in tensors
    )
    buffer = _optimizer_workspace_buffer(
        numpy,
        slot=slot,
        size=sum(tensor.size for tensor in tensors),
        dtype=numpy.float64,
    )
    numpy.concatenate(arrays, out=buffer)
    return buffer

def _optimizer_batch_partitions(
    *groups: Sequence[Tensor],
) -> tuple[tuple[int, ...], ...] | None:
    """Group structurally compatible optimizer records by dtype."""
    if not groups or not groups[0]:
        return None
    count = len(groups[0])
    if any(len(group) != count for group in groups):
        return None
    partitions: dict[Any, list[int]] = {}
    for index, tensors in enumerate(zip(*groups)):
        shape = tensors[0].shape
        dtype = tensors[0].dtype
        if any(
            tensor.shape != shape or tensor.dtype != dtype
            for tensor in tensors
        ):
            return None
        partitions.setdefault(dtype, []).append(index)
    return tuple(tuple(indices) for indices in partitions.values())

def _optimizer_partition(
    values: Sequence[Any],
    indices: tuple[int, ...],
) -> tuple[Any, ...]:
    """Select one optimizer batch partition while preserving record order."""
    return tuple(values[index] for index in indices)

def _optimizer_invalid_flag(numpy: Any) -> Any:
    """Return a cleared scalar device flag for fused optimizer validation."""
    flag = _optimizer_workspace_buffer(
        numpy,
        slot="invalid",
        size=1,
        dtype=numpy.uint32,
    )
    flag.fill(0)
    return flag

def _split_optimizer_storage(
    result: Any,
    references: Sequence[Tensor],
    numpy: Any,
) -> tuple[Storage, ...] | None:
    """Retain slices of one batched result without copying them again."""
    if not references:
        return ()
    dtype = references[0].dtype
    total = sum(reference.size for reference in references)
    storage = _storage(
        result,
        dtype=dtype,
        output_shape=(total,),
        numpy=numpy,
    )
    if storage is None:
        return None
    storages: list[Storage] = []
    offset = 0
    for reference in references:
        end = offset + reference.size
        buffer = storage.buffer[offset:end]
        if storage.kind == "cuda":
            storages.append(CudaStorage(buffer, dtype))
        else:
            storages.append(NumPyStorage(buffer, dtype))
        offset = end
    return tuple(storages)

def _optimizer_scalar_batch(
    values: Sequence[float],
    references: Sequence[Tensor],
    numpy: Any,
) -> Any:
    """Expand one scalar per parameter into its batched element layout."""
    if values and all(value == values[0] for value in values[1:]):
        return float(values[0])
    scalars = numpy.asarray(tuple(values), dtype=numpy.float64)
    counts = numpy.asarray(
        tuple(reference.size for reference in references),
        dtype=numpy.int64,
    )
    return numpy.repeat(scalars, counts)
