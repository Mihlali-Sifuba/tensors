"""Padding, tiling, column extraction, and storage shared by both passes."""

from __future__ import annotations
import cupy
import itertools
from typing import Any
from typing import TYPE_CHECKING
from tensors.backend.cuda.storage import CudaStorage
from tensors.backend.storage import Storage
from tensors.backend.cuda.conversion import _finite_operands
from tensors.backend.cuda.conversion import _shape_size
from tensors.backend.cuda.conversion import _view

if TYPE_CHECKING:
    from tensors.dtype import DataType
    from tensors.tensor import Tensor
_CONVOLUTION_COLUMN_MAX_ELEMENTS = 4 * 1_024 * 1_024


def _convolution_tile_shape(
    batch: int,
    channels: int,
    kernel_spatial: tuple[int, ...],
    output_spatial: tuple[int, ...],
) -> tuple[int, tuple[int, ...]]:
    """Choose a spatial and batch tile bounded by the column memory budget."""
    patch = channels * _shape_size(kernel_spatial)
    max_positions = max(1, _CONVOLUTION_COLUMN_MAX_ELEMENTS // max(patch, 1))
    extents = list(output_spatial)
    while _shape_size(tuple(extents)) > max_positions:
        axis = max(range(len(extents)), key=extents.__getitem__)
        extents[axis] = max(1, (extents[axis] + 1) // 2)
    positions = _shape_size(tuple(extents))
    batch_extent = max(
        1, min(batch, _CONVOLUTION_COLUMN_MAX_ELEMENTS // max(patch * positions, 1))
    )
    return (batch_extent, tuple(extents))


def _convolution_tiles(
    batch: int,
    channels: int,
    kernel_spatial: tuple[int, ...],
    output_spatial: tuple[int, ...],
) -> Any:
    """Yield bounded batch slices and spatial output tiles."""
    batch_extent, tile_spatial = _convolution_tile_shape(
        batch, channels, kernel_spatial, output_spatial
    )
    starts = tuple(
        (range(0, size, tile) for size, tile in zip(output_spatial, tile_spatial))
    )
    for batch_start in range(0, batch, batch_extent):
        batch_stop = min(batch_start + batch_extent, batch)
        for spatial_start in itertools.product(*starts):
            spatial_extent = tuple(
                (
                    min(tile, size - start)
                    for tile, size, start in zip(
                        tile_spatial, output_spatial, spatial_start
                    )
                )
            )
            yield (slice(batch_start, batch_stop), spatial_start, spatial_extent)


def _pad_convolution_input(values: Any, padding: tuple[int, ...]) -> Any:
    """Pad spatial axes once before processing bounded column tiles."""
    if not any((pad > 0 for pad in padding)):
        return values
    pad_width = ((0, 0), (0, 0)) + tuple(((pad, pad) for pad in padding))
    return cupy.pad(values, pad_width)


def _convolution_columns(
    values: Any,
    kernel_spatial: tuple[int, ...],
    output_start: tuple[int, ...],
    output_extent: tuple[int, ...],
    stride: tuple[int, ...],
    dilation: tuple[int, ...],
) -> Any:
    """Gather one bounded receptive-field tile into a column array."""
    batch, channels = (int(values.shape[0]), int(values.shape[1]))
    columns = cupy.empty(
        (batch, channels) + kernel_spatial + output_extent, dtype=values.dtype
    )
    for offsets in itertools.product(*(range(size) for size in kernel_spatial)):
        source: list[Any] = [slice(None), slice(None)]
        for axis, offset in enumerate(offsets):
            start = output_start[axis] * stride[axis] + offset * dilation[axis]
            stop = start + (output_extent[axis] - 1) * stride[axis] + 1
            source.append(slice(start, stop, stride[axis]))
        columns[(slice(None), slice(None)) + offsets] = values[tuple(source)]
    return columns


def _convolution_operands(
    inputs: Tensor, kernel: Tensor, dtype: DataType
) -> tuple[Any, Any] | None:
    """Return finite convolution operands in the requested working dtype."""
    provider_dtype = cupy.dtype(dtype.name)
    try:
        input_values = _view(inputs).astype(provider_dtype, copy=False)
        kernel_values = _view(kernel).astype(provider_dtype, copy=False)
    except (TypeError, ValueError):
        return None
    if not _finite_operands(input_values, kernel_values):
        return None
    return (input_values, kernel_values)


def _convolution_storage(
    result: Any, *, dtype: DataType, output_shape: tuple[int, ...]
) -> Storage:
    """Retain a finite native convolution result without precision round trips."""
    target_dtype = cupy.dtype(dtype.name)
    contiguous = cupy.ascontiguousarray(result, dtype=target_dtype).reshape(-1)
    storage: Storage
    storage = CudaStorage(contiguous, dtype)
    if storage.size != _shape_size(output_shape):
        raise RuntimeError("Convolution kernel returned an unexpected result size")
    return storage
