"""Grouped cross-correlation kernels and their vector-Jacobian products.

Padding, tiling, column extraction, and scatter-add machinery is specific to
convolution and stays here."""

from __future__ import annotations

import itertools
from typing import Any, TYPE_CHECKING, cast

from ..storage import CudaStorage, NumPyStorage, Storage
from .core import _array_kind, _errstate, _finite_operands, _numpy, _shape_size, _view

if TYPE_CHECKING:
    from ...dtype import DataType
    from ...tensor import Tensor

_CONVOLUTION_COLUMN_MAX_ELEMENTS = 4 * 1024 * 1024

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
        1,
        min(
            batch,
            _CONVOLUTION_COLUMN_MAX_ELEMENTS
            // max(patch * positions, 1),
        ),
    )
    return batch_extent, tuple(extents)

def _convolution_tiles(
    batch: int,
    channels: int,
    kernel_spatial: tuple[int, ...],
    output_spatial: tuple[int, ...],
) -> Any:
    """Yield bounded batch slices and spatial output tiles."""
    batch_extent, tile_spatial = _convolution_tile_shape(
        batch,
        channels,
        kernel_spatial,
        output_spatial,
    )
    starts = tuple(
        range(0, size, tile)
        for size, tile in zip(output_spatial, tile_spatial)
    )
    for batch_start in range(0, batch, batch_extent):
        batch_stop = min(batch_start + batch_extent, batch)
        for spatial_start in itertools.product(*starts):
            spatial_extent = tuple(
                min(tile, size - start)
                for tile, size, start in zip(
                    tile_spatial,
                    output_spatial,
                    spatial_start,
                )
            )
            yield (
                slice(batch_start, batch_stop),
                spatial_start,
                spatial_extent,
            )

def _pad_convolution_input(
    values: Any,
    padding: tuple[int, ...],
    numpy: Any,
) -> Any:
    """Pad spatial axes once before processing bounded column tiles."""
    if not any(pad > 0 for pad in padding):
        return values
    pad_width = ((0, 0), (0, 0)) + tuple((pad, pad) for pad in padding)
    return numpy.pad(values, pad_width)

def _convolution_columns(
    values: Any,
    kernel_spatial: tuple[int, ...],
    output_start: tuple[int, ...],
    output_extent: tuple[int, ...],
    stride: tuple[int, ...],
    dilation: tuple[int, ...],
    numpy: Any,
) -> Any:
    """Gather one bounded receptive-field tile into a column array."""
    batch, channels = int(values.shape[0]), int(values.shape[1])
    columns = numpy.empty(
        (batch, channels) + kernel_spatial + output_extent,
        dtype=values.dtype,
    )
    for offsets in itertools.product(
        *(range(size) for size in kernel_spatial)
    ):
        source: list[Any] = [slice(None), slice(None)]
        for axis, offset in enumerate(offsets):
            start = (
                output_start[axis] * stride[axis]
                + offset * dilation[axis]
            )
            stop = start + (output_extent[axis] - 1) * stride[axis] + 1
            source.append(slice(start, stop, stride[axis]))
        columns[(slice(None), slice(None)) + offsets] = values[tuple(source)]
    return columns

def _convolution_scatter_add(
    result: Any,
    columns: Any,
    batch_slice: slice,
    kernel_spatial: tuple[int, ...],
    output_start: tuple[int, ...],
    output_extent: tuple[int, ...],
    stride: tuple[int, ...],
    dilation: tuple[int, ...],
) -> None:
    """Accumulate one bounded column-gradient tile into a padded input."""
    for offsets in itertools.product(
        *(range(size) for size in kernel_spatial)
    ):
        destination: list[Any] = [batch_slice, slice(None)]
        for axis, offset in enumerate(offsets):
            start = (
                output_start[axis] * stride[axis]
                + offset * dilation[axis]
            )
            stop = start + (output_extent[axis] - 1) * stride[axis] + 1
            destination.append(slice(start, stop, stride[axis]))
        result[tuple(destination)] += columns[
            (slice(None), slice(None)) + offsets
        ]

def _convolution_operands(
    inputs: Tensor,
    kernel: Tensor,
    dtype: DataType,
    numpy: Any,
) -> tuple[Any, Any] | None:
    """Return finite convolution operands in the requested working dtype."""
    provider_dtype = numpy.dtype(dtype.name)
    try:
        input_values = _view(inputs, numpy).astype(provider_dtype, copy=False)
        kernel_values = _view(kernel, numpy).astype(provider_dtype, copy=False)
    except (TypeError, ValueError):
        return None
    if not _finite_operands(input_values, kernel_values, numpy=numpy):
        return None
    return input_values, kernel_values

def _convolution_storage(
    result: Any,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
    numpy: Any,
) -> Storage:
    """Retain a finite native convolution result without precision round trips."""
    target_dtype = numpy.dtype(dtype.name)
    contiguous = numpy.ascontiguousarray(
        result,
        dtype=target_dtype,
    ).reshape(-1)
    storage: Storage
    if _array_kind(numpy) == "cuda":
        storage = CudaStorage(contiguous, dtype)
    else:
        storage = NumPyStorage(contiguous, dtype)
    if storage.size != _shape_size(output_shape):
        raise RuntimeError("Convolution kernel returned an unexpected result size")
    return storage

def convolution(
    inputs: Tensor,
    kernel: Tensor,
    bias: Tensor | None,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
    groups: int,
) -> Storage | None:
    """Run grouped cross-correlation in bounded native matrix-product tiles."""
    if dtype.kind != "floating":
        return None

    numpy = _numpy()
    operands = _convolution_operands(inputs, kernel, dtype, numpy)
    if operands is None:
        return None
    input_values, kernel_values = operands
    rank = len(stride)
    batched = inputs.ndim == rank + 2
    if not batched:
        input_values = input_values.reshape((1,) + tuple(input_values.shape))

    bias_values = None
    if bias is not None:
        try:
            bias_values = _view(bias, numpy).astype(
                numpy.dtype(dtype.name),
                copy=False,
            )
        except (TypeError, ValueError):
            return None
        if not _finite_operands(bias_values, numpy=numpy):
            return None

    batch = int(input_values.shape[0])
    in_channels = int(input_values.shape[1])
    out_channels = int(kernel_values.shape[0])
    kernel_spatial = tuple(int(size) for size in kernel_values.shape[2:])
    output_spatial = tuple(
        int(size) for size in (
            output_shape[2:] if batched else output_shape[1:]
        )
    )
    padded_values = _pad_convolution_input(input_values, padding, numpy)
    result = numpy.empty(
        (batch, out_channels) + output_spatial,
        dtype=input_values.dtype,
    )
    group_outputs = out_channels // groups
    group_patch = (in_channels // groups) * _shape_size(kernel_spatial)
    matrix = kernel_values.reshape(groups, group_outputs, group_patch)

    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        for batch_slice, output_start, output_extent in _convolution_tiles(
            batch,
            in_channels,
            kernel_spatial,
            output_spatial,
        ):
            values_tile = padded_values[batch_slice]
            columns = _convolution_columns(
                values_tile,
                kernel_spatial,
                output_start,
                output_extent,
                stride,
                dilation,
                numpy,
            )
            tile_batch = int(columns.shape[0])
            positions = _shape_size(output_extent)
            tile = numpy.matmul(
                matrix,
                columns.reshape(
                    tile_batch,
                    groups,
                    group_patch,
                    positions,
                ),
            ).reshape((tile_batch, out_channels) + output_extent)
            target = (batch_slice, slice(None)) + tuple(
                slice(start, start + extent)
                for start, extent in zip(output_start, output_extent)
            )
            result[target] = tile
        if bias_values is not None:
            result += bias_values.reshape(
                (1, out_channels) + (1,) * rank
            )

    if not bool(numpy.all(numpy.isfinite(result))):
        return None
    logical_result = result if batched else result[0]
    return _convolution_storage(
        logical_result,
        dtype=dtype,
        output_shape=output_shape,
        numpy=numpy,
    )

def convolution_gradient(
    grad: Tensor,
    inputs: Tensor,
    kernel: Tensor,
    *,
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
    groups: int,
    include_bias: bool,
    needs_input_grad: tuple[bool, ...] = (True, True, True),
) -> tuple[Storage | None, ...] | None:
    """Run the requested convolution VJPs in bounded native tiles."""
    if grad.dtype.kind != "floating":
        return None
    need_input = needs_input_grad[0]
    need_kernel = needs_input_grad[1]
    need_bias = include_bias and needs_input_grad[2]
    if not (need_input or need_kernel or need_bias):
        return (None,) * (3 if include_bias else 2)

    numpy = _numpy()
    operands = _convolution_operands(inputs, kernel, grad.dtype, numpy)
    if operands is None:
        return None
    input_values, kernel_values = operands
    rank = len(stride)
    batched = inputs.ndim == rank + 2
    if not batched:
        input_values = input_values.reshape((1,) + tuple(input_values.shape))
    try:
        upstream = _view(grad, numpy).astype(
            numpy.dtype(grad.dtype.name),
            copy=False,
        )
    except (TypeError, ValueError):
        return None
    if not batched:
        upstream = upstream.reshape((1,) + tuple(upstream.shape))
    if not _finite_operands(upstream, numpy=numpy):
        return None

    batch = int(input_values.shape[0])
    in_channels = int(input_values.shape[1])
    out_channels = int(kernel_values.shape[0])
    spatial = tuple(int(size) for size in input_values.shape[2:])
    kernel_spatial = tuple(int(size) for size in kernel_values.shape[2:])
    output_spatial = tuple(int(size) for size in upstream.shape[2:])
    group_outputs = out_channels // groups
    group_patch = (in_channels // groups) * _shape_size(kernel_spatial)
    padded_values = _pad_convolution_input(input_values, padding, numpy)
    padded_spatial = tuple(
        extent + 2 * padding[axis] for axis, extent in enumerate(spatial)
    )
    input_result = numpy.zeros(
        (batch, in_channels) + padded_spatial,
        dtype=upstream.dtype,
    )
    kernel_result = numpy.zeros(
        kernel_values.shape,
        dtype=upstream.dtype,
    )
    matrix = kernel_values.reshape(groups, group_outputs, group_patch)

    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        for batch_slice, output_start, output_extent in _convolution_tiles(
            batch,
            in_channels,
            kernel_spatial,
            output_spatial,
        ):
            columns = _convolution_columns(
                padded_values[batch_slice],
                kernel_spatial,
                output_start,
                output_extent,
                stride,
                dilation,
                numpy,
            )
            target = (batch_slice, slice(None)) + tuple(
                slice(start, start + extent)
                for start, extent in zip(output_start, output_extent)
            )
            upstream_tile = upstream[target]
            tile_batch = int(columns.shape[0])
            positions = _shape_size(output_extent)
            column_matrix = columns.reshape(
                tile_batch,
                groups,
                group_patch,
                positions,
            )
            upstream_matrix = upstream_tile.reshape(
                tile_batch,
                groups,
                group_outputs,
                positions,
            )
            if need_kernel:
                kernel_result += numpy.matmul(
                    upstream_matrix,
                    column_matrix.transpose(0, 1, 3, 2),
                ).sum(axis=0).reshape(kernel_values.shape)
            if need_input:
                input_columns = numpy.matmul(
                    matrix.transpose(0, 2, 1),
                    upstream_matrix,
                ).reshape(
                    (tile_batch, in_channels)
                    + kernel_spatial
                    + output_extent
                )
                _convolution_scatter_add(
                    input_result,
                    input_columns,
                    batch_slice,
                    kernel_spatial,
                    output_start,
                    output_extent,
                    stride,
                    dilation,
                )

        results: list[Any] = []
        if need_input and any(padding):
            interior = (slice(None), slice(None)) + tuple(
                slice(pad, pad + extent)
                for pad, extent in zip(padding, spatial)
            )
            input_result = input_result[interior]
        results.append(
            (input_result if batched else input_result[0])
            if need_input
            else None
        )
        results.append(kernel_result if need_kernel else None)
        if include_bias:
            results.append(
                upstream.sum(
                    axis=(0,) + tuple(range(2, 2 + len(output_spatial)))
                )
                if need_bias
                else None
            )

    if any(
        value is not None and not bool(numpy.all(numpy.isfinite(value)))
        for value in results
    ):
        return None
    shapes: list[tuple[int, ...]] = [
        tuple(inputs.shape),
        tuple(kernel.shape),
    ]
    if include_bias:
        shapes.append((out_channels,))
    storages = tuple(
        _convolution_storage(
            value,
            dtype=grad.dtype,
            output_shape=shape,
            numpy=numpy,
        )
        if value is not None
        else None
        for value, shape in zip(results, shapes)
    )
    return cast("tuple[Storage | None, ...]", storages)
