"""Grouped cross-correlation vector-Jacobian products."""

from __future__ import annotations

import itertools
from typing import Any, TYPE_CHECKING, cast

from ...storage import Storage
from ..core import _errstate, _finite_operands, _numpy, _shape_size, _view
from .common import (
    _convolution_columns,
    _convolution_operands,
    _convolution_storage,
    _convolution_tiles,
    _pad_convolution_input,
)

if TYPE_CHECKING:
    from ....tensor import Tensor

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
