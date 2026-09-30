"""NumPy implementation of the grouped cross-correlation VJPs.

Each gradient element is the pairwise sum of a fixed sequence of product
terms (docs/summation-semantics.md section 11). The sequences are built for a
tile of destinations at once by gathering both factors with flat indices;
Python only iterates over tiles.

**Zero slots.** Each operand is viewed as rows (one per batch and channel)
and given one extra element per row that holds ``+0``. A term that
contributes nothing indexes that slot for both of its factors, so it is an
exact ``+0 * +0`` term in its place in the sequence, as the specification
requires, without a separate masking pass and without removing it.

**Cached geometry.** Which output position a ``(input position, kernel
offset)`` pair reaches, and which input position a ``(kernel offset, output
position)`` pair reads, depends only on the spatial geometry. Those maps are
built once per geometry and kept in a bounded LRU cache of
``_PLAN_CACHE_ENTRIES`` entries per kind, keyed by the spatial input, kernel
and output extents, stride, padding and dilation.
Batch, channels and groups do not enter the key: they are applied to a
cached map by broadcast additions. Plans are NumPy arrays. A plan holds two integer
arrays of ``input_positions * kernel_offsets`` or ``kernel_offsets *
output_positions`` elements; for LeNet-5's layers that is under 0.5 MiB.

**Temporary memory** is bounded per tile in bytes: ``16 * E``, where ``E``
is ``_CONVOLUTION_COLUMN_MAX_ELEMENTS``, so 64 MiB at the default ``E`` of
4 Mi. Each term of a tile costs its two flat 64-bit gather indices and four
values — the two gathered factors, the product, and the pairwise tree's
levels, which together hold fewer values than the products — so a tile holds
``T = 16 * E // (width * (16 + 4 * itemsize))`` destinations, where
``width`` is the number of terms per destination: about 2 Mi terms for
float32 and 1.4 Mi for float64. The zero-slotted operand copies add one copy
of the upstream gradient and of the kernel or input, which are small next to
a tile.
"""

from __future__ import annotations
import functools
import itertools
import math
import numpy
from typing import Any
from typing import TYPE_CHECKING
from typing import cast
from tensors.backend.storage import Storage
from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.kernels.convolution import common as convolution_common
from tensors.backend.numpy.kernels.convolution.common import _convolution_operands
from tensors.backend.numpy.kernels.convolution.common import _convolution_storage
from tensors.backend.numpy.kernels.reductions.pairwise import (
    pairwise_float_sum,
    to_declared_dtype,
)

if TYPE_CHECKING:
    from tensors.dtype import DataType


_PLAN_CACHE_ENTRIES = 32


def _row_major_coordinates(shape: tuple[int, ...]) -> Any:
    """Every coordinate of ``shape`` in row-major order, as ``(count, rank)``."""
    count = math.prod(shape)
    flat = numpy.arange(count, dtype=numpy.int64)
    return numpy.stack(numpy.unravel_index(flat, shape), axis=-1).reshape(count, len(shape))


def _row_major_strides(shape: tuple[int, ...]) -> Any:
    """The element strides of a contiguous array of ``shape``."""
    return numpy.asarray(
        [math.prod(shape[axis + 1 :]) for axis in range(len(shape))], dtype=numpy.int64
    )


def _tile_size(width: int, itemsize: int) -> int:
    """Destinations per tile, so one tile's working set stays within budget.

    The budget is in bytes: ``16 * _CONVOLUTION_COLUMN_MAX_ELEMENTS``. Each
    term costs its two 64-bit gather indices plus four values of ``itemsize``
    bytes — the two gathered factors, the product, and the pairwise tree's
    levels, which together hold fewer values than the products.
    """
    budget = 16 * convolution_common._CONVOLUTION_COLUMN_MAX_ELEMENTS
    per_term = 2 * 8 + 4 * itemsize
    return max(1, budget // (max(width, 1) * per_term))


def _with_zero_slot(values: Any, rows: int, row_size: int) -> Any:
    """``values`` as ``rows`` rows with one ``+0`` appended to each, flattened."""
    grid = values.reshape(rows, row_size)
    zeros = numpy.zeros((rows, 1), dtype=values.dtype)
    return numpy.concatenate((grid, zeros), axis=1).reshape(-1)


@functools.lru_cache(maxsize=_PLAN_CACHE_ENTRIES)
def _input_plan(
    spatial: tuple[int, ...],
    kernel_spatial: tuple[int, ...],
    output_spatial: tuple[int, ...],
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
) -> tuple[Any, Any]:
    """Where each ``(input position, kernel offset)`` pair reads its factors.

    Returns two ``(input_positions, kernel_offsets)`` arrays: the row-major
    output position the pair reaches, and the kernel offset itself. A pair
    that reaches no output position — ``x + padding - u * dilation`` not a
    multiple of the stride, or outside the output — reads the zero slot of
    both rows instead: output position ``output_positions`` and kernel offset
    ``kernel_offsets``.
    """
    positions = _row_major_coordinates(spatial)
    offsets = _row_major_coordinates(kernel_spatial)
    stride_array = numpy.asarray(stride, dtype=numpy.int64)
    numerator = (
        positions[:, None, :]
        + numpy.asarray(padding, dtype=numpy.int64)
        - offsets[None, :, :] * numpy.asarray(dilation, dtype=numpy.int64)
    )
    reached = numerator // stride_array
    valid = (
        (numerator % stride_array == 0)
        & (reached >= 0)
        & (reached < numpy.asarray(output_spatial, dtype=numpy.int64))
    ).all(axis=-1)
    output_count = math.prod(output_spatial)
    offset_count = math.prod(kernel_spatial)
    linear = (reached * _row_major_strides(output_spatial)).sum(axis=-1)
    upstream_slot = numpy.where(valid, linear, output_count)
    kernel_slot = numpy.where(
        valid, numpy.arange(offset_count, dtype=numpy.int64)[None, :], offset_count
    )
    return upstream_slot, kernel_slot


@functools.lru_cache(maxsize=_PLAN_CACHE_ENTRIES)
def _kernel_plan(
    spatial: tuple[int, ...],
    kernel_spatial: tuple[int, ...],
    output_spatial: tuple[int, ...],
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
) -> tuple[Any, Any]:
    """Where each ``(kernel offset, output position)`` pair reads its factors.

    Returns two ``(kernel_offsets, output_positions)`` arrays: the row-major
    input position ``p * stride - padding + u * dilation`` and the output
    position itself. A pair whose input position falls in the padding reads
    the zero slot of both rows: input position ``input_positions`` and output
    position ``output_positions``.
    """
    offsets = _row_major_coordinates(kernel_spatial)
    positions = _row_major_coordinates(output_spatial)
    source = (
        positions[None, :, :] * numpy.asarray(stride, dtype=numpy.int64)
        - numpy.asarray(padding, dtype=numpy.int64)
        + offsets[:, None, :] * numpy.asarray(dilation, dtype=numpy.int64)
    )
    inside = (
        (source >= 0) & (source < numpy.asarray(spatial, dtype=numpy.int64))
    ).all(axis=-1)
    input_count = math.prod(spatial)
    output_count = math.prod(output_spatial)
    linear = (source * _row_major_strides(spatial)).sum(axis=-1)
    input_slot = numpy.where(inside, linear, input_count)
    upstream_slot = numpy.where(
        inside, numpy.arange(output_count, dtype=numpy.int64)[None, :], output_count
    )
    return input_slot, upstream_slot


def _input_vjp(
    upstream: Any,
    kernel_values: Any,
    input_result: Any,
    *,
    spatial: tuple[int, ...],
    kernel_spatial: tuple[int, ...],
    output_spatial: tuple[int, ...],
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
    group_channels: int,
    group_outputs: int,
) -> None:
    """Fill ``input_result`` with the input VJP, one destination tile at a time.

    Destination ``(b, c, x)`` has ``kernel_offsets * group_outputs`` terms,
    ordered by kernel offset in row-major order and, within an offset, by
    the group's output channel ``o``: ``upstream[b, o, reached(x, u)] *
    kernel[o, c % group_channels, u]``, or ``+0 * +0`` where ``u`` reaches no
    output position.
    """
    batch, in_channels = (int(input_result.shape[0]), int(input_result.shape[1]))
    out_channels = int(upstream.shape[1])
    input_count = math.prod(spatial)
    offset_count = math.prod(kernel_spatial)
    output_count = math.prod(output_spatial)
    upstream_slot, kernel_slot = _input_plan(
        spatial, kernel_spatial, output_spatial, stride, padding, dilation
    )
    upstream_rows = _with_zero_slot(upstream, batch * out_channels, output_count)
    kernel_rows = _with_zero_slot(
        kernel_values, out_channels * group_channels, offset_count
    )
    width = offset_count * group_outputs
    local_outputs = numpy.arange(group_outputs, dtype=numpy.int64)
    destinations = input_result.reshape(-1)
    total = int(destinations.shape[0])
    tile = _tile_size(width, upstream.dtype.itemsize)
    for start in range(0, total, tile):
        stop = min(start + tile, total)
        count = stop - start
        destination = numpy.arange(start, stop, dtype=numpy.int64)
        position = destination % input_count
        batch_channel = destination // input_count
        channel = batch_channel % in_channels
        batch_index = batch_channel // in_channels
        out_channel = (channel // group_channels)[:, None] * group_outputs + local_outputs
        upstream_row = (batch_index[:, None] * out_channels + out_channel) * (
            output_count + 1
        )
        kernel_row = (
            out_channel * group_channels + (channel % group_channels)[:, None]
        ) * (offset_count + 1)
        upstream_index = upstream_row[:, None, :] + upstream_slot[position][:, :, None]
        kernel_index = kernel_row[:, None, :] + kernel_slot[position][:, :, None]
        left = numpy.take(upstream_rows, upstream_index.reshape(count, width))
        right = numpy.take(kernel_rows, kernel_index.reshape(count, width))
        products = left * right
        destinations[start:stop] = pairwise_float_sum(products, (1,)).reshape(count)


def _kernel_vjp(
    upstream: Any,
    input_values: Any,
    kernel_result: Any,
    *,
    spatial: tuple[int, ...],
    kernel_spatial: tuple[int, ...],
    output_spatial: tuple[int, ...],
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
    group_channels: int,
    group_outputs: int,
) -> None:
    """Fill ``kernel_result`` with the kernel VJP, one destination tile at a time.

    Weight ``(o, c, u)`` has ``batch * output_positions`` terms, ordered by
    batch and, within it, output position in row-major order:
    ``upstream[b, o, p] * input[b, i, source(u, p)]`` for the group's input
    channel ``i``, or ``+0 * +0`` where the source falls in the padding.
    """
    batch, out_channels = (int(upstream.shape[0]), int(upstream.shape[1]))
    in_channels = int(input_values.shape[1])
    input_count = math.prod(spatial)
    offset_count = math.prod(kernel_spatial)
    output_count = math.prod(output_spatial)
    input_slot, upstream_slot = _kernel_plan(
        spatial, kernel_spatial, output_spatial, stride, padding, dilation
    )
    upstream_rows = _with_zero_slot(upstream, batch * out_channels, output_count)
    input_rows = _with_zero_slot(input_values, batch * in_channels, input_count)
    width = batch * output_count
    batches = numpy.arange(batch, dtype=numpy.int64)
    destinations = kernel_result.reshape(-1)
    total = int(destinations.shape[0])
    tile = _tile_size(width, upstream.dtype.itemsize)
    for start in range(0, total, tile):
        stop = min(start + tile, total)
        count = stop - start
        destination = numpy.arange(start, stop, dtype=numpy.int64)
        offset = destination % offset_count
        weight_row = destination // offset_count
        local_channel = weight_row % group_channels
        out_channel = weight_row // group_channels
        in_channel = (out_channel // group_outputs) * group_channels + local_channel
        upstream_row = (batches[None, :] * out_channels + out_channel[:, None]) * (
            output_count + 1
        )
        input_row = (batches[None, :] * in_channels + in_channel[:, None]) * (
            input_count + 1
        )
        upstream_index = upstream_row[:, :, None] + upstream_slot[offset][:, None, :]
        input_index = input_row[:, :, None] + input_slot[offset][:, None, :]
        left = numpy.take(upstream_rows, upstream_index.reshape(count, width))
        right = numpy.take(input_rows, input_index.reshape(count, width))
        products = left * right
        destinations[start:stop] = pairwise_float_sum(products, (1,)).reshape(count)


def convolution_gradient(
    grad_values: Any,
    input_values: Any,
    kernel_values: Any,
    grad_shape: tuple[int, ...],
    input_shape: tuple[int, ...],
    kernel_shape: tuple[int, ...],
    *,
    dtype: DataType,
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
    groups: int,
    include_bias: bool,
    needs_input_grad: tuple[bool, ...] = (True, True, True),
) -> tuple[Storage | None, ...] | None:
    """Run the requested convolution VJPs in bounded native tiles."""
    if dtype.kind != "floating":
        return None
    need_input = needs_input_grad[0]
    need_kernel = needs_input_grad[1]
    need_bias = include_bias and needs_input_grad[2]
    if not (need_input or need_kernel or need_bias):
        return (None,) * (3 if include_bias else 2)
    operands = _convolution_operands(input_values, kernel_values, dtype)
    if operands is None:
        return None
    input_values, kernel_values = operands
    working_dtype = input_values.dtype
    rank = len(stride)
    batched = len(input_shape) == rank + 2
    if not batched:
        input_values = input_values.reshape((1,) + tuple(input_values.shape))
    try:
        upstream = to_declared_dtype(grad_values, dtype)
    except (TypeError, ValueError):
        return None
    if tuple(upstream.shape) != grad_shape:
        return None
    if not batched:
        upstream = upstream.reshape((1,) + tuple(upstream.shape))
    batch = int(input_values.shape[0])
    in_channels = int(input_values.shape[1])
    out_channels = int(kernel_values.shape[0])
    spatial = tuple((int(size) for size in input_values.shape[2:]))
    kernel_spatial = tuple((int(size) for size in kernel_values.shape[2:]))
    output_spatial = tuple((int(size) for size in upstream.shape[2:]))
    group_outputs = out_channels // groups
    group_channels = in_channels // groups
    offsets = tuple(itertools.product(*(range(size) for size in kernel_spatial)))
    output_position_count = math.prod(output_spatial)
    input_result = (
        numpy.zeros((batch, in_channels) + spatial, dtype=working_dtype)
        if need_input
        else None
    )
    kernel_result = (
        numpy.zeros(kernel_shape, dtype=working_dtype) if need_kernel else None
    )
    with _errstate(over="ignore", under="ignore", invalid="ignore"):
        if need_input:
            assert input_result is not None
            width = group_outputs * len(offsets)
            if width == 0:
                input_result.fill(0.0)
            elif width > convolution_common._CONVOLUTION_COLUMN_MAX_ELEMENTS:
                return None
            else:
                _input_vjp(
                    upstream,
                    kernel_values,
                    input_result,
                    spatial=spatial,
                    kernel_spatial=kernel_spatial,
                    output_spatial=output_spatial,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    group_channels=group_channels,
                    group_outputs=group_outputs,
                )

        if need_kernel:
            assert kernel_result is not None
            contribution_count = batch * output_position_count
            if contribution_count == 0:
                kernel_result.fill(0.0)
            elif (
                contribution_count > convolution_common._CONVOLUTION_COLUMN_MAX_ELEMENTS
            ):
                return None
            else:
                _kernel_vjp(
                    upstream,
                    input_values,
                    kernel_result,
                    spatial=spatial,
                    kernel_spatial=kernel_spatial,
                    output_spatial=output_spatial,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    group_channels=group_channels,
                    group_outputs=group_outputs,
                )

        results: list[Any] = []
        results.append(
            (input_result if batched else input_result[0]) if need_input else None
        )
        results.append(kernel_result if need_kernel else None)
        if include_bias:
            if not need_bias:
                results.append(None)
            elif out_channels == 0 or batch * output_position_count == 0:
                results.append(numpy.zeros((out_channels,), dtype=working_dtype))
            else:
                grouped = numpy.moveaxis(upstream, 1, 0).reshape(out_channels, -1)
                bias_result = pairwise_float_sum(grouped, (1,))
                results.append(bias_result.reshape(out_channels))
    shapes: list[tuple[int, ...]] = [input_shape, kernel_shape]
    requested = [need_input, need_kernel]
    if include_bias:
        shapes.append((out_channels,))
        requested.append(need_bias)
    storages: list[Storage | None] = []
    for value, shape, branch_requested in zip(results, shapes, requested):
        if not branch_requested:
            storages.append(None)
            continue
        if value is None:
            return None
        storage = _convolution_storage(value, dtype=dtype, output_shape=shape)
        if storage is None:
            return None
        storages.append(storage)
    return cast("tuple[Storage | None, ...]", tuple(storages))
