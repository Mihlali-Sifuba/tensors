"""CuPy implementation of the grouped cross-correlation VJPs.

Each gradient element is the pairwise sum of a fixed sequence of product
terms (docs/summation-semantics.md section 11). The sequences are built for a
tile of destinations at once with native index arithmetic and gathers; Python
only iterates over tiles. Terms that contribute nothing are set to exact zero
in place, never removed, because the pairwise tree is defined over the whole
sequence.

Temporary memory is bounded per tile by ``_CONVOLUTION_COLUMN_MAX_ELEMENTS``
(``E``). A tile holds ``T = E // max(width, offsets * rank)`` destinations for
the input VJP and ``T = E // max(width, positions * rank)`` for the kernel
VJP, where ``width`` is the number of terms per destination. The principal
arrays per tile are:

- integer coordinates, numerators and positions: ``(T, offsets, rank)`` or
  ``(T, positions, rank)``, at most ``E`` elements each;
- the two gathered factor arrays, ``(T, width)`` values each, at most ``E``;
- the products and the pairwise tree's levels, together at most ``2 * E``.

So a tile needs roughly ``4 * E`` values plus ``3 * E`` 64-bit integers: about
224 MiB for float64 and 160 MiB for float32 at the default ``E`` of 4 Mi.
"""

from __future__ import annotations
import itertools
import math
import cupy
from typing import Any
from typing import TYPE_CHECKING
from typing import cast
from tensors.backend.storage import Storage
from tensors.backend.cuda.conversion import _errstate
from tensors.backend.cuda.kernels.convolution import common as convolution_common
from tensors.backend.cuda.kernels.convolution.common import _convolution_operands
from tensors.backend.cuda.kernels.convolution.common import _convolution_storage
from tensors.backend.cuda.kernels.linalg.contraction import pairwise_matmul
from tensors.backend.cuda.kernels.reductions.pairwise import (
    pairwise_float_sum,
    to_declared_dtype,
)

if TYPE_CHECKING:
    from tensors.dtype import DataType


def _row_major_coordinates(shape: tuple[int, ...]) -> Any:
    """Every coordinate of ``shape`` in row-major order, as ``(count, rank)``."""
    count = math.prod(shape)
    flat = cupy.arange(count, dtype=cupy.int64)
    return cupy.stack(cupy.unravel_index(flat, shape), axis=-1).reshape(count, len(shape))


def _tile_size(width: int, index_elements: int) -> int:
    """Destinations per tile, so term and coordinate arrays stay in budget.

    ``width`` is the number of product terms per destination and
    ``index_elements`` the number of integer coordinates each destination
    needs. A tile holds at most ``_CONVOLUTION_COLUMN_MAX_ELEMENTS`` of each.
    """
    per_destination = max(width, index_elements, 1)
    return max(1, convolution_common._CONVOLUTION_COLUMN_MAX_ELEMENTS // per_destination)


def _zero_invalid(values: Any, valid: Any) -> Any:
    """Replace invalid terms by exact zero, in place and without compaction.

    Every term keeps its position in the sequence, because the pairwise tree
    is defined over that sequence and removing a term would reshape it.
    """
    cupy.putmask(values, ~cupy.broadcast_to(valid, values.shape), 0)
    return values


def _input_vjp(
    upstream: Any,
    kernel_values: Any,
    input_result: Any,
    *,
    kernel_spatial: tuple[int, ...],
    output_spatial: tuple[int, ...],
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
    group_channels: int,
    group_outputs: int,
) -> None:
    """Fill ``input_result`` with the input VJP, one destination tile at a time.

    Each destination ``(b, c, x)`` has ``len(offsets) * group_outputs``
    terms, ordered by kernel offset in row-major order and, within an offset,
    by the group's output channel. The offset ``u`` reaches output position
    ``(x + padding - u * dilation) / stride`` when that is an integer inside
    the output; any other combination is an exact zero term in its place.
    """
    rank = len(kernel_spatial)
    destination_shape = tuple(int(size) for size in input_result.shape)
    offsets = _row_major_coordinates(kernel_spatial)
    offset_count = int(offsets.shape[0])
    width = offset_count * group_outputs
    stride_array = cupy.asarray(stride, dtype=cupy.int64)
    padding_array = cupy.asarray(padding, dtype=cupy.int64)
    dilation_array = cupy.asarray(dilation, dtype=cupy.int64)
    output_array = cupy.asarray(output_spatial, dtype=cupy.int64)
    local_outputs = cupy.arange(group_outputs, dtype=cupy.int64)
    destinations = input_result.reshape(-1)
    total = int(destinations.shape[0])
    tile = _tile_size(width, offset_count * rank)
    for start in range(0, total, tile):
        stop = min(start + tile, total)
        count = stop - start
        coordinates = cupy.unravel_index(
            cupy.arange(start, stop, dtype=cupy.int64), destination_shape
        )
        batch_index = coordinates[0]
        channel = coordinates[1]
        spatial_index = cupy.stack(coordinates[2:], axis=-1).reshape(count, rank)
        numerator = (
            spatial_index[:, None, :] + padding_array - offsets[None, :, :] * dilation_array
        )
        position = numerator // stride_array
        valid = (
            (numerator % stride_array == 0) & (position >= 0) & (position < output_array)
        ).all(axis=-1)
        position = cupy.where(valid[..., None], position, 0)
        out_channel = (channel // group_channels)[:, None] * group_outputs + local_outputs
        local_channel = channel % group_channels
        upstream_terms = upstream[
            (batch_index[:, None, None], out_channel[:, None, :])
            + tuple(position[:, :, None, axis] for axis in range(rank))
        ]
        kernel_terms = kernel_values[
            (out_channel[:, None, :], local_channel[:, None, None])
            + tuple(offsets[None, :, None, axis] for axis in range(rank))
        ]
        mask = valid[:, :, None]
        left = _zero_invalid(upstream_terms, mask).reshape(count, width)
        right = _zero_invalid(kernel_terms, mask).reshape(count, width)
        destinations[start:stop] = pairwise_matmul(
            left[:, None, :], right[:, :, None]
        ).reshape(count)


def _kernel_vjp(
    upstream: Any,
    input_values: Any,
    kernel_result: Any,
    *,
    spatial: tuple[int, ...],
    output_spatial: tuple[int, ...],
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
    group_channels: int,
    group_outputs: int,
) -> None:
    """Fill ``kernel_result`` with the kernel VJP, one destination tile at a time.

    Each weight ``(o, c, u)`` has ``batch * output_positions`` terms, ordered
    by batch and, within it, output position in row-major order. Position
    ``p`` reads input ``p * stride - padding + u * dilation``; a source in the
    padding is an exact zero term in its place.
    """
    rank = len(spatial)
    batch = int(upstream.shape[0])
    destination_shape = tuple(int(size) for size in kernel_result.shape)
    positions = _row_major_coordinates(output_spatial)
    position_count = int(positions.shape[0])
    width = batch * position_count
    stride_array = cupy.asarray(stride, dtype=cupy.int64)
    padding_array = cupy.asarray(padding, dtype=cupy.int64)
    dilation_array = cupy.asarray(dilation, dtype=cupy.int64)
    spatial_array = cupy.asarray(spatial, dtype=cupy.int64)
    batches = cupy.arange(batch, dtype=cupy.int64)
    destinations = kernel_result.reshape(-1)
    total = int(destinations.shape[0])
    tile = _tile_size(width, position_count * rank)
    for start in range(0, total, tile):
        stop = min(start + tile, total)
        count = stop - start
        coordinates = cupy.unravel_index(
            cupy.arange(start, stop, dtype=cupy.int64), destination_shape
        )
        out_channel = coordinates[0]
        local_channel = coordinates[1]
        offset = cupy.stack(coordinates[2:], axis=-1).reshape(count, rank)
        source = (
            positions[None, :, :] * stride_array
            - padding_array
            + offset[:, None, :] * dilation_array
        )
        inside = ((source >= 0) & (source < spatial_array)).all(axis=-1)
        source = cupy.where(inside[..., None], source, 0)
        input_channel = (out_channel // group_outputs) * group_channels + local_channel
        upstream_terms = upstream[
            (batches[None, :, None], out_channel[:, None, None])
            + tuple(positions[None, None, :, axis] for axis in range(rank))
        ]
        input_terms = input_values[
            (batches[None, :, None], input_channel[:, None, None])
            + tuple(source[:, None, :, axis] for axis in range(rank))
        ]
        mask = inside[:, None, :]
        left = _zero_invalid(upstream_terms, mask).reshape(count, width)
        right = _zero_invalid(input_terms, mask).reshape(count, width)
        destinations[start:stop] = pairwise_matmul(
            left[:, None, :], right[:, :, None]
        ).reshape(count)


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
        cupy.zeros((batch, in_channels) + spatial, dtype=working_dtype)
        if need_input
        else None
    )
    kernel_result = (
        cupy.zeros(kernel_shape, dtype=working_dtype) if need_kernel else None
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
                results.append(cupy.zeros((out_channels,), dtype=working_dtype))
            else:
                grouped = cupy.moveaxis(upstream, 1, 0).reshape(out_channels, -1)
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
