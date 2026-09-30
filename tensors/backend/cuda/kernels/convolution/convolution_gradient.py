"""CuPy implementation of the grouped cross-correlation VJPs."""

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
                tile_size = max(
                    1, convolution_common._CONVOLUTION_COLUMN_MAX_ELEMENTS // width
                )
                destinations = itertools.product(
                    range(batch), range(in_channels), *map(range, spatial)
                )
                while tile := list(itertools.islice(destinations, tile_size)):
                    left = cupy.zeros((len(tile), width), dtype=working_dtype)
                    right = cupy.zeros_like(left)
                    for row, (batch_index, channel, *coordinate) in enumerate(tile):
                        group = channel // group_channels
                        local_channel = channel % group_channels
                        term = 0
                        for offset in offsets:
                            output_position = []
                            valid = True
                            for axis in range(rank):
                                numerator = (
                                    coordinate[axis]
                                    + padding[axis]
                                    - offset[axis] * dilation[axis]
                                )
                                if numerator % stride[axis]:
                                    valid = False
                                    break
                                position = numerator // stride[axis]
                                if not 0 <= position < output_spatial[axis]:
                                    valid = False
                                    break
                                output_position.append(position)
                            for local_output in range(group_outputs):
                                if valid:
                                    output_channel = (
                                        group * group_outputs + local_output
                                    )
                                    left[row, term] = upstream[
                                        (batch_index, output_channel, *output_position)
                                    ]
                                    right[row, term] = kernel_values[
                                        (output_channel, local_channel, *offset)
                                    ]
                                term += 1
                    products = pairwise_matmul(left[:, None, :], right[:, :, None])
                    for destination, value in zip(tile, products.reshape(-1)):
                        input_result[destination] = value

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
                tile_size = max(
                    1,
                    convolution_common._CONVOLUTION_COLUMN_MAX_ELEMENTS
                    // contribution_count,
                )
                kernel_destinations = itertools.product(
                    range(out_channels),
                    range(group_channels),
                    *map(range, kernel_spatial),
                )
                while tile := list(itertools.islice(kernel_destinations, tile_size)):
                    left = cupy.zeros(
                        (len(tile), contribution_count), dtype=working_dtype
                    )
                    right = cupy.zeros_like(left)
                    for row, (output_channel, local_channel, *offset) in enumerate(
                        tile
                    ):
                        group = output_channel // group_outputs
                        input_channel = group * group_channels + local_channel
                        term = 0
                        for batch_index in range(batch):
                            for position in itertools.product(
                                *(range(size) for size in output_spatial)
                            ):
                                source = tuple(
                                    position[axis] * stride[axis]
                                    - padding[axis]
                                    + offset[axis] * dilation[axis]
                                    for axis in range(rank)
                                )
                                if all(
                                    0 <= source[axis] < spatial[axis]
                                    for axis in range(rank)
                                ):
                                    left[row, term] = upstream[
                                        (batch_index, output_channel, *position)
                                    ]
                                    right[row, term] = input_values[
                                        (batch_index, input_channel, *source)
                                    ]
                                term += 1
                    products = pairwise_matmul(left[:, None, :], right[:, :, None])
                    for destination, value in zip(tile, products.reshape(-1)):
                        kernel_result[destination] = value
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
