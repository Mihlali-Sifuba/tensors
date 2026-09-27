"""CuPy implementation of grouped cross-correlation."""

from __future__ import annotations
import cupy
from typing import TYPE_CHECKING, Any
from tensors.backend.storage import Storage
from tensors.backend.cuda.conversion import _errstate
from tensors.backend.cuda.conversion import _shape_size
from tensors.backend.cuda.conversion import _widen
from tensors.backend.cuda.kernels.convolution.common import (
    _CONVOLUTION_COLUMN_MAX_ELEMENTS,
)
from tensors.backend.cuda.kernels.convolution.common import _convolution_columns
from tensors.backend.cuda.kernels.convolution.common import _convolution_operands
from tensors.backend.cuda.kernels.convolution.common import _convolution_storage
from tensors.backend.cuda.kernels.convolution.common import _convolution_tiles
from tensors.backend.cuda.kernels.convolution.common import _pad_convolution_input
from tensors.backend.cuda.kernels.linalg.contraction import certified_matmul
from tensors.backend.cuda.kernels.reductions.exact import certified_float_sum

if TYPE_CHECKING:
    from tensors.dtype import DataType


def convolution(
    input_values: Any,
    kernel_values: Any,
    bias_values: Any | None,
    input_shape: tuple[int, ...],
    kernel_shape: tuple[int, ...],
    bias_shape: tuple[int, ...] | None,
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
    operands = _convolution_operands(input_values, kernel_values)
    if operands is None:
        return None
    input_values, kernel_values = operands
    rank = len(stride)
    batched = len(input_shape) == rank + 2
    if not batched:
        input_values = input_values.reshape((1,) + tuple(input_values.shape))
    if bias_values is not None:
        try:
            bias_values = _widen(cupy.asarray(bias_values))
        except (TypeError, ValueError):
            return None
        if tuple(bias_values.shape) != bias_shape:
            return None
    batch = int(input_values.shape[0])
    in_channels = int(input_values.shape[1])
    out_channels = int(kernel_values.shape[0])
    kernel_spatial = tuple((int(size) for size in kernel_values.shape[2:]))
    output_spatial = tuple(
        (int(size) for size in (output_shape[2:] if batched else output_shape[1:]))
    )
    if any(padding) and not bool(cupy.all(cupy.isfinite(kernel_values))):
        return None
    padded_values = _pad_convolution_input(input_values, padding)
    result = cupy.empty(
        (batch, out_channels) + output_spatial, dtype=input_values.dtype
    )
    group_outputs = out_channels // groups
    group_patch = in_channels // groups * _shape_size(kernel_spatial)
    matrix = kernel_values.reshape(groups, group_outputs, group_patch)
    with _errstate(over="ignore", under="ignore", invalid="ignore"):
        for batch_slice, output_start, output_extent in _convolution_tiles(
            batch, in_channels, kernel_spatial, output_spatial
        ):
            values_tile = padded_values[batch_slice]
            columns = _convolution_columns(
                values_tile,
                kernel_spatial,
                output_start,
                output_extent,
                stride,
                dilation,
            )
            tile_batch = int(columns.shape[0])
            positions = _shape_size(output_extent)
            column_matrix = columns.reshape(
                tile_batch, groups, group_patch, positions
            )
            column_elements = int(column_matrix.size)
            channel_extent = max(
                1,
                min(
                    group_outputs,
                    _CONVOLUTION_COLUMN_MAX_ELEMENTS // max(column_elements, 1),
                ),
            )
            for group in range(groups):
                channel_base = group * group_outputs
                for channel_start in range(0, group_outputs, channel_extent):
                    channel_stop = min(
                        channel_start + channel_extent, group_outputs
                    )
                    tile = certified_matmul(
                        matrix[group : group + 1, channel_start:channel_stop],
                        column_matrix[:, group : group + 1],
                    )
                    if tile is None:
                        return None
                    target = (
                        batch_slice,
                        slice(
                            channel_base + channel_start,
                            channel_base + channel_stop,
                        ),
                    ) + tuple(
                        slice(start, start + extent)
                        for start, extent in zip(output_start, output_extent)
                    )
                    result[target] = tile.reshape(
                        (tile_batch, channel_stop - channel_start) + output_extent
                    )
        if bias_values is not None:
            terms = cupy.stack(
                (
                    result,
                    cupy.broadcast_to(
                        bias_values.reshape((1, out_channels) + (1,) * rank),
                        result.shape,
                    ),
                ),
                axis=0,
            )
            summed = certified_float_sum(terms, (0,))
            if summed is None:
                return None
            result = cupy.squeeze(summed, axis=0)
    logical_result = result if batched else result[0]
    return _convolution_storage(logical_result, dtype=dtype, output_shape=output_shape)
