"""Grouped cross-correlation forward execution."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
    from ....dtype import DataType
    from ....tensor import Tensor

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
