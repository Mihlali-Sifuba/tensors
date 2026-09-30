"""Reference grouped cross-correlation for the Python backend."""

from __future__ import annotations
from typing import Any
from tensors.backend.python.kernels.reductions.pairwise import (
    pairwise_float_sum,
    pairwise_product_sum,
)
from tensors.backend.python.storage import PythonStorage
from tensors.strides import Strides
from tensors.utils.convolution import (
    contributions as _contributions,
    resolve_geometry,
)


def convolution(
    input_values,
    kernel_values,
    bias_values,
    input_shape,
    kernel_shape,
    bias_shape,
    *,
    dtype,
    output_shape,
    stride,
    padding,
    dilation,
    groups,
):
    """Accumulate each receptive field in the specified order.

    A floating output element is the pairwise sum, in the declared dtype, of
    the products over its whole receptive field: input channels of the group
    in order, and within each channel the kernel offsets in row-major order,
    as every backend enumerates them. A tap that falls in the padding
    contributes an exact zero term rather than being skipped, so the tree has
    the same shape on every backend. The bias is then one more rounded
    addition.
    """

    geometry = resolve_geometry(
        len(stride),
        input_shape,
        kernel_shape,
        bias_shape,
        stride,
        padding,
        dilation,
        groups,
    )
    values: list[Any] = []
    if dtype.kind == "integer":
        for _, out_channel, pairs in _contributions(geometry, kernel_shape):
            total = sum(
                (
                    int(input_values[source]) * int(kernel_values[weight])
                    for source, weight in pairs
                )
            )
            if bias_values is not None:
                total += int(bias_values[out_channel])
            values.append(total)
        return PythonStorage.from_values(values, dtype)
    input_strides = Strides.contiguous(geometry.canonical_input_shape)
    kernel_strides = Strides.contiguous(tuple(kernel_shape))
    rank = geometry.rank
    for batch_index in range(geometry.batch):
        for out_channel in range(geometry.out_channels):
            group = out_channel // geometry.group_outputs
            for position in geometry.positions:
                terms: list[tuple[Any, Any]] = []
                for channel in range(geometry.group_channels):
                    input_channel = group * geometry.group_channels + channel
                    for offset in geometry.offsets:
                        weight = (
                            out_channel * kernel_strides[0]
                            + channel * kernel_strides[1]
                            + sum(
                                offset[axis] * kernel_strides[axis + 2]
                                for axis in range(rank)
                            )
                        )
                        source = (
                            batch_index * input_strides[0]
                            + input_channel * input_strides[1]
                        )
                        inside = True
                        for axis in range(rank):
                            coordinate = (
                                position[axis] * geometry.stride[axis]
                                - geometry.padding[axis]
                                + offset[axis] * geometry.dilation[axis]
                            )
                            if not 0 <= coordinate < geometry.spatial[axis]:
                                inside = False
                                break
                            source += coordinate * input_strides[axis + 2]
                        terms.append(
                            (input_values[source], kernel_values[weight])
                            if inside
                            else (0.0, 0.0)
                        )
                total = pairwise_product_sum(terms, dtype)
                if bias_values is not None:
                    total = pairwise_float_sum(
                        (total, bias_values[out_channel]), dtype
                    )
                values.append(total)
    return PythonStorage.from_values(values, dtype)
