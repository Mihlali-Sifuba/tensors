"""Reference grouped cross-correlation VJPs for the Python backend."""

from __future__ import annotations

import itertools

from tensors.backend.python.kernels.reductions.pairwise import (
    pairwise_float_sum,
    pairwise_product_sum,
)
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.strides import Strides
from tensors.utils.convolution import resolve_geometry


def convolution_gradient(
    grad_values,
    input_values,
    kernel_values,
    grad_shape,
    input_shape,
    kernel_shape,
    *,
    dtype,
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
    groups: int,
    include_bias: bool,
    needs_input_grad: tuple[bool, ...],
) -> list[Storage | None]:
    """Evaluate the requested VJPs as fixed-width pairwise contractions.

    Each gradient element is the pairwise sum, in the declared dtype, of a
    fixed sequence of product terms, the sequence every backend uses:

    - an input gradient runs over the kernel offsets in row-major order and,
      within each offset, over the group's output channels; a combination
      that no output position reaches is an exact zero term;
    - a kernel gradient runs over the batch and, within it, the output
      positions in row-major order; a position whose source falls in the
      padding is an exact zero term;
    - a bias gradient is the pairwise sum of the upstream gradient over the
      batch and the output positions.
    """

    geometry = resolve_geometry(
        len(stride),
        input_shape,
        kernel_shape,
        None,
        stride,
        padding,
        dilation,
        groups,
    )
    need_values, need_kernel = (needs_input_grad[0], needs_input_grad[1])
    need_bias = include_bias and needs_input_grad[2]
    if not (need_values or need_kernel or need_bias):
        return [None] * len(needs_input_grad)

    rank = geometry.rank
    input_strides = Strides.contiguous(geometry.canonical_input_shape)
    kernel_strides = Strides.contiguous(tuple(kernel_shape))
    grad_strides = Strides.contiguous(
        (geometry.batch, geometry.out_channels) + geometry.output_spatial
    )
    offsets = geometry.offsets
    positions = geometry.positions

    def upstream(batch_index, out_channel, position):
        index = batch_index * grad_strides[0] + out_channel * grad_strides[1]
        for axis in range(rank):
            index += position[axis] * grad_strides[axis + 2]
        return grad_values[index]

    def kernel_at(out_channel, channel, offset):
        index = out_channel * kernel_strides[0] + channel * kernel_strides[1]
        for axis in range(rank):
            index += offset[axis] * kernel_strides[axis + 2]
        return kernel_values[index]

    def input_at(batch_index, channel, coordinate):
        index = batch_index * input_strides[0] + channel * input_strides[1]
        for axis in range(rank):
            index += coordinate[axis] * input_strides[axis + 2]
        return input_values[index]

    results: list[Storage | None] = [None, None]
    if need_values:
        input_gradient = []
        for batch_index, channel, *coordinate in itertools.product(
            range(geometry.batch),
            range(geometry.in_channels),
            *map(range, geometry.spatial),
        ):
            group = channel // geometry.group_channels
            local_channel = channel % geometry.group_channels
            terms = []
            for offset in offsets:
                position = []
                valid = True
                for axis in range(rank):
                    numerator = (
                        coordinate[axis]
                        + geometry.padding[axis]
                        - offset[axis] * geometry.dilation[axis]
                    )
                    if numerator % geometry.stride[axis]:
                        valid = False
                        break
                    step = numerator // geometry.stride[axis]
                    if not 0 <= step < geometry.output_spatial[axis]:
                        valid = False
                        break
                    position.append(step)
                for local_output in range(geometry.group_outputs):
                    out_channel = group * geometry.group_outputs + local_output
                    terms.append(
                        (
                            upstream(batch_index, out_channel, position),
                            kernel_at(out_channel, local_channel, offset),
                        )
                        if valid
                        else (0.0, 0.0)
                    )
            input_gradient.append(pairwise_product_sum(terms, dtype))
        results[0] = PythonStorage.from_values(input_gradient, dtype)
    if need_kernel:
        kernel_gradient = []
        for out_channel, local_channel, *offset in itertools.product(
            range(geometry.out_channels),
            range(geometry.group_channels),
            *map(range, geometry.kernel_spatial),
        ):
            group = out_channel // geometry.group_outputs
            input_channel = group * geometry.group_channels + local_channel
            terms = []
            for batch_index in range(geometry.batch):
                for position in positions:
                    source = tuple(
                        position[axis] * geometry.stride[axis]
                        - geometry.padding[axis]
                        + offset[axis] * geometry.dilation[axis]
                        for axis in range(rank)
                    )
                    inside = all(
                        0 <= source[axis] < geometry.spatial[axis]
                        for axis in range(rank)
                    )
                    terms.append(
                        (
                            upstream(batch_index, out_channel, position),
                            input_at(batch_index, input_channel, source),
                        )
                        if inside
                        else (0.0, 0.0)
                    )
            kernel_gradient.append(pairwise_product_sum(terms, dtype))
        results[1] = PythonStorage.from_values(kernel_gradient, dtype)
    if include_bias:
        results.append(
            PythonStorage.from_values(
                [
                    pairwise_float_sum(
                        (
                            upstream(batch_index, out_channel, position)
                            for batch_index in range(geometry.batch)
                            for position in positions
                        ),
                        dtype,
                    )
                    for out_channel in range(geometry.out_channels)
                ],
                dtype,
            )
            if need_bias
            else None
        )
    return results
