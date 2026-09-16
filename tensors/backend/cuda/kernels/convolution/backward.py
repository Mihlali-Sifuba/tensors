"""Grouped cross-correlation vector-Jacobian products."""

from __future__ import annotations
import itertools
from typing import Any


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
    for offsets in itertools.product(*(range(size) for size in kernel_spatial)):
        destination: list[Any] = [batch_slice, slice(None)]
        for axis, offset in enumerate(offsets):
            start = output_start[axis] * stride[axis] + offset * dilation[axis]
            stop = start + (output_extent[axis] - 1) * stride[axis] + 1
            destination.append(slice(start, stop, stride[axis]))
        result[tuple(destination)] += columns[(slice(None), slice(None)) + offsets]
