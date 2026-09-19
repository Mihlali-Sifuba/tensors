"""Resolved convolution extents and the index map they imply.

A convolution's shape is fully determined by the input, kernel, and bias
shapes plus stride, padding, dilation, and groups. Resolving those extents,
and enumerating which input and weight positions each output touches, needs
no values at all. Keeping both here lets the operation layer and every
backend kernel agree on the geometry while each owns its own accumulation.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TypeAlias

from tensors.strides import Strides

SpatialArgument: TypeAlias = int | tuple[int, ...] | list[int]


def spatial_argument(
    value: SpatialArgument, rank: int, name: str, *, minimum: int
) -> tuple[int, ...]:
    """Normalize an int or per-axis sequence into immutable graph metadata."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer or a sequence of integers")
    if isinstance(value, int):
        values = (value,) * rank
    elif isinstance(value, (tuple, list)):
        values = tuple(value)
        if len(values) != rank:
            raise ValueError(
                f"{name} must contain {rank} values for a {rank}D convolution, got {len(values)}"
            )
    else:
        raise TypeError(f"{name} must be an integer or a sequence of integers")
    for entry in values:
        if isinstance(entry, bool) or not isinstance(entry, int):
            raise TypeError(f"{name} entries must be integers")
        if entry < minimum:
            raise ValueError(f"{name} entries must be at least {minimum}")
    return values


@dataclass(frozen=True)
class Geometry:
    """Validated convolution extents shared by both passes."""

    rank: int
    batched: bool
    batch: int
    in_channels: int
    out_channels: int
    groups: int
    group_channels: int
    group_outputs: int
    spatial: tuple[int, ...]
    kernel_spatial: tuple[int, ...]
    output_spatial: tuple[int, ...]
    stride: tuple[int, ...]
    padding: tuple[int, ...]
    dilation: tuple[int, ...]

    @property
    def output_shape(self) -> tuple[int, ...]:
        """Return the logical shape produced by this convolution."""
        leading = (
            (self.batch, self.out_channels) if self.batched else (self.out_channels,)
        )
        return leading + self.output_spatial

    @property
    def canonical_input_shape(self) -> tuple[int, ...]:
        """Return the input shape with an explicit batch dimension."""
        return (self.batch, self.in_channels) + self.spatial

    @property
    def offsets(self) -> tuple[tuple[int, ...], ...]:
        """Return every kernel offset in row-major kernel order."""
        return tuple(itertools.product(*(range(size) for size in self.kernel_spatial)))

    @property
    def positions(self) -> tuple[tuple[int, ...], ...]:
        """Return every output coordinate in row-major output order."""
        return tuple(itertools.product(*(range(size) for size in self.output_spatial)))


def resolve_geometry(
    rank: int,
    input_shape: Sequence[int],
    kernel_shape: Sequence[int],
    bias_shape: Sequence[int] | None,
    stride: SpatialArgument,
    padding: SpatialArgument,
    dilation: SpatialArgument,
    groups: int,
) -> Geometry:
    """Validate operand shapes and resolve every convolution extent."""
    input_shape = tuple(input_shape)
    kernel_shape = tuple(kernel_shape)
    input_rank = len(input_shape)
    if input_rank not in {rank + 1, rank + 2}:
        raise ValueError(
            f"conv{rank}d input must have {rank + 1} unbatched dimensions or {rank + 2} batched dimensions, got {input_rank}"
        )
    if len(kernel_shape) != rank + 2:
        raise ValueError(
            f"conv{rank}d kernel must have {rank + 2} dimensions (output channels, input channels, {rank} spatial), got {len(kernel_shape)}"
        )
    if isinstance(groups, bool) or not isinstance(groups, int):
        raise TypeError("groups must be an integer")
    if groups < 1:
        raise ValueError("groups must be at least 1")
    batched = input_rank == rank + 2
    batch = input_shape[0] if batched else 1
    in_channels = input_shape[1] if batched else input_shape[0]
    out_channels, kernel_channels = (kernel_shape[0], kernel_shape[1])
    if in_channels % groups:
        raise ValueError(
            f"Input channels {in_channels} is not divisible by groups {groups}"
        )
    if out_channels % groups:
        raise ValueError(
            f"Output channels {out_channels} is not divisible by groups {groups}"
        )
    if kernel_channels != in_channels // groups:
        raise ValueError(
            f"Kernel expects {kernel_channels} input channels per group but the input provides {in_channels // groups}"
        )
    strides = spatial_argument(stride, rank, "stride", minimum=1)
    paddings = spatial_argument(padding, rank, "padding", minimum=0)
    dilations = spatial_argument(dilation, rank, "dilation", minimum=1)
    spatial = tuple(input_shape[2:] if batched else input_shape[1:])
    kernel_spatial = tuple(kernel_shape[2:])
    if any((size < 1 for size in kernel_spatial)):
        raise ValueError("Kernel spatial dimensions must be at least 1")
    output_spatial = []
    for axis in range(rank):
        extent = (
            spatial[axis]
            + 2 * paddings[axis]
            - dilations[axis] * (kernel_spatial[axis] - 1)
            - 1
        )
        if extent < 0:
            span = dilations[axis] * (kernel_spatial[axis] - 1) + 1
            raise ValueError(
                f"Kernel span {span} on spatial axis {axis} exceeds the padded input extent {spatial[axis] + 2 * paddings[axis]}"
            )
        output_spatial.append(extent // strides[axis] + 1)
    if bias_shape is not None:
        bias_shape = tuple(bias_shape)
        if len(bias_shape) != 1 or bias_shape[0] != out_channels:
            raise ValueError(
                f"Bias shape {bias_shape} does not match the expected ({out_channels},)"
            )
    return Geometry(
        rank=rank,
        batched=batched,
        batch=batch,
        in_channels=in_channels,
        out_channels=out_channels,
        groups=groups,
        group_channels=in_channels // groups,
        group_outputs=out_channels // groups,
        spatial=spatial,
        kernel_spatial=kernel_spatial,
        output_spatial=tuple(output_spatial),
        stride=strides,
        padding=paddings,
        dilation=dilations,
    )


def contributions(
    geometry: Geometry,
    kernel_shape: Sequence[int],
) -> Iterator[tuple[int, int, list[tuple[int, int]]]]:
    """Yield every ``(output, input, weight)`` index a convolution touches.

    Output indices are produced in logical row-major order, which lets callers
    accumulate results without materializing coordinates twice.
    """
    input_strides = Strides.contiguous(geometry.canonical_input_shape)
    kernel_strides = Strides.contiguous(tuple(kernel_shape))
    offsets = geometry.offsets
    positions = geometry.positions
    output_index = 0
    for batch_index in range(geometry.batch):
        batch_base = batch_index * input_strides[0]
        for out_channel in range(geometry.out_channels):
            group = out_channel // geometry.group_outputs
            weight_base = out_channel * kernel_strides[0]
            group_base = batch_base + group * geometry.group_channels * input_strides[1]
            for position in positions:
                pairs: list[tuple[int, int]] = []
                for channel in range(geometry.group_channels):
                    source_base = group_base + channel * input_strides[1]
                    channel_weight = weight_base + channel * kernel_strides[1]
                    for offset in offsets:
                        source = source_base
                        weight = channel_weight
                        inside = True
                        for axis in range(geometry.rank):
                            coordinate = (
                                position[axis] * geometry.stride[axis]
                                - geometry.padding[axis]
                                + offset[axis] * geometry.dilation[axis]
                            )
                            if not 0 <= coordinate < geometry.spatial[axis]:
                                inside = False
                                break
                            source += coordinate * input_strides[axis + 2]
                            weight += offset[axis] * kernel_strides[axis + 2]
                        if inside:
                            pairs.append((source, weight))
                yield (output_index, out_channel, pairs)
                output_index += 1


__all__ = [
    "Geometry",
    "SpatialArgument",
    "contributions",
    "resolve_geometry",
    "spatial_argument",
]
