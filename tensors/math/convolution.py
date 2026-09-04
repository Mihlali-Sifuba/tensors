"""Cross-correlation convolution over one, two, or three spatial axes.

The public functions accept batched or unbatched channel-first inputs and
slide a kernel over their spatial axes without reversing it. That is the
operation deep-learning frameworks call convolution; see :func:`conv2d` for
the exact definition and its relationship to true convolution.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, List, Optional, TypeAlias, overload

from .._typing import TensorData, TensorLike, TensorResult
from ..backend import execute_convolution, execute_convolution_gradient
from ..dtype import DataType, result_dtype
from ..strides import Strides
from ..ops.operation import Operation
from ..tensor import Tensor
from .sum import _stable_float_sum, _stable_product_sum

if TYPE_CHECKING:
    from ..variable import Variable


SpatialArgument: TypeAlias = int | tuple[int, ...] | list[int]


def _spatial_argument(
    value: SpatialArgument,
    rank: int,
    name: str,
    *,
    minimum: int,
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
                f"{name} must contain {rank} values for a {rank}D "
                f"convolution, got {len(values)}"
            )
    else:
        raise TypeError(f"{name} must be an integer or a sequence of integers")
    for entry in values:
        if isinstance(entry, bool) or not isinstance(entry, int):
            raise TypeError(f"{name} entries must be integers")
        if entry < minimum:
            raise ValueError(f"{name} entries must be at least {minimum}")
    return values


def _as_tensor(value: TensorLike) -> Tensor:
    """Return the Tensor value behind any accepted convolution operand."""
    from ..variable import Variable

    if isinstance(value, Tensor):
        return value
    if isinstance(value, Variable):
        return value.data
    return Tensor(value)


@dataclass(frozen=True)
class _Geometry:
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
        leading = (self.batch, self.out_channels) if self.batched else (
            self.out_channels,
        )
        return leading + self.output_spatial

    @property
    def canonical_input_shape(self) -> tuple[int, ...]:
        """Return the input shape with an explicit batch dimension."""
        return (self.batch, self.in_channels) + self.spatial

    @property
    def offsets(self) -> tuple[tuple[int, ...], ...]:
        """Return every kernel offset in row-major kernel order."""
        return tuple(
            itertools.product(*(range(size) for size in self.kernel_spatial))
        )

    @property
    def positions(self) -> tuple[tuple[int, ...], ...]:
        """Return every output coordinate in row-major output order."""
        return tuple(
            itertools.product(*(range(size) for size in self.output_spatial))
        )


def _geometry(
    rank: int,
    inputs: Tensor,
    kernel: Tensor,
    bias: Tensor | None,
    stride: SpatialArgument,
    padding: SpatialArgument,
    dilation: SpatialArgument,
    groups: int,
) -> _Geometry:
    """Validate operands and resolve every convolution extent."""
    if inputs.ndim not in {rank + 1, rank + 2}:
        raise ValueError(
            f"conv{rank}d input must have {rank + 1} unbatched dimensions "
            f"or {rank + 2} batched dimensions, got {inputs.ndim}"
        )
    if kernel.ndim != rank + 2:
        raise ValueError(
            f"conv{rank}d kernel must have {rank + 2} dimensions "
            f"(output channels, input channels, {rank} spatial), got "
            f"{kernel.ndim}"
        )
    if isinstance(groups, bool) or not isinstance(groups, int):
        raise TypeError("groups must be an integer")
    if groups < 1:
        raise ValueError("groups must be at least 1")

    batched = inputs.ndim == rank + 2
    batch = inputs.shape[0] if batched else 1
    in_channels = inputs.shape[1] if batched else inputs.shape[0]
    out_channels, kernel_channels = kernel.shape[0], kernel.shape[1]
    if in_channels % groups:
        raise ValueError(
            f"Input channels {in_channels} is not divisible by groups {groups}"
        )
    if out_channels % groups:
        raise ValueError(
            f"Output channels {out_channels} is not divisible by groups "
            f"{groups}"
        )
    if kernel_channels != in_channels // groups:
        raise ValueError(
            f"Kernel expects {kernel_channels} input channels per group but "
            f"the input provides {in_channels // groups}"
        )

    strides = _spatial_argument(stride, rank, "stride", minimum=1)
    paddings = _spatial_argument(padding, rank, "padding", minimum=0)
    dilations = _spatial_argument(dilation, rank, "dilation", minimum=1)
    spatial = tuple(inputs.shape[2:] if batched else inputs.shape[1:])
    kernel_spatial = tuple(kernel.shape[2:])
    if any(size < 1 for size in kernel_spatial):
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
                f"Kernel span {span} on spatial axis {axis} exceeds the "
                f"padded input extent {spatial[axis] + 2 * paddings[axis]}"
            )
        output_spatial.append(extent // strides[axis] + 1)

    if bias is not None:
        if bias.ndim != 1 or bias.shape[0] != out_channels:
            raise ValueError(
                f"Bias shape {tuple(bias.shape)} does not match the expected "
                f"({out_channels},)"
            )

    return _Geometry(
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


def _contributions(
    geometry: _Geometry,
    inputs: Tensor,
    kernel: Tensor,
) -> Iterator[tuple[int, int, list[tuple[int, int]]]]:
    """Yield every ``(output, input, weight)`` index a convolution touches.

    Output indices are produced in logical row-major order, which lets callers
    accumulate results without materializing coordinates twice.
    """
    input_strides = Strides.contiguous(geometry.canonical_input_shape)
    kernel_strides = Strides.contiguous(kernel.shape)
    offsets = geometry.offsets
    positions = geometry.positions
    output_index = 0
    for batch_index in range(geometry.batch):
        batch_base = batch_index * input_strides[0]
        for out_channel in range(geometry.out_channels):
            group = out_channel // geometry.group_outputs
            weight_base = out_channel * kernel_strides[0]
            group_base = batch_base + (
                group * geometry.group_channels * input_strides[1]
            )
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
                yield output_index, out_channel, pairs
                output_index += 1


class ConvND(Operation):
    """Internal rank-parameterized cross-correlation graph operation."""

    __slots__ = ("rank", "stride", "padding", "dilation", "groups")

    def __init__(
        self,
        *,
        rank: int,
        stride: SpatialArgument = 1,
        padding: SpatialArgument = 0,
        dilation: SpatialArgument = 1,
        groups: int = 1,
    ) -> None:
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "stride", stride)
        object.__setattr__(self, "padding", padding)
        object.__setattr__(self, "dilation", dilation)
        object.__setattr__(self, "groups", groups)

    @property
    def name(self) -> str:
        """Return the rank-specific label, such as ``conv2d``."""
        return f"conv{self.rank}d"

    def forward(
        self,
        inputs: Tensor,
        kernel: Tensor,
        bias: Tensor | None = None,
    ) -> Tensor:
        """Correlate ``inputs`` with ``kernel`` and add an optional bias."""
        geometry = self._geometry_for(inputs, kernel, bias)
        dtype = result_dtype(inputs.dtype, kernel)
        if bias is not None:
            dtype = result_dtype(dtype, bias)

        accelerated = execute_convolution(
            inputs,
            kernel,
            bias,
            dtype=dtype,
            output_shape=geometry.output_shape,
            stride=geometry.stride,
            padding=geometry.padding,
            dilation=geometry.dilation,
            groups=geometry.groups,
        )
        if accelerated is not None:
            return Tensor._from_owned_storage(
                accelerated,
                dtype=dtype,
                shape=geometry.output_shape,
            )
        return self._reference_forward(inputs, kernel, bias, geometry, dtype)

    @staticmethod
    def _reference_forward(
        inputs: Tensor,
        kernel: Tensor,
        bias: Tensor | None,
        geometry: _Geometry,
        dtype: DataType,
    ) -> Tensor:
        """Accumulate each receptive field with reference-exact arithmetic."""
        input_data = inputs._data
        kernel_data = kernel._data
        bias_data = bias._data if bias is not None else None
        exact = dtype.kind == "integer"
        values: list[Any] = []
        for _, out_channel, pairs in _contributions(geometry, inputs, kernel):
            total: Any
            if exact:
                total = sum(
                    int(input_data[source]) * int(kernel_data[weight])
                    for source, weight in pairs
                )
                if bias_data is not None:
                    total += int(bias_data[out_channel])
            else:
                total = _stable_product_sum([
                    (float(input_data[source]), float(kernel_data[weight]))
                    for source, weight in pairs
                ])
                if bias_data is not None:
                    total = _stable_float_sum(
                        [total, float(bias_data[out_channel])]
                    )
            values.append(total)
        return Tensor(values, dtype=dtype, shape=geometry.output_shape)

    def _geometry_for(
        self,
        inputs: Tensor,
        kernel: Tensor,
        bias: Tensor | None,
    ) -> "_Geometry":
        """Resolve this invocation's spatial geometry for given operands."""
        return _geometry(
            self.rank,
            inputs,
            kernel,
            bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def backward(
        self,
        grad: Tensor,
        *inputs: Tensor,
        needs_input_grad: tuple[bool, ...],
    ) -> List[Optional[Tensor]]:
        """Differentiate a convolution with respect to its requested inputs."""
        values, kernel = inputs[0], inputs[1]
        bias = inputs[2] if len(inputs) > 2 else None
        geometry = _geometry(
            self.rank,
            values,
            kernel,
            bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        if tuple(grad.shape) != geometry.output_shape:
            raise ValueError(
                f"Gradient shape {tuple(grad.shape)} does not match output "
                f"shape {geometry.output_shape}"
            )

        shapes: list[tuple[int, ...]] = [
            tuple(values.shape),
            tuple(kernel.shape),
        ]
        if bias is not None:
            shapes.append((geometry.out_channels,))
        accelerated = execute_convolution_gradient(
            grad,
            values,
            kernel,
            stride=geometry.stride,
            padding=geometry.padding,
            dilation=geometry.dilation,
            groups=geometry.groups,
            include_bias=bias is not None,
            needs_input_grad=needs_input_grad,
        )
        if accelerated is not None:
            return [
                Tensor._from_owned_storage(
                    storage,
                    dtype=grad.dtype,
                    shape=shape,
                )
                if storage is not None
                else None
                for storage, shape in zip(accelerated, shapes)
            ]
        return self._reference_backward(
            grad,
            values,
            kernel,
            bias,
            geometry,
            needs_input_grad,
        )

    def backward_graph(
        self,
        grad,
        *inputs,
        needs_input_grad: tuple[bool, ...],
    ):
        """Build a differentiable convolution VJP from primitive graph ops."""
        from ..math.reshape import reshape
        from ..math.stack import stack
        from ..ops._utils import zero_like_graph

        values, kernel = inputs[0], inputs[1]
        bias = inputs[2] if len(inputs) > 2 else None
        geometry = _geometry(
            self.rank,
            values.data,
            kernel.data,
            bias.data if bias is not None else None,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        grad_flat = reshape(grad, (grad.size,))
        values_flat = reshape(values, (values.size,))
        kernel_flat = reshape(kernel, (kernel.size,))
        input_terms: list[list[Any]] = [[] for _ in range(values.size)]
        kernel_terms: list[list[Any]] = [[] for _ in range(kernel.size)]
        bias_terms: list[list[Any]] = [
            [] for _ in range(geometry.out_channels)
        ]

        for output_index, out_channel, pairs in _contributions(
            geometry,
            values.data,
            kernel.data,
        ):
            upstream = grad_flat[output_index]
            for source, weight in pairs:
                input_terms[source].append(upstream * kernel_flat[weight])
                kernel_terms[weight].append(upstream * values_flat[source])
            if bias is not None:
                bias_terms[out_channel].append(upstream)

        def accumulate(terms: list[Any], reference: Any) -> Any:
            if not terms:
                return zero_like_graph(reference)
            result = terms[0]
            for term in terms[1:]:
                result = result + term
            return result

        input_gradient = reshape(
            stack([
                reshape(accumulate(terms, values_flat[index]), (1,))
                for index, terms in enumerate(input_terms)
            ]),
            values.shape,
        )
        kernel_gradient = reshape(
            stack([
                reshape(accumulate(terms, kernel_flat[index]), (1,))
                for index, terms in enumerate(kernel_terms)
            ]),
            kernel.shape,
        )
        results = [input_gradient, kernel_gradient]
        if bias is not None:
            bias_flat = reshape(bias, (bias.size,))
            results.append(
                reshape(
                    stack([
                        reshape(accumulate(terms, bias_flat[index]), (1,))
                        for index, terms in enumerate(bias_terms)
                    ]),
                    bias.shape,
                )
            )
        return results

    @staticmethod
    def _reference_backward(
        grad: Tensor,
        values: Tensor,
        kernel: Tensor,
        bias: Tensor | None,
        geometry: _Geometry,
        needs_input_grad: tuple[bool, ...],
    ) -> List[Optional[Tensor]]:
        """Collect the requested VJP terms before summing them stably."""
        need_values, need_kernel = needs_input_grad[0], needs_input_grad[1]
        need_bias = bias is not None and needs_input_grad[2]
        grad_data = grad._data
        input_data = values._data
        kernel_data = kernel._data
        input_terms: list[list[tuple[float, float]]] = [
            [] for _ in range(values.size if need_values else 0)
        ]
        kernel_terms: list[list[tuple[float, float]]] = [
            [] for _ in range(kernel.size if need_kernel else 0)
        ]
        bias_terms: list[list[float]] = [
            [] for _ in range(geometry.out_channels if need_bias else 0)
        ]
        if not (need_values or need_kernel or need_bias):
            return [None] * len(needs_input_grad)

        for index, out_channel, pairs in _contributions(
            geometry, values, kernel
        ):
            upstream = float(grad_data[index])
            if need_values or need_kernel:
                for source, weight in pairs:
                    if need_values:
                        input_terms[source].append(
                            (upstream, float(kernel_data[weight]))
                        )
                    if need_kernel:
                        kernel_terms[weight].append(
                            (upstream, float(input_data[source]))
                        )
            if need_bias:
                bias_terms[out_channel].append(upstream)

        results: List[Optional[Tensor]] = [
            Tensor(
                [_stable_product_sum(terms) for terms in input_terms],
                dtype=grad.dtype,
                shape=values.shape,
            )
            if need_values
            else None,
            Tensor(
                [_stable_product_sum(terms) for terms in kernel_terms],
                dtype=grad.dtype,
                shape=kernel.shape,
            )
            if need_kernel
            else None,
        ]
        if bias is not None:
            results.append(
                Tensor(
                    [_stable_float_sum(terms) for terms in bias_terms],
                    dtype=grad.dtype,
                    shape=(geometry.out_channels,),
                )
                if need_bias
                else None
            )
        return results


def _convolve(
    rank: int,
    inputs: TensorLike,
    kernel: TensorLike,
    bias: TensorLike | None,
    stride: SpatialArgument,
    padding: SpatialArgument,
    dilation: SpatialArgument,
    groups: int,
) -> TensorResult:
    """Dispatch a convolution over Tensor or Variable operands."""
    from ..variable import Variable

    # Normalize before tracing so replayed graph metadata is immutable and
    # does not need to be revalidated against the original argument forms.
    strides = _spatial_argument(stride, rank, "stride", minimum=1)
    paddings = _spatial_argument(padding, rank, "padding", minimum=0)
    dilations = _spatial_argument(
        dilation, rank, "dilation", minimum=1
    )
    operands: list[TensorLike] = [inputs, kernel]
    if bias is not None:
        operands.append(bias)

    if any(isinstance(operand, Variable) for operand in operands):
        variables = [
            operand
            if isinstance(operand, Variable)
            else Variable(operand, requires_grad=False)
            for operand in operands
        ]
        values = [variable.data for variable in variables]
        operation = ConvND(
            rank=rank,
            stride=strides,
            padding=paddings,
            dilation=dilations,
            groups=groups,
        )
        result: TensorResult = Variable._from_operation(
            operation.forward(
                values[0],
                values[1],
                values[2] if len(values) > 2 else None,
            ),
            operation,
            variables,
        )
        return result

    tensors = [_as_tensor(operand) for operand in operands]
    operation = ConvND(
        rank=rank,
        stride=strides,
        padding=paddings,
        dilation=dilations,
        groups=groups,
    )
    return operation.forward(
        tensors[0],
        tensors[1],
        tensors[2] if len(tensors) > 2 else None,
    )


@overload
def conv1d(
    inputs: Variable,
    kernel: TensorLike,
    bias: TensorLike | None = ...,
    *,
    stride: SpatialArgument = ...,
    padding: SpatialArgument = ...,
    dilation: SpatialArgument = ...,
    groups: int = ...,
) -> Variable: ...


@overload
def conv1d(
    inputs: TensorLike,
    kernel: Variable,
    bias: TensorLike | None = ...,
    *,
    stride: SpatialArgument = ...,
    padding: SpatialArgument = ...,
    dilation: SpatialArgument = ...,
    groups: int = ...,
) -> Variable: ...


@overload
def conv1d(
    inputs: TensorLike,
    kernel: TensorLike,
    bias: Variable,
    *,
    stride: SpatialArgument = ...,
    padding: SpatialArgument = ...,
    dilation: SpatialArgument = ...,
    groups: int = ...,
) -> Variable: ...


@overload
def conv1d(
    inputs: TensorData,
    kernel: TensorData,
    bias: TensorData | None = ...,
    *,
    stride: SpatialArgument = ...,
    padding: SpatialArgument = ...,
    dilation: SpatialArgument = ...,
    groups: int = ...,
) -> Tensor: ...


def conv1d(
    inputs: TensorLike,
    kernel: TensorLike,
    bias: TensorLike | None = None,
    *,
    stride: SpatialArgument = 1,
    padding: SpatialArgument = 0,
    dilation: SpatialArgument = 1,
    groups: int = 1,
) -> TensorResult:
    """Correlate a batched or unbatched 1D signal with a kernel.

    Args:
        inputs: Values shaped ``(batch, in_channels, length)`` or
            ``(in_channels, length)``.
        kernel: Weights shaped
            ``(out_channels, in_channels // groups, kernel_length)``.
        bias: Optional per-output-channel offset shaped ``(out_channels,)``.
        stride: Step between successive kernel placements.
        padding: Implicit zeros added to both ends of the spatial axis.
        dilation: Spacing between kernel taps.
        groups: Number of channel groups convolved independently.

    Returns:
        Values shaped ``(batch, out_channels, out_length)`` where
        ``out_length`` is
        ``(length + 2 * padding - dilation * (kernel_length - 1) - 1)
        // stride + 1``.
    """
    return _convolve(
        1, inputs, kernel, bias, stride, padding, dilation, groups
    )


@overload
def conv2d(
    inputs: Variable,
    kernel: TensorLike,
    bias: TensorLike | None = ...,
    *,
    stride: SpatialArgument = ...,
    padding: SpatialArgument = ...,
    dilation: SpatialArgument = ...,
    groups: int = ...,
) -> Variable: ...


@overload
def conv2d(
    inputs: TensorLike,
    kernel: Variable,
    bias: TensorLike | None = ...,
    *,
    stride: SpatialArgument = ...,
    padding: SpatialArgument = ...,
    dilation: SpatialArgument = ...,
    groups: int = ...,
) -> Variable: ...


@overload
def conv2d(
    inputs: TensorLike,
    kernel: TensorLike,
    bias: Variable,
    *,
    stride: SpatialArgument = ...,
    padding: SpatialArgument = ...,
    dilation: SpatialArgument = ...,
    groups: int = ...,
) -> Variable: ...


@overload
def conv2d(
    inputs: TensorData,
    kernel: TensorData,
    bias: TensorData | None = ...,
    *,
    stride: SpatialArgument = ...,
    padding: SpatialArgument = ...,
    dilation: SpatialArgument = ...,
    groups: int = ...,
) -> Tensor: ...


def conv2d(
    inputs: TensorLike,
    kernel: TensorLike,
    bias: TensorLike | None = None,
    *,
    stride: SpatialArgument = 1,
    padding: SpatialArgument = 0,
    dilation: SpatialArgument = 1,
    groups: int = 1,
) -> TensorResult:
    """Correlate a batched or unbatched 2D signal with a kernel.

    The kernel is not reversed, so for one group and unit stride and dilation
    the result is

    .. math::

        y[n, o, i, j] = b[o] + \\sum_{c} \\sum_{u} \\sum_{v}
        x[n, c, i + u, j + v] \\, w[o, c, u, v].

    That is the cross-correlation deep-learning frameworks name
    ``convolution``. True convolution reverses the kernel on its spatial axes;
    reverse ``kernel`` yourself when the signal-processing definition is
    required.

    Args:
        inputs: Values shaped ``(batch, in_channels, height, width)`` or
            ``(in_channels, height, width)``.
        kernel: Weights shaped
            ``(out_channels, in_channels // groups, kernel_height,
            kernel_width)``.
        bias: Optional per-output-channel offset shaped ``(out_channels,)``.
        stride: Step between kernel placements, per axis or shared.
        padding: Implicit zeros added to both ends of each spatial axis.
        dilation: Spacing between kernel taps, per axis or shared.
        groups: Number of channel groups convolved independently. Setting
            ``groups`` to ``in_channels`` gives a depthwise convolution.

    Returns:
        Values shaped ``(batch, out_channels, out_height, out_width)``.
    """
    return _convolve(
        2, inputs, kernel, bias, stride, padding, dilation, groups
    )


@overload
def conv3d(
    inputs: Variable,
    kernel: TensorLike,
    bias: TensorLike | None = ...,
    *,
    stride: SpatialArgument = ...,
    padding: SpatialArgument = ...,
    dilation: SpatialArgument = ...,
    groups: int = ...,
) -> Variable: ...


@overload
def conv3d(
    inputs: TensorLike,
    kernel: Variable,
    bias: TensorLike | None = ...,
    *,
    stride: SpatialArgument = ...,
    padding: SpatialArgument = ...,
    dilation: SpatialArgument = ...,
    groups: int = ...,
) -> Variable: ...


@overload
def conv3d(
    inputs: TensorLike,
    kernel: TensorLike,
    bias: Variable,
    *,
    stride: SpatialArgument = ...,
    padding: SpatialArgument = ...,
    dilation: SpatialArgument = ...,
    groups: int = ...,
) -> Variable: ...


@overload
def conv3d(
    inputs: TensorData,
    kernel: TensorData,
    bias: TensorData | None = ...,
    *,
    stride: SpatialArgument = ...,
    padding: SpatialArgument = ...,
    dilation: SpatialArgument = ...,
    groups: int = ...,
) -> Tensor: ...


def conv3d(
    inputs: TensorLike,
    kernel: TensorLike,
    bias: TensorLike | None = None,
    *,
    stride: SpatialArgument = 1,
    padding: SpatialArgument = 0,
    dilation: SpatialArgument = 1,
    groups: int = 1,
) -> TensorResult:
    """Correlate a batched or unbatched 3D volume with a kernel."""
    return _convolve(
        3, inputs, kernel, bias, stride, padding, dilation, groups
    )


__all__ = ["conv1d", "conv2d", "conv3d"]
