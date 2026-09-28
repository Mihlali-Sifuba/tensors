"""Cross-correlation convolution over one, two, or three spatial axes.

The public functions accept batched or unbatched channel-first inputs and
slide a kernel over their spatial axes without reversing it. That is the
operation deep-learning frameworks call convolution; see :func:`conv2d` for
the exact definition and its relationship to true convolution.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any, List, Optional, overload
from tensors._typing import TensorData, TensorLike, TensorResult
from tensors.backend import execute_convolution, execute_convolution_gradient
from tensors.dtype import result_dtype
from tensors.operations.base import Operation
from tensors.tensor import Tensor
from tensors.utils.convolution import (
    Geometry as _Geometry,
    SpatialArgument,
    contributions as _contributions,
    resolve_geometry,
    spatial_argument as _spatial_argument,
)
from tensors.graph.expression import as_tensor_operand

if TYPE_CHECKING:
    from tensors.graph.node import VariableNode
    from tensors.variable import Variable


def _as_tensor(value: TensorLike) -> Tensor:
    """Return the Tensor value behind any accepted convolution operand."""
    from tensors.variable import Variable

    if isinstance(value, Tensor):
        return value
    if isinstance(value, Variable):
        return value.data
    return as_tensor_operand(value)


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
        self, inputs: Tensor, kernel: Tensor, bias: Tensor | None = None
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
        return Tensor._from_owned_storage(
            accelerated, dtype=dtype, shape=geometry.output_shape
        )

    def _geometry_for(
        self, inputs: Tensor, kernel: Tensor, bias: Tensor | None
    ) -> "_Geometry":
        """Resolve this invocation's spatial geometry for given operands."""
        return resolve_geometry(
            self.rank,
            inputs.shape,
            kernel.shape,
            bias.shape if bias is not None else None,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def backward(
        self, grad: Tensor, *inputs: Tensor, needs_input_grad: tuple[bool, ...]
    ) -> List[Optional[Tensor]]:
        """Differentiate a convolution with respect to its requested inputs."""
        values, kernel = (inputs[0], inputs[1])
        bias = inputs[2] if len(inputs) > 2 else None
        geometry = resolve_geometry(
            self.rank,
            values.shape,
            kernel.shape,
            bias.shape if bias is not None else None,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        if tuple(grad.shape) != geometry.output_shape:
            raise ValueError(
                f"Gradient shape {tuple(grad.shape)} does not match output shape {geometry.output_shape}"
            )
        shapes: list[tuple[int, ...]] = [tuple(values.shape), tuple(kernel.shape)]
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
        return [
            (
                Tensor._from_owned_storage(storage, dtype=grad.dtype, shape=shape)
                if storage is not None
                else None
            )
            for storage, shape in zip(accelerated, shapes)
        ]


def _convolve(
    rank: int,
    inputs: TensorLike | VariableNode,
    kernel: TensorLike | VariableNode,
    bias: TensorLike | VariableNode | None,
    stride: SpatialArgument,
    padding: SpatialArgument,
    dilation: SpatialArgument,
    groups: int,
) -> TensorResult | VariableNode:
    """Dispatch a convolution over graph values or Tensors.

    Every rank reaches this one boundary, so a graph value in any
    operand position is answered here: Variables calculate the result
    now, a vertex records the operation for a program that runs later,
    and a Tensor beside either enters the graph as a non-gradient leaf.
    An absent bias stays absent rather than becoming a placeholder
    operand, which is the shape ConvND.forward reads.
    """
    from tensors.graph.expression import (
        apply_operation,
        as_graph_operand,
        is_graph_operand,
    )

    strides = _spatial_argument(stride, rank, "stride", minimum=1)
    paddings = _spatial_argument(padding, rank, "padding", minimum=0)
    dilations = _spatial_argument(dilation, rank, "dilation", minimum=1)
    operands: list[TensorLike | VariableNode] = [inputs, kernel]
    if bias is not None:
        operands.append(bias)
    operation = ConvND(
        rank=rank, stride=strides, padding=paddings, dilation=dilations, groups=groups
    )
    if any((is_graph_operand(operand) for operand in operands)):
        return apply_operation(
            operation, tuple((as_graph_operand(operand) for operand in operands))
        )
    tensors = [_as_tensor(operand) for operand in operands]
    return operation.forward(
        tensors[0], tensors[1], tensors[2] if len(tensors) > 2 else None
    )


@overload
def conv1d(
    inputs: VariableNode,
    kernel: TensorLike | VariableNode,
    bias: TensorLike | VariableNode | None = ...,
    *,
    stride: SpatialArgument = ...,
    padding: SpatialArgument = ...,
    dilation: SpatialArgument = ...,
    groups: int = ...,
) -> VariableNode: ...


@overload
def conv1d(
    inputs: TensorLike,
    kernel: VariableNode,
    bias: TensorLike | VariableNode | None = ...,
    *,
    stride: SpatialArgument = ...,
    padding: SpatialArgument = ...,
    dilation: SpatialArgument = ...,
    groups: int = ...,
) -> VariableNode: ...


@overload
def conv1d(
    inputs: TensorLike,
    kernel: TensorLike,
    bias: VariableNode,
    *,
    stride: SpatialArgument = ...,
    padding: SpatialArgument = ...,
    dilation: SpatialArgument = ...,
    groups: int = ...,
) -> VariableNode: ...


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
    inputs: TensorLike | VariableNode,
    kernel: TensorLike | VariableNode,
    bias: TensorLike | VariableNode | None = None,
    *,
    stride: SpatialArgument = 1,
    padding: SpatialArgument = 0,
    dilation: SpatialArgument = 1,
    groups: int = 1,
) -> TensorResult | VariableNode:
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
    return _convolve(1, inputs, kernel, bias, stride, padding, dilation, groups)


@overload
def conv2d(
    inputs: VariableNode,
    kernel: TensorLike | VariableNode,
    bias: TensorLike | VariableNode | None = ...,
    *,
    stride: SpatialArgument = ...,
    padding: SpatialArgument = ...,
    dilation: SpatialArgument = ...,
    groups: int = ...,
) -> VariableNode: ...


@overload
def conv2d(
    inputs: TensorLike,
    kernel: VariableNode,
    bias: TensorLike | VariableNode | None = ...,
    *,
    stride: SpatialArgument = ...,
    padding: SpatialArgument = ...,
    dilation: SpatialArgument = ...,
    groups: int = ...,
) -> VariableNode: ...


@overload
def conv2d(
    inputs: TensorLike,
    kernel: TensorLike,
    bias: VariableNode,
    *,
    stride: SpatialArgument = ...,
    padding: SpatialArgument = ...,
    dilation: SpatialArgument = ...,
    groups: int = ...,
) -> VariableNode: ...


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
    inputs: TensorLike | VariableNode,
    kernel: TensorLike | VariableNode,
    bias: TensorLike | VariableNode | None = None,
    *,
    stride: SpatialArgument = 1,
    padding: SpatialArgument = 0,
    dilation: SpatialArgument = 1,
    groups: int = 1,
) -> TensorResult | VariableNode:
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
    return _convolve(2, inputs, kernel, bias, stride, padding, dilation, groups)


@overload
def conv3d(
    inputs: VariableNode,
    kernel: TensorLike | VariableNode,
    bias: TensorLike | VariableNode | None = ...,
    *,
    stride: SpatialArgument = ...,
    padding: SpatialArgument = ...,
    dilation: SpatialArgument = ...,
    groups: int = ...,
) -> VariableNode: ...


@overload
def conv3d(
    inputs: TensorLike,
    kernel: VariableNode,
    bias: TensorLike | VariableNode | None = ...,
    *,
    stride: SpatialArgument = ...,
    padding: SpatialArgument = ...,
    dilation: SpatialArgument = ...,
    groups: int = ...,
) -> VariableNode: ...


@overload
def conv3d(
    inputs: TensorLike,
    kernel: TensorLike,
    bias: VariableNode,
    *,
    stride: SpatialArgument = ...,
    padding: SpatialArgument = ...,
    dilation: SpatialArgument = ...,
    groups: int = ...,
) -> VariableNode: ...


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
    inputs: TensorLike | VariableNode,
    kernel: TensorLike | VariableNode,
    bias: TensorLike | VariableNode | None = None,
    *,
    stride: SpatialArgument = 1,
    padding: SpatialArgument = 0,
    dilation: SpatialArgument = 1,
    groups: int = 1,
) -> TensorResult | VariableNode:
    """Correlate a batched or unbatched 3D volume with a kernel."""
    return _convolve(3, inputs, kernel, bias, stride, padding, dilation, groups)


__all__ = ["conv1d", "conv2d", "conv3d"]
