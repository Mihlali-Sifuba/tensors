"""Differentiable tensor indexing and slicing."""

from typing import List

from ..backend import execute_slice_scatter
from ..shape import Shape
from ..graph.operation import Operation
from ..tensor import Tensor
from ..utils.slicing import (
    logical_linear_indices_from_ranges,
    slice_ranges_and_shape_from_key,
)


def _logical_linear_indices(tensor: Tensor, key) -> tuple[List[int], Shape]:
    """Return selected logical linear indices and the selection shape."""
    keys = key if isinstance(key, tuple) else (key,)
    ranges, selection_shape = slice_ranges_and_shape_from_key(
        keys,
        tensor.shape,
    )

    # Gradients are newly allocated compact tensors, so these are canonical
    # logical linear indices rather than the source tensor's storage indices.
    return (
        logical_linear_indices_from_ranges(ranges, tensor.shape),
        selection_shape,
    )


class Slice(Operation):
    """Tensor indexing with a scatter-style backward pass."""

    __slots__ = ("key",)
    name = "slice"

    def __init__(
        self,
        *,
        key,
    ) -> None:
        object.__setattr__(self, "key", key)

    def forward(self, a: Tensor) -> Tensor:
        key = self.key
        result = a[key]
        if isinstance(result, Tensor):
            return result
        return Tensor(result, dtype=a.dtype, shape=())

    def backward(self, grad: Tensor, *inputs: Tensor) -> List[Tensor]:
        source = inputs[0]
        selected, _ = _logical_linear_indices(source, self.key)
        values = [0.0] * source.size
        for logical_linear_index, grad_value in zip(selected, grad._data):
            values[logical_linear_index] += grad_value
        return [Tensor(values, dtype=grad.dtype, shape=source.shape)]

    def backward_graph(self, grad, *inputs):
        """Build a differentiable scatter for a slice VJP."""
        return [_slice_scatter(grad, inputs[0].shape, self.key)]


class SliceScatter(Operation):
    """Scatter slice-shaped values back into a larger zero tensor."""

    __slots__ = ("source_shape", "key")
    name = "slice_scatter"

    def __init__(self, *, source_shape: tuple[int, ...], key) -> None:
        object.__setattr__(self, "source_shape", source_shape)
        object.__setattr__(self, "key", key)

    def forward(self, grad: Tensor) -> Tensor:
        key = self.key
        source_shape = self.source_shape
        template = Tensor(
            [0.0] * Shape.from_iterable(source_shape).size,
            dtype=grad.dtype,
            shape=source_shape,
        )
        selected, selection_shape = _logical_linear_indices(template, key)
        if selection_shape.size != grad.size:
            raise ValueError(
                f"Slice gradient has {grad.size} values; "
                f"expected {selection_shape.size}"
            )
        accelerated = execute_slice_scatter(
            grad,
            selected,
            output_shape=source_shape,
        )
        if accelerated is not None:
            return Tensor._from_owned_storage(
                accelerated,
                dtype=grad.dtype,
                shape=source_shape,
            )
        values = [0.0] * template.size
        for logical_linear_index, grad_value in zip(selected, grad._data):
            values[logical_linear_index] += grad_value
        return Tensor(values, dtype=grad.dtype, shape=source_shape)

    def backward(self, grad: Tensor, *inputs: Tensor) -> List[Tensor]:
        return [Slice(key=self.key).forward(grad)]

    def backward_graph(self, grad, *inputs):
        return [grad[self.key]]


def _slice_scatter(grad, source_shape: tuple[int, ...], key):
    """Return a differentiable slice-scatter Variable."""
    from ..variable import Variable

    operation = SliceScatter(source_shape=source_shape, key=key)
    return Variable._from_operation(
        operation.forward(grad.data),
        operation,
        (grad,),
    )
