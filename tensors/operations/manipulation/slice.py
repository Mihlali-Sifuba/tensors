"""Differentiable tensor indexing and slicing."""

from __future__ import annotations

from typing import Any, List, Optional

from tensors.backend import execute_slice_scatter
from tensors.operations.base import Operation
from tensors.tensor import Tensor
from tensors.utils.slicing import (
    logical_linear_indices_from_ranges,
    slice_ranges_and_shape_from_key,
)


def _logical_linear_indices(shape, key):
    """Return selected logical linear indices and the selection shape."""
    keys = key if isinstance(key, tuple) else (key,)
    ranges, selection_shape = slice_ranges_and_shape_from_key(keys, shape)
    return (logical_linear_indices_from_ranges(ranges, shape), selection_shape)


class Slice(Operation):
    """Tensor indexing with a scatter-style backward pass."""

    __slots__ = ("key",)
    name = "slice"

    def __init__(self, *, key) -> None:
        object.__setattr__(self, "key", key)

    def forward(self, a: Tensor) -> Tensor:
        key = self.key
        result = a[key]
        if isinstance(result, Tensor):
            return result
        return Tensor(result, dtype=a.dtype, shape=())

    def backward(
        self, grad: Tensor, *inputs: Tensor, needs_input_grad: tuple[bool, ...]
    ) -> List[Optional[Any]]:
        """Scatter the requested slice gradient through the selected backend."""
        if not needs_input_grad[0]:
            return [None]
        source_shape = tuple(inputs[0].shape)
        if isinstance(grad, Tensor):
            return [SliceScatter(source_shape=source_shape, key=self.key).forward(grad)]
        return [_slice_scatter(grad, source_shape, self.key)]


class SliceScatter(Operation):
    """Scatter slice-shaped values back into a larger zero tensor."""

    __slots__ = ("source_shape", "key")
    name = "slice_scatter"

    def __init__(self, *, source_shape: tuple[int, ...], key) -> None:
        object.__setattr__(self, "source_shape", source_shape)
        object.__setattr__(self, "key", key)

    def forward(self, grad: Tensor) -> Tensor:
        source_shape = self.source_shape
        selected, selection_shape = _logical_linear_indices(source_shape, self.key)
        if selection_shape.size != grad.size:
            raise ValueError(
                f"Slice gradient has {grad.size} values; expected {selection_shape.size}"
            )
        accelerated = execute_slice_scatter(grad, selected, output_shape=source_shape)
        return Tensor._from_owned_storage(
            accelerated, dtype=grad.dtype, shape=source_shape
        )

    def backward(
        self, grad: Tensor, *inputs: Tensor, needs_input_grad: tuple[bool, ...]
    ) -> List[Optional[Any]]:
        """Select the requested cotangent, preserving graph construction."""
        if not needs_input_grad[0]:
            return [None]
        if isinstance(grad, Tensor):
            return [Slice(key=self.key).forward(grad)]
        return [grad[self.key]]


def _slice_scatter(grad, source_shape: tuple[int, ...], key):
    """Return a differentiable slice-scatter Variable."""
    from tensors.variable import Variable

    operation = SliceScatter(source_shape=source_shape, key=key)
    return Variable._apply_operation(operation, (grad,))
