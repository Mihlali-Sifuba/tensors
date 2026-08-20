"""Differentiable tensor indexing and slicing."""

from typing import List

from ..tensor import Tensor
from ..utils.shape import shape_size


def _flat_indices(tensor: Tensor, key) -> List[int]:
    """Return source flat indices selected by a supported tensor key."""
    keys = key if isinstance(key, tuple) else (key,)
    if len(keys) > tensor.ndim:
        raise IndexError(f"Too many indices: {len(keys)} for {tensor.ndim}D tensor")
    keys = keys + (slice(None),) * (tensor.ndim - len(keys))

    ranges = []
    for dim, part in enumerate(keys):
        if isinstance(part, int):
            index = part if part >= 0 else part + tensor.shape[dim]
            if not 0 <= index < tensor.shape[dim]:
                raise IndexError("Index out of range")
            ranges.append(range(index, index + 1))
        elif isinstance(part, slice):
            ranges.append(range(*part.indices(tensor.shape[dim])))
        else:
            raise TypeError(f"Unsupported index type: {type(part)}")

    strides = []
    for dim in range(tensor.ndim):
        stride = 1
        for trailing in tensor.shape[dim + 1:]:
            stride *= trailing
        strides.append(stride)

    selected = []

    def collect(dim, offset):
        if dim == tensor.ndim:
            selected.append(offset)
            return
        for index in ranges[dim]:
            collect(dim + 1, offset + index * strides[dim])

    collect(0, 0)
    return selected


class Slice:
    """Tensor indexing with a scatter-style backward pass."""

    @staticmethod
    def forward(a: Tensor, key) -> Tensor:
        result = a[key]
        if isinstance(result, Tensor):
            return result
        return Tensor(result, dtype=a.dtype)

    @staticmethod
    def backward(grad: Tensor, *inputs: Tensor, **kwargs: object) -> List[Tensor]:
        source = inputs[0]
        selected = _flat_indices(source, kwargs["key"])
        values = [0.0] * source.size
        for flat_index, grad_value in zip(selected, grad._data):
            values[flat_index] += grad_value
        return [Tensor(values, dtype=grad.dtype, shape=source.shape)]

    @staticmethod
    def backward_graph(grad, *inputs, **kwargs: object):
        """Build a differentiable scatter for a slice VJP."""
        return [_slice_scatter(grad, inputs[0].shape, kwargs["key"])]


class SliceScatter:
    """Scatter slice-shaped values back into a larger zero tensor."""

    @staticmethod
    def forward(grad: Tensor, source_shape: tuple[int, ...], key) -> Tensor:
        template = Tensor([0.0] * shape_size(source_shape), dtype=grad.dtype, shape=source_shape)
        selected = _flat_indices(template, key)
        if len(selected) != grad.size:
            raise ValueError(
                f"Slice gradient has {grad.size} values; expected {len(selected)}"
            )
        values = [0.0] * template.size
        for flat_index, grad_value in zip(selected, grad._data):
            values[flat_index] += grad_value
        return Tensor(values, dtype=grad.dtype, shape=source_shape)

    @staticmethod
    def backward(grad: Tensor, *inputs: Tensor, **kwargs: object) -> List[Tensor]:
        return [Slice.forward(grad, kwargs["key"])]

    @staticmethod
    def backward_graph(grad, *inputs, **kwargs: object):
        return [grad[kwargs["key"]]]


def _slice_scatter(grad, source_shape: tuple[int, ...], key):
    """Return a differentiable slice-scatter Variable."""
    from ..variable import Variable

    return Variable._from_operation(
        SliceScatter.forward(grad.data, source_shape, key),
        "slice_scatter",
        SliceScatter,
        [grad],
        source_shape=source_shape,
        key=key,
    )
