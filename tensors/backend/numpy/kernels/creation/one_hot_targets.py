"""NumPy implementation of dense target expansion."""

from __future__ import annotations
import numpy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.conversion import _storage
from tensors.backend.numpy.conversion import tensor_to_logical_array

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def one_hot_targets(logits: Tensor, targets: Tensor, axis: int) -> Storage | None:
    """Expand validated class indices directly into native dense storage."""
    values = tensor_to_logical_array(targets).astype(numpy.float64, copy=False)
    class_count = logits.shape[axis]
    with _errstate(invalid="ignore"):
        integral = values == numpy.floor(values)
    valid = numpy.all(
        numpy.isfinite(values) & integral & (values >= 0.0) & (values < class_count)
    )
    if not bool(valid):
        return None
    sample_shape = logits.shape[:axis] + logits.shape[axis + 1 :]
    indices = values.astype(numpy.int64).reshape(sample_shape)
    expanded_indices = numpy.expand_dims(indices, axis=axis)
    result = numpy.zeros(logits.shape, dtype=numpy.float64)
    numpy.put_along_axis(result, expanded_indices, 1.0, axis=axis)
    return _storage(result, dtype=logits.dtype, output_shape=logits.shape)
