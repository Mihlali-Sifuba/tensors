"""CuPy implementation of dense target expansion."""

from __future__ import annotations
import cupy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.cuda.conversion import _errstate
from tensors.backend.cuda.conversion import _storage
from tensors.backend.cuda.conversion import _working_values

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def one_hot_targets(logits: Tensor, targets: Tensor, axis: int) -> Storage | None:
    """Expand validated class indices directly into native dense storage."""
    values = _working_values(targets)
    class_count = logits.shape[axis]
    with _errstate(invalid="ignore"):
        integral = values == cupy.floor(values)
    valid = cupy.all(
        cupy.isfinite(values) & integral & (values >= 0.0) & (values < class_count)
    )
    if not bool(valid):
        return None
    sample_shape = logits.shape[:axis] + logits.shape[axis + 1 :]
    indices = values.astype(cupy.int64).reshape(sample_shape)
    expanded_indices = cupy.expand_dims(indices, axis=axis)
    result = cupy.zeros(logits.shape, dtype=cupy.float64)
    cupy.put_along_axis(result, expanded_indices, 1.0, axis=axis)
    return _storage(result, dtype=logits.dtype, output_shape=logits.shape)
