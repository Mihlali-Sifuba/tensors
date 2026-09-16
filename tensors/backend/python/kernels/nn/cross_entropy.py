"""Reference multiclass cross-entropy for the Python backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tensors.backend.python.kernels.nn._normalization import _log_softmax_values
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.math._reduction import reduction_groups
from tensors.math.mean import _stable_float_mean
from tensors.math.sum import _stable_float_sum

if TYPE_CHECKING:
    from tensors.backend.types import LossReduction
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


def cross_entropy(
    logits: Tensor,
    targets: Tensor,
    axis: int,
    *,
    reduction: LossReduction,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Sum ``-target * log_softmax(logits)`` over ``axis``, then reduce.

    Zero-weighted classes are skipped rather than multiplied, so a target of
    zero contributes nothing even where the log probability is ``-inf``.
    """
    log_probabilities = _log_softmax_values(logits, axis)
    _, _, groups = reduction_groups(logits, axis, keepdims=False)
    losses = []
    for group in groups:
        contributions = [
            -float(targets._data[index]) * log_probabilities[index]
            for index in group
            if targets._data[index] != 0
        ]
        losses.append(_stable_float_sum(contributions))
    if reduction == "none":
        return PythonStorage.from_values(losses, dtype)
    if reduction == "mean":
        total = _stable_float_mean(losses)
    else:
        total = _stable_float_sum(losses)
    return PythonStorage.from_values([total], dtype)
