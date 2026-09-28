"""Reference multiclass cross-entropy for the Python backend."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from tensors.backend.python.kernels.nn._normalization import _floating_dtype
from tensors.backend.python.kernels.nn.log_softmax import log_softmax
from tensors.backend.python.storage import PythonStorage
from tensors.utils.reductions import reduction_groups
from tensors.utils.summation import stable_float_mean, stable_float_sum

if TYPE_CHECKING:
    from tensors.backend.types import LossReduction
    from tensors.dtype import DataType


def cross_entropy(
    logits_values: Sequence[Any],
    target_values: Sequence[Any],
    logits_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
    axis: int,
    *,
    reduction: LossReduction,
    dtype: DataType,
    logits_dtype: DataType,
    output_shape: tuple[int, ...],
) -> PythonStorage | None:
    """Sum zero-safe negative target times log-softmax over the class axis."""
    if logits_shape != target_shape:
        return None
    log_probabilities = log_softmax(
        logits_values,
        logits_shape,
        axis,
        dtype=_floating_dtype(logits_dtype),
    ).buffer
    _, _, groups = reduction_groups(logits_shape, axis, keepdims=False)
    losses = []
    for group in groups:
        contributions = [
            -float(target_values[index]) * float(log_probabilities[index])
            for index in group
            if target_values[index] != 0
        ]
        losses.append(stable_float_sum(contributions))
    if reduction == "none":
        result = losses
    elif reduction == "mean":
        result = [stable_float_mean(losses)]
    else:
        result = [stable_float_sum(losses)]
    storage = PythonStorage.from_values(result, dtype)
    if storage.size != math.prod(output_shape):
        raise RuntimeError("Python loss kernel returned an unexpected result size")
    return storage
