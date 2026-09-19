"""NumPy implementation of the batched SGD parameter update."""

from __future__ import annotations
import numpy
from collections.abc import Sequence
from typing import TYPE_CHECKING
from typing import cast
from tensors.backend.storage import Storage
from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.kernels.optim.batching import _optimizer_batch_partitions
from tensors.backend.numpy.kernels.optim.batching import _optimizer_batch_values
from tensors.backend.numpy.kernels.optim.batching import _optimizer_partition
from tensors.backend.numpy.kernels.optim.batching import _split_optimizer_storage

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def sgd_updates(
    parameters: Sequence[Tensor], gradients: Sequence[Tensor], learning_rate: float
) -> tuple[Storage, ...] | None:
    """Apply one native SGD update to several compatible parameters."""
    partitions = _optimizer_batch_partitions(parameters, gradients)
    if partitions is None:
        return None
    if len(partitions) > 1:
        merged: list[Storage | None] = [None] * len(parameters)
        for indices in partitions:
            result = sgd_updates(
                _optimizer_partition(parameters, indices),
                _optimizer_partition(gradients, indices),
                learning_rate,
            )
            if result is None:
                return None
            for index, storage in zip(indices, result):
                merged[index] = storage
        return tuple((cast(Storage, storage) for storage in merged))
    values = _optimizer_batch_values(parameters, slot="input:0")
    gradient_values = _optimizer_batch_values(gradients, slot="input:1")
    if values is None or gradient_values is None:
        return None
    with _errstate(over="ignore", under="ignore", invalid="ignore"):
        result = values - learning_rate * gradient_values
    valid = (
        numpy.all(numpy.isfinite(values))
        & numpy.all(numpy.isfinite(gradient_values))
        & numpy.all(numpy.isfinite(result))
    )
    if not bool(valid):
        return None
    return _split_optimizer_storage(result, parameters)
