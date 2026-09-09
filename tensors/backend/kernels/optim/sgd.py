"""Stochastic gradient descent updates."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

from ...storage import Storage
from ..core import (
    _array_kind,
    _errstate,
    _finite_operands,
    _numpy,
    _storage,
    _view,
)
from .batching import (
    _optimizer_batch_partitions,
    _optimizer_batch_values,
    _optimizer_invalid_flag,
    _optimizer_partition,
    _split_optimizer_storage,
)
from .cuda import _cuda_sgd_batch_kernel

if TYPE_CHECKING:
    from ....tensor import Tensor

def sgd_update(
    parameter: Tensor,
    gradient: Tensor,
    learning_rate: float,
) -> Storage | None:
    """Apply one fused SGD update."""
    numpy = _numpy()
    values = _view(parameter, numpy).astype(numpy.float64, copy=False)
    gradients = _view(gradient, numpy).astype(numpy.float64, copy=False)
    if not _finite_operands(values, gradients, numpy=numpy):
        return None
    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        result = values - learning_rate * gradients
    if not bool(numpy.all(numpy.isfinite(result))):
        return None
    return _storage(
        result,
        dtype=parameter.dtype,
        output_shape=parameter.shape,
        numpy=numpy,
    )

def sgd_updates(
    parameters: Sequence[Tensor],
    gradients: Sequence[Tensor],
    learning_rate: float,
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
        return tuple(cast(Storage, storage) for storage in merged)
    numpy = _numpy()
    values = _optimizer_batch_values(parameters, numpy, slot="input:0")
    gradient_values = _optimizer_batch_values(
        gradients,
        numpy,
        slot="input:1",
    )
    if values is None or gradient_values is None:
        return None
    if _array_kind(numpy) == "cuda":
        result = numpy.empty_like(values)
        invalid = _optimizer_invalid_flag(numpy)
        _cuda_sgd_batch_kernel()(
            values,
            gradient_values,
            learning_rate,
            result,
            invalid,
        )
        if bool(invalid[0]):
            return None
    else:
        with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
            result = values - learning_rate * gradient_values
        valid = (
            numpy.all(numpy.isfinite(values))
            & numpy.all(numpy.isfinite(gradient_values))
            & numpy.all(numpy.isfinite(result))
        )
        if not bool(valid):
            return None
    return _split_optimizer_storage(result, parameters, numpy)
