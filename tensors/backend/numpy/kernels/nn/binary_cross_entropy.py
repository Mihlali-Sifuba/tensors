"""NumPy implementation of binary cross-entropy."""

from __future__ import annotations
import numpy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.conversion import _finite_operands
from tensors.backend.numpy.conversion import _storage
from tensors.backend.numpy.conversion import tensor_to_logical_array

if TYPE_CHECKING:
    from tensors.dtype import DataType
    from tensors.tensor import Tensor
    from tensors.backend.types import LossReduction
from tensors.backend.numpy.kernels.nn.losses import _reduce_losses


def binary_cross_entropy(
    prediction: Tensor,
    target: Tensor,
    *,
    from_logits: bool,
    reduction: LossReduction,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Run fused binary cross-entropy."""
    values = tensor_to_logical_array(prediction).astype(numpy.float64, copy=False)
    targets = tensor_to_logical_array(target).astype(numpy.float64, copy=False)
    if values.shape != targets.shape:
        return None
    if from_logits:
        if not _finite_operands(values, targets):
            return None
        with _errstate(over="ignore", under="ignore", invalid="ignore"):
            correction = numpy.log1p(numpy.exp(-numpy.abs(values)))
            losses = numpy.where(
                values >= 0.0,
                (1.0 - targets) * values + correction,
                -targets * values + correction,
            )
    else:
        invalid = ~numpy.isfinite(values) | ((values < 0.0) | (values > 1.0))
        target_valid = numpy.all(numpy.isfinite(targets))
        value_valid = ~numpy.any(invalid)
        status = int(
            numpy.asarray(~target_valid, dtype=numpy.uint8)
            | numpy.asarray(~value_valid, dtype=numpy.uint8) * numpy.uint8(2)
        )
        if status & 1:
            return None
        if status & 2:
            raise ValueError(
                "binary cross-entropy probabilities must be between 0 and 1"
            )
        with _errstate(divide="ignore", invalid="ignore"):
            losses = numpy.where(
                values == 0.0,
                numpy.where(targets == 0.0, 0.0, numpy.inf),
                numpy.where(
                    values == 1.0,
                    numpy.where(targets == 1.0, 0.0, numpy.inf),
                    -targets * numpy.log(values)
                    - (1.0 - targets) * numpy.log1p(-values),
                ),
            )
    result = _reduce_losses(losses, reduction)
    return _storage(result, dtype=dtype, output_shape=output_shape)
