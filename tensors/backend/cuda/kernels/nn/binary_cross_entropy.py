"""CuPy implementation of binary cross-entropy."""

from __future__ import annotations
import cupy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.cuda.conversion import _errstate
from tensors.backend.cuda.conversion import _finite_operands
from tensors.backend.cuda.conversion import _storage
from tensors.backend.cuda.conversion import _working_values

if TYPE_CHECKING:
    from tensors.dtype import DataType
    from tensors.tensor import Tensor
    from tensors.backend.types import LossReduction
from tensors.backend.cuda.kernels.nn.losses import _reduce_losses


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
    values = _working_values(prediction)
    targets = _working_values(target)
    if values.shape != targets.shape:
        return None
    if from_logits:
        if not _finite_operands(values, targets):
            return None
        with _errstate(over="ignore", under="ignore", invalid="ignore"):
            correction = cupy.log1p(cupy.exp(-cupy.abs(values)))
            losses = cupy.where(
                values >= 0.0,
                (1.0 - targets) * values + correction,
                -targets * values + correction,
            )
    else:
        invalid = ~cupy.isfinite(values) | ((values < 0.0) | (values > 1.0))
        target_valid = cupy.all(cupy.isfinite(targets))
        value_valid = ~cupy.any(invalid)
        status = int(
            cupy.asarray(~target_valid, dtype=cupy.uint8)
            | cupy.asarray(~value_valid, dtype=cupy.uint8) * cupy.uint8(2)
        )
        if status & 1:
            return None
        if status & 2:
            raise ValueError(
                "binary cross-entropy probabilities must be between 0 and 1"
            )
        with _errstate(divide="ignore", invalid="ignore"):
            losses = cupy.where(
                values == 0.0,
                cupy.where(targets == 0.0, 0.0, cupy.inf),
                cupy.where(
                    values == 1.0,
                    cupy.where(targets == 1.0, 0.0, cupy.inf),
                    -targets * cupy.log(values) - (1.0 - targets) * cupy.log1p(-values),
                ),
            )
    result = _reduce_losses(losses, reduction)
    return _storage(result, dtype=dtype, output_shape=output_shape)
