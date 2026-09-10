"""Interval clamping and its vector-Jacobian product."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...storage import Storage
from ..core import _numpy, _storage, _view

if TYPE_CHECKING:
    from ....dtype import DataType
    from ....tensor import Tensor

def clip(
    value: Tensor,
    min_value: int | float | None,
    max_value: int | float | None,
    *,
    dtype: DataType,
) -> Storage | None:
    """Clip tensor values to optional scalar bounds."""
    numpy = _numpy()
    values = _view(value, numpy)
    result = numpy.clip(values, min_value, max_value)
    return _storage(
        result,
        dtype=dtype,
        output_shape=value.shape,
        numpy=numpy,
    )

def clip_gradient(
    grad: Tensor,
    value: Tensor,
    min_value: int | float | None,
    max_value: int | float | None,
) -> Storage | None:
    """Run the clipping VJP with zero boundary subgradients."""
    numpy = _numpy()
    values = _view(value, numpy).astype(numpy.float64, copy=False)
    upstream = _view(grad, numpy).astype(numpy.float64, copy=False)
    mask = numpy.ones(value.shape, dtype=bool)
    if min_value is not None:
        mask &= values > min_value
    if max_value is not None:
        mask &= values < max_value
    result = numpy.where(numpy.isnan(values), numpy.nan, numpy.where(mask, upstream, 0.0))
    return _storage(
        result,
        dtype=grad.dtype,
        output_shape=value.shape,
        numpy=numpy,
    )
