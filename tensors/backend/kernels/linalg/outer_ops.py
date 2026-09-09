"""Vector outer products and their vector-Jacobian products."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...storage import Storage
from ..core import (
    _errstate,
    _finite_operands,
    _numpy,
    _operand,
    _storage,
    _view,
)
from ..reductions.stability import _stable_sum_candidate

if TYPE_CHECKING:
    from ....dtype import DataType
    from ....tensor import Tensor

def outer(
    left: Tensor,
    right: Tensor,
    *,
    dtype: DataType,
) -> Storage | None:
    """Run a vector outer product."""
    numpy = _numpy()
    try:
        left_values = _operand(left, dtype, numpy)
        right_values = _operand(right, dtype, numpy)
    except (TypeError, ValueError):
        return None
    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        result = numpy.multiply.outer(left_values, right_values)
    return _storage(
        result,
        dtype=dtype,
        output_shape=(left.size, right.size),
        numpy=numpy,
    )

def outer_gradient(
    grad: Tensor,
    left: Tensor,
    right: Tensor,
    *,
    needs_input_grad: tuple[bool, ...] = (True, True),
) -> tuple[Storage | None, Storage | None] | None:
    """Run the requested stable native outer-product VJPs."""
    numpy = _numpy()
    upstream = _view(grad, numpy).astype(numpy.float64, copy=False)
    left_values = _view(left, numpy).astype(numpy.float64, copy=False)
    right_values = _view(right, numpy).astype(numpy.float64, copy=False)
    if not _finite_operands(
        upstream,
        left_values,
        right_values,
        numpy=numpy,
    ):
        return None
    need_left, need_right = needs_input_grad
    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        left_terms = upstream * right_values if need_left else None
        right_terms = upstream * left_values[:, None] if need_right else None
    if left_terms is not None and not _stable_sum_candidate(
        left_terms, (1,), numpy
    ):
        return None
    if right_terms is not None and not _stable_sum_candidate(
        right_terms, (0,), numpy
    ):
        return None
    left_storage = None
    if left_terms is not None:
        left_storage = _storage(
            numpy.sum(left_terms, axis=1),
            dtype=grad.dtype,
            output_shape=left.shape,
            numpy=numpy,
        )
        if left_storage is None:
            return None
    right_storage = None
    if right_terms is not None:
        right_storage = _storage(
            numpy.sum(right_terms, axis=0),
            dtype=grad.dtype,
            output_shape=right.shape,
            numpy=numpy,
        )
        if right_storage is None:
            return None
    return left_storage, right_storage
