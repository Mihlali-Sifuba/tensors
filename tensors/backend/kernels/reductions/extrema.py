"""Index-of-extremum reductions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...storage import Storage
from ..core import _numpy, _storage, _view

if TYPE_CHECKING:
    from ....tensor import Tensor
    from ...types import ArgExtremumOperation

def arg_extremum(
    operation: ArgExtremumOperation,
    value: Tensor,
    axis: int | None,
    *,
    keepdims: bool,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Run a first-occurrence argmin or argmax reduction."""
    if value.size == 0:
        return None
    from ....dtype import int64

    numpy = _numpy()
    values = _view(value, numpy)
    function = numpy.argmin if operation == "argmin" else numpy.argmax
    result = function(values, axis=axis, keepdims=keepdims)
    return _storage(
        result,
        dtype=int64,
        output_shape=output_shape,
        numpy=numpy,
    )
