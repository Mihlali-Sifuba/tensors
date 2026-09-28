"""Reference the square root VJP for the Python backend."""

from __future__ import annotations
from array import array
from collections.abc import Iterable
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import DataType
import math


def _rounded(value: float, typecode: str) -> float:
    """Round one intermediate into the declared format.

    `docs/sqrt-semantics.md` section 6 fixes the evaluation order and
    requires each step to round in the declared dtype, so a ``float32`` VJP
    is not evaluated in binary64 and narrowed once at the end. Python has
    only binary64 arithmetic, so each step is rounded through a typed buffer
    of the declared width. A step that leaves the format's range rounds to
    the infinity it overflowed towards, which is what the format's own
    arithmetic would produce.
    """
    if typecode != "f":
        return value
    try:
        return array("f", [value])[0]
    except OverflowError:
        return math.inf if value > 0 else -math.inf


def sqrt_gradient(
    grad_values: Iterable[float],
    values: Iterable[float],
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Scale the upstream gradient by ``1 / (2 * sqrt(x))``.

    A negative primal is a value rather than an error and gives NaN, in
    keeping with the forward rule; only a zero primal raises. See
    docs/sqrt-semantics.md section 6.
    """
    typecode = dtype.typecode
    result = []
    for upstream, item in zip(grad_values, values):
        if item == 0:
            raise ValueError("sqrt derivative is undefined at zero")
        if item < 0 or (isinstance(item, float) and math.isnan(item)):
            result.append(math.nan)
            continue
        root = _rounded(math.sqrt(item), typecode)
        denominator = _rounded(2.0 * root, typecode)
        result.append(_rounded(upstream / denominator, typecode))
    if len(result) != math.prod(output_shape):
        raise RuntimeError("Sqrt VJP kernel returned an unexpected result size")
    return PythonStorage.from_values(result, dtype)
