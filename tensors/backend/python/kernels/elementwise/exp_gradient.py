"""Reference the exponential VJP for the Python backend."""

from __future__ import annotations
import math
from collections.abc import Iterable
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import DataType

_INFINITY = float("inf")


def exp_gradient(
    grad_values: Iterable[float],
    values: Iterable[float],
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Scale the upstream gradient by ``exp(x)``.

    Exp is its own derivative, so this evaluates the same function the
    forward does and multiplies. The overflow translation is the forward's
    too: an overflowing derivative is ``+inf``, and the product that follows
    is then an ordinary IEEE multiplication — a zero upstream against an
    infinite derivative gives NaN, which is the honest answer where the
    product has no limit.
    """
    result = []
    for upstream, item in zip(grad_values, values):
        try:
            derivative = math.exp(float(item))
        except OverflowError:
            derivative = _INFINITY
        result.append(upstream * derivative)
    if len(result) != math.prod(output_shape):
        raise RuntimeError("Exp VJP kernel returned an unexpected result size")
    return PythonStorage.from_values(result, dtype)
