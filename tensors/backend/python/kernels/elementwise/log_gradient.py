"""Reference the natural logarithm VJP for the Python backend."""

from __future__ import annotations
import math
from collections.abc import Iterable
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import DataType

_INFINITY = float("inf")
_NAN = float("nan")


def log_gradient(
    grad_values: Iterable[float],
    values: Iterable[float],
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Scale the upstream gradient by ``1 / x``.

    This is an ordinary division and
    `docs/arithmetic-semantics.md` section 7.2 governs it: Python raises
    where IEEE 754 delivers a value, so a zero primal gives a signed
    infinity and ``0 / 0`` gives NaN, exactly as the division kernel does. A
    primal reaching this from a completed forward pass is positive — the
    forward refuses anything else — so the zero and negative cases are
    reachable only by invoking the VJP directly, and they divide rather than
    being refused a second time.
    """
    result = []
    for upstream, item in zip(grad_values, values):
        if item == 0:
            if upstream == 0 or upstream != upstream:
                result.append(_NAN)
            else:
                result.append(
                    math.copysign(
                        _INFINITY,
                        math.copysign(1.0, upstream) * math.copysign(1.0, item),
                    )
                )
            continue
        result.append(upstream / item)
    if len(result) != math.prod(output_shape):
        raise RuntimeError("Log VJP kernel returned an unexpected result size")
    return PythonStorage.from_values(result, dtype)
