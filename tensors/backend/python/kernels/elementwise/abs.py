"""Reference absolute value for the Python backend."""

from __future__ import annotations
from collections.abc import Iterable
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import DataType
import builtins
import math


def abs(
    values: Iterable[int | float],
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Return the exact magnitude of every prepared value.

    Every value reaching here is representable: `execute_abs` has already
    refused a signed integer dtype's least value, so the magnitude never
    exceeds what the declared dtype holds and the typed buffer below is not
    where that rule is discovered. Both signed zeros give ``0``, which the
    declared dtype renders as canonical positive zero.
    """
    result = [builtins.abs(item) for item in values]
    if len(result) != math.prod(output_shape):
        raise RuntimeError("Abs kernel returned an unexpected result size")
    return PythonStorage.from_values(result, dtype)
