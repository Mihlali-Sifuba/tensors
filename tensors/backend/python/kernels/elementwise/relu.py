"""Reference the rectified linear unit for the Python backend."""

from __future__ import annotations
from collections.abc import Iterable
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import DataType
import math


def _relu(value):
    if isinstance(value, float) and math.isnan(value):
        return math.nan
    return value if value > 0 else 0


def relu(
    values: Iterable[int | float],
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Rectify each prepared value, leaving NaN in place.

    Everything that is not strictly positive becomes the integer ``0``, which
    the declared dtype renders as canonical positive zero for a floating
    format. That covers ``-0.0`` and ``-inf`` as well as ordinary negatives,
    and it is why ReLU canonicalises both zeros where ``sqrt`` does not.
    """
    result = [_relu(item) for item in values]
    if len(result) != math.prod(output_shape):
        raise RuntimeError("ReLU kernel returned an unexpected result size")
    return PythonStorage.from_values(result, dtype)
