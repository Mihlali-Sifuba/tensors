"""Reference softplus for the Python backend."""

from __future__ import annotations
import math
from collections.abc import Iterable
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import DataType


def softplus(
    values: Iterable[int | float],
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Apply softplus through ``log1p`` so a large input does not overflow.

    ``log(1 + exp(x))`` overflows ``exp`` well before the result leaves the
    range: at ``x = 800`` the answer is ``800`` and ``exp(800)`` is not
    representable. Writing it as ``log1p(exp(-|x|)) + max(x, 0)`` keeps the
    exponent negative, so the term stays within range for every input, and
    ``log1p`` keeps the precision an ordinary ``log(1 + z)`` loses for small
    ``z`` — which is the whole of the result for a large negative ``x``.

    Every input has a result: ``+inf`` gives ``+inf``, ``-inf`` gives ``0.0``,
    and NaN propagates through both terms.
    """
    result = []
    for item in values:
        item = float(item)
        result.append(math.log1p(math.exp(-abs(item))) + max(item, 0.0))
    if len(result) != math.prod(output_shape):
        raise RuntimeError("Softplus kernel returned an unexpected result size")
    return PythonStorage.from_values(result, dtype)
