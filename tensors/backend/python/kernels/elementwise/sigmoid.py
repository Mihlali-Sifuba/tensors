"""Reference the logistic function for the Python backend."""

from __future__ import annotations
import math
from collections.abc import Iterable
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import DataType


def _sigmoid(value: float) -> float:
    """Return the logistic function of one value.

    ``1 / (1 + exp(-x))`` overflows ``exp`` for a large negative ``x``, and
    ``exp(x) / (1 + exp(x))`` overflows for a large positive one, so each is
    used only where its exponent is negative. The two are the same function;
    the branch exists so the intermediate stays in range, not to approximate.

    This is a per-element rule with a second caller outside this module: the
    binary cross-entropy VJP evaluates ``sigmoid(x) - t`` for a logit input
    and needs the same branch, to the same digits, or the two would disagree
    about a saturated sigmoid.
    """
    if value >= 0:
        magnitude = math.exp(-value)
        return 1.0 / (1.0 + magnitude)
    magnitude = math.exp(value)
    return magnitude / (1.0 + magnitude)


def sigmoid(
    values: Iterable[int | float],
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Apply the logistic function to each prepared value.

    Every input has a result. ``exp`` underflows to zero rather than raising
    for a large magnitude, so ``sigmoid(-800)`` is ``0.0`` and
    ``sigmoid(800)`` is ``1.0``; both infinities reach those same values, and
    NaN fails the comparison and propagates through the second branch.
    """
    result = [_sigmoid(item) for item in values]
    if len(result) != math.prod(output_shape):
        raise RuntimeError("Sigmoid kernel returned an unexpected result size")
    return PythonStorage.from_values(result, dtype)
