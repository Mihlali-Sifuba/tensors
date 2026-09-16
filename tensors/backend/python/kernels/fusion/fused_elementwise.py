"""Dispatch for fused elementwise chains.

The Python interpretation of a chain lives here with the dispatch entry point
that selects it, unchanged: it is the fallback the Python backend uses instead
of a compiled kernel.
"""

from __future__ import annotations
from collections.abc import Sequence
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.types import FusedElementwiseStep

if TYPE_CHECKING:
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


def fused_elementwise(
    values: Sequence[Tensor],
    steps: Sequence[FusedElementwiseStep],
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> tuple[Storage, ...] | None:
    """Interpret a simple same-shape chain without intermediate Tensors."""
    from array import array
    import operator
    from tensors.backend.python.storage import PythonStorage

    supported = {"add", "subtract", "multiply", "negate"}
    if any((step[0] not in supported for step in steps)) or any(
        (value.shape != output_shape for value in values)
    ):
        return None
    sources = tuple((value._data for value in values))
    current = sources[0]
    storages: list[Storage] = []
    functions = {
        "add": operator.add,
        "subtract": operator.sub,
        "multiply": operator.mul,
    }
    for operation, scalar, reverse, operand_index in steps:
        if operation == "negate":
            converted = array(dtype.typecode, (-item for item in current))
        else:
            operand = (
                scalar
                if scalar is not None
                else current if operand_index == -1 else sources[operand_index]
            )
            if isinstance(operand, (int, float)):
                pairs = ((item, operand) for item in current)
            else:
                pairs = zip(current, operand)
            if reverse:
                pairs = ((right, left) for left, right in pairs)
            function = functions[operation]
            converted = array(
                dtype.typecode, (function(left, right) for left, right in pairs)
            )
        storage = PythonStorage(converted, dtype)
        storages.append(storage)
        current = storage.buffer
    return tuple(storages)
