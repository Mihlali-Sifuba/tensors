"""Evaluate a fused elementwise chain in one NumPy pass."""

from __future__ import annotations

import numpy
from typing import TYPE_CHECKING

from tensors.backend.numpy.conversion import _errstate, tensor_to_logical_array
from tensors.backend.numpy.storage import NumPyStorage
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.backend.types import FusedElementwiseStep
    from tensors.dtype import DataType
    from tensors.tensor import Tensor

_SUPPORTED = {"add", "subtract", "multiply", "negate"}


def fused_elementwise(
    values: tuple[Tensor, ...],
    steps: tuple[FusedElementwiseStep, ...],
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> tuple[Storage, ...] | None:
    """Evaluate a typed expression chain without public Tensor dispatch.

    Returns one storage per step, or ``None`` when the chain uses a step or a
    shape this kernel does not fuse, so the caller runs the steps separately.
    """
    if dtype.kind != "floating" or not values or len(steps) < 2:
        return None
    if any(step[0] not in _SUPPORTED for step in steps) or any(
        value.shape != output_shape for value in values
    ):
        return None
    provider_dtype = numpy.dtype(dtype.name)
    sources = tuple(
        tensor_to_logical_array(value).astype(provider_dtype, copy=False)
        for value in values
    )
    current = sources[0]
    storages: list[Storage] = []
    functions = {
        "add": numpy.add,
        "subtract": numpy.subtract,
        "multiply": numpy.multiply,
        "negate": numpy.negative,
    }
    with _errstate(over="ignore", under="ignore", invalid="ignore"):
        for operation, scalar, reverse, operand_index in steps:
            if operation == "negate":
                result = functions[operation](current)
            else:
                if scalar is not None:
                    operand = scalar
                elif operand_index == -1:
                    operand = current
                else:
                    operand = sources[operand_index]
                left, right = (operand, current) if reverse else (current, operand)
                result = functions[operation](left, right)
            storage = NumPyStorage(
                numpy.asarray(result, dtype=provider_dtype).reshape(-1),
                dtype,
            )
            storages.append(storage)
            current = storage.buffer.reshape(output_shape)
    return tuple(storages)
