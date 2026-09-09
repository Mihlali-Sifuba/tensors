"""Dispatch for fused elementwise chains.

The Python interpretation of a chain lives here with the dispatch entry point
that selects it, unchanged: it is the fallback the Python backend uses instead
of a compiled kernel.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from ..config import get_backend
from ..loading import _backend_kernel
from ..policy import _CUDA_FUSION_MIN_WORK, _shape_size
from ..storage import Storage
from ..types import FusedElementwiseStep

if TYPE_CHECKING:
    from ...dtype import DataType
    from ...tensor import Tensor

def execute_fused_elementwise(
    values: Sequence[Tensor],
    steps: Sequence[FusedElementwiseStep],
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> tuple[Storage, ...] | None:
    """Run a compatible floating-point expression chain as one backend plan."""
    work = _shape_size(output_shape)
    backend = get_backend()
    if (
        not values
        or len(steps) < 2
        or dtype.kind != "floating"
    ):
        return None
    if backend == "python":
        return _execute_python_fused_elementwise(
            values,
            steps,
            dtype=dtype,
            output_shape=output_shape,
        )
    if (
        backend == "cuda"
        and (
            work * len(steps) < _CUDA_FUSION_MIN_WORK
            and len(steps) < 64
        )
    ):
        return None
    fused_elementwise = _backend_kernel("fused_elementwise")
    return fused_elementwise(
        tuple(values),
        tuple(steps),
        dtype=dtype,
        output_shape=output_shape,
    )

def _execute_python_fused_elementwise(
    values: Sequence[Tensor],
    steps: Sequence[FusedElementwiseStep],
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> tuple[Storage, ...] | None:
    """Interpret a simple same-shape chain without intermediate Tensors."""
    from array import array
    import operator

    from ..storage import PythonStorage

    supported = {"add", "subtract", "multiply", "negate"}
    if (
        any(step[0] not in supported for step in steps)
        or any(value.shape != output_shape for value in values)
    ):
        return None
    sources = tuple(value._data for value in values)
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
                else current
                if operand_index == -1
                else sources[operand_index]
            )
            if isinstance(operand, (int, float)):
                pairs = ((item, operand) for item in current)
            else:
                pairs = zip(current, operand)
            if reverse:
                pairs = ((right, left) for left, right in pairs)
            function = functions[operation]
            converted = array(
                dtype.typecode,
                (function(left, right) for left, right in pairs),
            )
        storage = PythonStorage(converted, dtype)
        storages.append(storage)
        current = storage.buffer
    return tuple(storages)

def execute_fused_elementwise_backward(
    values: Sequence[Tensor],
    grad: Tensor,
    steps: Sequence[FusedElementwiseStep],
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
    requested_external: tuple[int, ...] = (),
) -> tuple[Storage, ...] | None:
    """Run the requested part of a chain VJP in one CUDA kernel.

    ``requested_external`` names the fused steps whose external operand
    gradient the current reverse pass wants. The backend refuses the request
    when its compact step form carries no such derivative, so the caller can
    fall back to ordinary operation execution.
    """
    work = _shape_size(output_shape)
    if (
        get_backend() != "cuda"
        or not values
        or len(steps) < 2
        or dtype.kind != "floating"
        or (
            work * len(steps) < _CUDA_FUSION_MIN_WORK
            and len(steps) < 64
        )
    ):
        return None
    fused_backward = _backend_kernel("fused_elementwise_backward")
    return fused_backward(
        tuple(values),
        grad,
        tuple(steps),
        dtype=dtype,
        output_shape=output_shape,
        requested_external=requested_external,
    )
