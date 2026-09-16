"""Run a fused elementwise chain as one generated CUDA kernel."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import cupy

from tensors.backend.cuda.kernels.fusion.common import _fused_arrays
from tensors.backend.cuda.kernels.fusion.errors import (
    _fused_domain_checks,
    _raise_fused_kernel_error,
)
from tensors.backend.cuda.kernels.fusion.expressions import _fused_step_expression
from tensors.backend.cuda.kernels.fusion.source import (
    _fused_kernel_source,
    _fused_output_statement,
    _fused_value_statements,
)
from tensors.backend.cuda.storage import CudaStorage
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.backend.types import FusedElementwiseStep
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


@lru_cache(maxsize=128)
def _cuda_fused_elementwise_kernel(
    steps: tuple[FusedElementwiseStep, ...],
    dtype_name: str,
    input_shapes: tuple[tuple[int, ...], ...],
    output_shape: tuple[int, ...],
) -> tuple[Any, bool]:
    """Compile and cache one typed broadcast-aware forward kernel."""
    storage_type = "float" if dtype_name == "float32" else "double"
    body = ["const double value_0 = (double)input_0[offset_0];"]
    validate_division = False
    for index, step in enumerate(steps):
        expression, denominator = _fused_step_expression(step, f"value_{index}")
        if denominator is not None:
            _, scalar, reverse, operand_index = step
            needs_check = scalar is None or reverse or operand_index is not None
            if needs_check:
                validate_division = True
                body.append(f"if (({denominator}) == 0.0) {{ atomicExch(error, 1); }}")
        body.extend(
            _fused_value_statements(
                f"value_{index + 1}",
                expression,
                dtype_name=dtype_name,
            )
        )
        checks = _fused_domain_checks(step, f"value_{index}", f"value_{index + 1}")
        if checks:
            validate_division = True
            for condition, code in checks:
                body.append(f"if ({condition}) {{ atomicExch(error, {code}); }}")
        body.append(
            _fused_output_statement(
                index,
                f"value_{index + 1}",
                storage_type=storage_type,
            )
        )
    signature = repr(("forward", steps, dtype_name, input_shapes, output_shape)).encode(
        "utf-8"
    )
    digest = hashlib.sha1(signature).hexdigest()[:16]
    name = f"tensors_fused_forward_{digest}"
    source = _fused_kernel_source(
        name=name,
        input_shapes=input_shapes,
        output_shape=output_shape,
        storage_type=storage_type,
        body=body,
        validate_division=validate_division,
        include_gradient=False,
    )
    return (cupy.RawKernel(source, name), validate_division)


def fused_elementwise(
    values: tuple[Tensor, ...],
    steps: tuple[FusedElementwiseStep, ...],
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> tuple[Storage, ...] | None:
    """Evaluate a typed expression chain without public Tensor dispatch.

    Returns one storage per step, or ``None`` when the chain cannot be
    compiled, so the caller runs the steps separately.
    """
    if dtype.kind != "floating" or not values or len(steps) < 2:
        return None
    size = 1
    for dimension in output_shape:
        size *= dimension
    result = cupy.empty((len(steps), size), dtype=cupy.dtype(dtype.name))
    if not size:
        return tuple(CudaStorage(result[index], dtype) for index in range(len(steps)))
    try:
        arrays = _fused_arrays(values, dtype)
        kernel, validate_division = _cuda_fused_elementwise_kernel(
            steps,
            dtype.name,
            tuple(value.shape for value in values),
            output_shape,
        )
        error = cupy.zeros((1,), dtype=cupy.int32) if validate_division else None
        threads = 256
        blocks = (size + threads - 1) // threads
        arguments = list(arrays)
        arguments.append(result)
        if error is not None:
            arguments.append(error)
        arguments.append(cupy.uint64(size))
        kernel((blocks,), (threads,), tuple(arguments))
        if error is not None:
            error_code = int(error.item())
            if error_code:
                _raise_fused_kernel_error(error_code)
    except (TypeError, ValueError):
        return None
    return tuple(CudaStorage(result[index], dtype) for index in range(len(steps)))
