"""Fused backward (VJP) compilation and execution."""

from __future__ import annotations

import hashlib
import importlib
from functools import lru_cache
from typing import Any, TYPE_CHECKING

from ...storage import CudaStorage, Storage
from ..core import _numpy, _view
from .common import _fused_arrays
from .errors import _fused_backward_checks, _raise_fused_kernel_error
from .expressions import (
    _fused_external_gradient_available,
    _fused_step_expression,
    _fused_vjp_expressions,
)
from .source import (
    _fused_kernel_source,
    _fused_output_statement,
    _fused_value_statements,
)

if TYPE_CHECKING:
    from ....dtype import DataType
    from ....tensor import Tensor
    from ...types import FusedElementwiseStep

@lru_cache(maxsize=128)
def _cuda_fused_elementwise_backward_kernel(
    steps: tuple[FusedElementwiseStep, ...],
    dtype_name: str,
    input_shapes: tuple[tuple[int, ...], ...],
    output_shape: tuple[int, ...],
    requested_external: tuple[int, ...],
) -> tuple[Any, bool]:
    """Compile and cache one typed VJP kernel for a fused chain."""
    cupy = importlib.import_module("cupy")
    storage_type = "float" if dtype_name == "float32" else "double"
    body = ["const double value_0 = (double)input_0[offset_0];"]
    for index, step in enumerate(steps):
        expression, _ = _fused_step_expression(step, f"value_{index}")
        body.extend(_fused_value_statements(
            f"value_{index + 1}",
            expression,
            dtype_name=dtype_name,
        ))

    # The generated layout depends on which external gradients were asked
    # for, so ``requested_external`` participates in the kernel cache key.
    external_rows = {}
    next_row = len(steps) + 1
    for index in requested_external:
        external_rows[index] = next_row
        next_row += 1

    body.append(
        f"const double upstream_{len(steps)} = (double)gradient[index];"
    )
    validate_errors = False
    for index in range(len(steps) - 1, -1, -1):
        upstream = f"upstream_{index + 1}"
        body.append(_fused_output_statement(
            index,
            upstream,
            storage_type=storage_type,
        ))
        current_contribution, operand_contribution = (
            _fused_vjp_expressions(
                steps[index],
                f"value_{index}",
                f"value_{index + 1}",
                upstream,
            )
        )
        checks = _fused_backward_checks(steps[index], f"value_{index}")
        if checks:
            validate_errors = True
            for condition, code in checks:
                body.append(
                    f"if ({condition}) {{ atomicExch(error, {code}); }}"
                )
        if operand_contribution is not None and index in external_rows:
            # Only a requested external operand gradient gets an output row.
            body.append(_fused_output_statement(
                external_rows[index],
                operand_contribution,
                storage_type=storage_type,
            ))
        body.extend(_fused_value_statements(
            f"upstream_{index}",
            current_contribution,
            dtype_name=dtype_name,
        ))
    body.append(_fused_output_statement(
        len(steps),
        "upstream_0",
        storage_type=storage_type,
    ))

    signature = repr((
        "backward",
        steps,
        dtype_name,
        input_shapes,
        output_shape,
    )).encode("utf-8")
    digest = hashlib.sha1(signature).hexdigest()[:16]
    name = f"tensors_fused_backward_{digest}"
    source = _fused_kernel_source(
        name=name,
        input_shapes=input_shapes,
        output_shape=output_shape,
        storage_type=storage_type,
        body=body,
        validate_division=validate_errors,
        include_gradient=True,
    )
    return cupy.RawKernel(source, name), validate_errors

def fused_elementwise_backward(
    values: tuple[Tensor, ...],
    grad: Tensor,
    steps: tuple[FusedElementwiseStep, ...],
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
    requested_external: tuple[int, ...] = (),
) -> tuple[Storage, ...] | None:
    """Evaluate the requested same-shape VJPs for a fused expression chain.

    ``requested_external`` names the step indices whose external operand
    gradient the caller wants. Rows are produced only for those, and the
    request is refused when a step's compact form carries no such derivative,
    so the caller can fall back to ordinary operation execution.
    """
    from ...config import get_backend

    if (
        get_backend() != "cuda"
        or dtype.kind != "floating"
        or not values
        or len(steps) < 2
        or grad.shape != output_shape
    ):
        return None
    if any(
        not _fused_external_gradient_available(steps[index])
        for index in requested_external
    ):
        return None
    cupy = _numpy()
    size = grad.size
    row_count = len(steps) + 1 + len(requested_external)
    result = cupy.empty((row_count, size), dtype=cupy.dtype(dtype.name))
    if not size:
        return tuple(CudaStorage(result[index], dtype) for index in range(row_count))
    try:
        arrays = _fused_arrays(values, dtype, cupy)
        gradient = _view(grad, cupy).astype(
            cupy.dtype(dtype.name),
            copy=False,
        ).reshape(-1)
        kernel, validate_errors = _cuda_fused_elementwise_backward_kernel(
            steps,
            dtype.name,
            tuple(value.shape for value in values),
            output_shape,
            requested_external,
        )
        error = cupy.zeros((1,), dtype=cupy.int32) if validate_errors else None
        threads = 256
        blocks = (size + threads - 1) // threads
        arguments = [*arrays, gradient, result]
        if error is not None:
            arguments.append(error)
        arguments.append(cupy.uint64(size))
        kernel(
            (blocks,),
            (threads,),
            tuple(arguments),
        )
        if error is not None:
            error_code = int(error.item())
            if error_code:
                _raise_fused_kernel_error(error_code)
    except (TypeError, ValueError):
        return None
    return tuple(
        CudaStorage(result[index], dtype)
        for index in range(row_count)
    )
