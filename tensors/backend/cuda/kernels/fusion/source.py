"""CUDA source assembly shared by the forward and backward kernels."""

from __future__ import annotations
from tensors.backend.cuda.kernels.fusion.expressions import _broadcast_offset_expression

#: NVRTC contracts a multiply followed by an add into a fused multiply-add by
#: default, which rounds once where the unfused sequence rounds twice.
#: ``docs/autodiff.md`` requires an optimisation to preserve the result of the
#: sequence it replaces, so the contraction is disabled rather than relied upon
#: not to happen. Storing every step to memory currently blocks it as a side
#: effect; that is a property of the present code generation, not a guarantee.
_FUSION_OPTIONS = ("--fmad=false",)


def _fused_value_statements(
    name: str, expression: str, *, dtype_name: str
) -> list[str]:
    """Assign a working value with the same rounding as a graph boundary."""
    if dtype_name == "float32":
        return [
            f"const float {name}_stored = (float)({expression});",
            f"const double {name} = (double){name}_stored;",
        ]
    return [f"const double {name} = (double)({expression});"]


def _fused_output_statement(row: int, expression: str, *, storage_type: str) -> str:
    return f"output[index + {row}ULL * size] = ({storage_type})({expression});"


def _fused_kernel_source(
    *,
    name: str,
    input_shapes: tuple[tuple[int, ...], ...],
    output_shape: tuple[int, ...],
    storage_type: str,
    body: list[str],
    validate_errors: bool,
    include_gradient: bool,
) -> str:
    """Build one broadcast-aware CUDA kernel source string."""
    parameters = [
        f"const {storage_type}* input_{index}" for index in range(len(input_shapes))
    ]
    if include_gradient:
        parameters.append(f"const {storage_type}* gradient")
    parameters.append(f"{storage_type}* output")
    if validate_errors:
        parameters.append("int* error")
    parameters.append("const unsigned long long size")
    offsets = [
        f"const unsigned long long offset_{index} = {_broadcast_offset_expression(shape, output_shape)};"
        for index, shape in enumerate(input_shapes)
    ]
    joined_parameters = ",\n    ".join(parameters)
    joined_body = "\n    ".join(offsets + body)
    return f'\nextern "C" __global__\nvoid {name}(\n    {joined_parameters}\n) {{\n    const unsigned long long index =\n        (unsigned long long)blockDim.x * blockIdx.x + threadIdx.x;\n    if (index >= size) {{\n        return;\n    }}\n    {joined_body}\n}}\n'
