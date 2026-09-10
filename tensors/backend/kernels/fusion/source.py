"""CUDA source assembly shared by the forward and backward kernels."""

from __future__ import annotations


from .expressions import _broadcast_offset_expression

def _fused_value_statements(
    name: str,
    expression: str,
    *,
    dtype_name: str,
) -> list[str]:
    """Assign a working value with the same rounding as a graph boundary."""
    if dtype_name == "float32":
        return [
            f"const float {name}_stored = (float)({expression});",
            f"const double {name} = (double){name}_stored;",
        ]
    return [f"const double {name} = (double)({expression});"]

def _fused_output_statement(
    row: int,
    expression: str,
    *,
    storage_type: str,
) -> str:
    return (
        f"output[index + {row}ULL * size] = "
        f"({storage_type})({expression});"
    )

def _fused_kernel_source(
    *,
    name: str,
    input_shapes: tuple[tuple[int, ...], ...],
    output_shape: tuple[int, ...],
    storage_type: str,
    body: list[str],
    validate_division: bool,
    include_gradient: bool,
) -> str:
    """Build one broadcast-aware CUDA kernel source string."""
    parameters = [
        f"const {storage_type}* input_{index}"
        for index in range(len(input_shapes))
    ]
    if include_gradient:
        parameters.append(f"const {storage_type}* gradient")
    parameters.append(f"{storage_type}* output")
    if validate_division:
        parameters.append("int* error")
    parameters.append("const unsigned long long size")
    offsets = [
        f"const unsigned long long offset_{index} = "
        f"{_broadcast_offset_expression(shape, output_shape)};"
        for index, shape in enumerate(input_shapes)
    ]
    joined_parameters = ",\n    ".join(parameters)
    joined_body = "\n    ".join(offsets + body)
    return f"""
extern "C" __global__
void {name}(
    {joined_parameters}
) {{
    const unsigned long long index =
        (unsigned long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= size) {{
        return;
    }}
    {joined_body}
}}
"""
