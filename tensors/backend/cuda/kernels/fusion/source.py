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


#: Conversions between binary32 and binary64, written in PTX.
#:
#: The compiler flushes a subnormal in ``cvt.f64.f32`` and ``cvt.rn.f32.f64``
#: just as it does in an arithmetic instruction, and ``--ftz=false`` does not
#: reach it. A fused binary32 expression therefore lost every subnormal at its
#: first conversion, disagreeing with the same expression evaluated eagerly,
#: which section 8.5 forbids and section 5.4 requires gradual underflow for.
#: Naming the conversions in PTX keeps them, exactly as the eager kernels do.
CONVERSIONS = """
__device__ __forceinline__ double _tensors_widen(float value) {
    double widened;
    asm("cvt.f64.f32 %0, %1;" : "=d"(widened) : "f"(value));
    return widened;
}
__device__ __forceinline__ float _tensors_narrow(double value) {
    float narrowed;
    asm("cvt.rn.f32.f64 %0, %1;" : "=f"(narrowed) : "d"(value));
    return narrowed;
}
"""


def _widen(expression: str, *, storage_type: str) -> str:
    """Read a stored value into the binary64 working precision."""
    if storage_type == "float":
        return f"_tensors_widen({expression})"
    return f"(double)({expression})"


def _narrow(expression: str, *, storage_type: str) -> str:
    """Round a working value back to the storage format."""
    if storage_type == "float":
        return f"_tensors_narrow({expression})"
    return f"(double)({expression})"


def _fused_value_statements(
    name: str, expression: str, *, dtype_name: str
) -> list[str]:
    """Assign a working value with the same rounding as a graph boundary."""
    if dtype_name == "float32":
        narrowed = _narrow(expression, storage_type="float")
        return [
            f"const float {name}_stored = {narrowed};",
            f"const double {name} = _tensors_widen({name}_stored);",
        ]
    return [f"const double {name} = (double)({expression});"]


def _fused_output_statement(row: int, expression: str, *, storage_type: str) -> str:
    stored = (
        f"_tensors_narrow({expression})"
        if storage_type == "float"
        else f"({storage_type})({expression})"
    )
    return f"output[index + {row}ULL * size] = {stored};"


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
    preamble = CONVERSIONS if storage_type == "float" else ""
    return f'{preamble}\nextern "C" __global__\nvoid {name}(\n    {joined_parameters}\n) {{\n    const unsigned long long index =\n        (unsigned long long)blockDim.x * blockIdx.x + threadIdx.x;\n    if (index >= size) {{\n        return;\n    }}\n    {joined_body}\n}}\n'
