"""Fused-elementwise execution.

A chain of elementwise steps is compiled into one CUDA kernel (generated as
source, compiled through a cached RawKernel) or interpreted in one NumPy
pass, both forward and backward."""

from __future__ import annotations

import hashlib
import importlib
from collections.abc import Sequence
from functools import lru_cache
from typing import Any, TYPE_CHECKING

from ..storage import CudaStorage, NumPyStorage, Storage
from ...strides import Strides
from .core import _errstate, _numpy, _view

if TYPE_CHECKING:
    from ...dtype import DataType
    from ...tensor import Tensor
    from ..types import FusedElementwiseStep

_FUSED_BINARY_OPERATIONS = frozenset({
    "add",
    "subtract",
    "multiply",
    "divide",
    "power",
})

_FUSED_UNARY_OPERATIONS = frozenset({
    "identity",
    "negate",
    "abs",
    "sqrt",
    "exp",
    "log",
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    "sinh",
    "cosh",
    "arcsinh",
    "arccosh",
    "arctanh",
    "sign",
    "relu",
    "sigmoid",
    "tanh",
    "softplus",
})

def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Return row-major element strides for a shape."""
    return Strides.contiguous(shape)

def _broadcast_offset_expression(
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
) -> str:
    """Return CUDA code mapping one flat output index to a broadcast input."""
    if len(input_shape) > len(output_shape):
        raise ValueError("Fused input rank exceeds the output rank")
    aligned = (1,) * (len(output_shape) - len(input_shape)) + input_shape
    input_strides = _contiguous_strides(aligned)
    output_strides = _contiguous_strides(output_shape)
    terms = []
    for input_size, output_size, input_stride, output_stride in zip(
        aligned,
        output_shape,
        input_strides,
        output_strides,
    ):
        if input_size not in {1, output_size}:
            raise ValueError("Fused input is not broadcast-compatible")
        if input_size == 1:
            continue
        terms.append(
            f"((index / {output_stride}ULL) % {output_size}ULL) "
            f"* {input_stride}ULL"
        )
    return " + ".join(terms) if terms else "0ULL"

def _fused_operand_expression(
    step: FusedElementwiseStep,
    current: str,
) -> str | None:
    """Return the scalar, current, or tensor operand expression for a step."""
    _, scalar, _, operand_index = step
    if scalar is not None:
        literal = format(float(scalar), ".17g")
        return literal if any(marker in literal for marker in ".eE") else f"{literal}.0"
    if operand_index == -1:
        return current
    if operand_index is not None:
        return f"((double)input_{operand_index}[offset_{operand_index}])"
    return None

def _fused_unary_expression(operation: str, value: str) -> str:
    """Return a range-stable CUDA expression for a total unary operation."""
    if operation == "identity":
        return value
    if operation == "negate":
        return f"-({value})"
    if operation == "abs":
        return f"fabs({value})"
    if operation == "sqrt":
        return f"sqrt({value})"
    if operation == "exp":
        return f"exp({value})"
    if operation == "log":
        return f"log({value})"
    if operation == "sin":
        return f"sin({value})"
    if operation == "cos":
        return f"cos({value})"
    if operation == "tan":
        return f"tan({value})"
    if operation == "arcsin":
        return f"asin({value})"
    if operation == "arccos":
        return f"acos({value})"
    if operation == "arctan":
        return f"atan({value})"
    if operation == "sinh":
        return f"sinh({value})"
    if operation == "cosh":
        return f"cosh({value})"
    if operation == "arcsinh":
        return f"asinh({value})"
    if operation == "arccosh":
        return f"acosh({value})"
    if operation == "arctanh":
        return f"atanh({value})"
    if operation == "sign":
        return (
            f"(isnan({value}) ? ({value}) : "
            f"(({value}) > 0.0 ? 1.0 : (({value}) < 0.0 ? -1.0 : 0.0)))"
        )
    if operation == "relu":
        return f"(isnan({value}) ? ({value}) : (({value}) > 0.0 ? ({value}) : 0.0))"
    if operation == "sigmoid":
        return (
            f"(({value}) >= 0.0 "
            f"? 1.0 / (1.0 + exp(-({value}))) "
            f": exp({value}) / (1.0 + exp({value})))"
        )
    if operation == "tanh":
        return f"tanh({value})"
    if operation == "softplus":
        return f"(log1p(exp(-fabs({value}))) + fmax(({value}), 0.0))"
    raise ValueError(f"Unsupported fused unary operation {operation!r}")

def _fused_step_expression(
    step: FusedElementwiseStep,
    current: str,
) -> tuple[str, str | None]:
    """Return a step expression and an optional denominator to validate."""
    operation, _, reverse, _ = step
    if operation in _FUSED_UNARY_OPERATIONS:
        return _fused_unary_expression(operation, current), None
    if operation not in _FUSED_BINARY_OPERATIONS:
        raise ValueError(f"Unsupported fused operation {operation!r}")
    operand = _fused_operand_expression(step, current)
    if operand is None:
        raise ValueError("A fused binary step requires an operand")
    left, right = (operand, current) if reverse else (current, operand)
    operators = {
        "add": "+",
        "subtract": "-",
        "multiply": "*",
        "divide": "/",
    }
    expression = (
        f"pow(({left}), ({right}))"
        if operation == "power"
        else f"({left}) {operators[operation]} ({right})"
    )
    return expression, right if operation == "divide" else None

def _fused_domain_checks(
    step: FusedElementwiseStep,
    value: str,
    result: str,
) -> tuple[tuple[str, int], ...]:
    """Return CUDA predicates that preserve public math-domain errors."""
    operation, _, reverse, _ = step
    operand = _fused_operand_expression(step, value)
    if operation == "sqrt":
        return ((f"({value}) < 0.0", 2),)
    if operation == "log":
        return ((f"({value}) <= 0.0", 3),)
    if operation in {"arcsin", "arccos"}:
        return ((f"({value}) < -1.0 || ({value}) > 1.0", 4),)
    if operation == "arccosh":
        return ((f"({value}) < 1.0", 5),)
    if operation == "arctanh":
        return ((
            f"!isnan({value}) && (({value}) <= -1.0 || ({value}) >= 1.0)",
            6,
        ),)
    if operation in {"sin", "cos", "tan"}:
        return ((f"isinf({value})", 7),)
    if operation == "power" and operand is not None:
        left, right = (operand, value) if reverse else (value, operand)
        return (
            (
                f"({left}) < 0.0 && trunc({right}) != ({right})",
                8,
            ),
            (f"({left}) == 0.0 && ({right}) < 0.0", 8),
            (
                f"isfinite({left}) && isfinite({right}) && isinf({result})",
                9,
            ),
        )
    return ()

def _raise_fused_kernel_error(code: int) -> None:
    """Raise the public exception represented by a fused-kernel error code."""
    if code == 1:
        raise ZeroDivisionError("Division by zero")
    messages = {
        2: "sqrt is only defined for non-negative values",
        3: "log is only defined for positive values",
        4: "inverse trigonometric function is only defined between -1 and 1",
        5: "arccosh is only defined for values greater than or equal to 1",
        6: "arctanh is only defined for values strictly between -1 and 1",
        7: "trigonometric functions are undefined for infinite values",
        8: "power is not defined for these real-valued inputs",
        10: "sqrt derivative is undefined at zero",
        11: "inverse trigonometric derivative is undefined at -1 and 1",
        12: "arccosh derivative is undefined at 1",
        13: "sign derivative is undefined at zero",
        14: "power derivative is undefined at a zero base",
    }
    if code == 9:
        raise OverflowError("power result is too large to represent")
    raise ValueError(messages.get(code, "invalid value in fused CUDA operation"))

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

@lru_cache(maxsize=128)
def _cuda_fused_elementwise_kernel(
    steps: tuple[FusedElementwiseStep, ...],
    dtype_name: str,
    input_shapes: tuple[tuple[int, ...], ...],
    output_shape: tuple[int, ...],
) -> tuple[Any, bool]:
    """Compile and cache one typed broadcast-aware forward kernel."""
    cupy = importlib.import_module("cupy")
    storage_type = "float" if dtype_name == "float32" else "double"
    body = ["const double value_0 = (double)input_0[offset_0];"]
    validate_division = False
    for index, step in enumerate(steps):
        expression, denominator = _fused_step_expression(
            step,
            f"value_{index}",
        )
        if denominator is not None:
            _, scalar, reverse, operand_index = step
            needs_check = scalar is None or reverse or operand_index is not None
            if needs_check:
                validate_division = True
                body.append(
                    f"if (({denominator}) == 0.0) {{ atomicExch(error, 1); }}"
                )
        body.extend(_fused_value_statements(
            f"value_{index + 1}",
            expression,
            dtype_name=dtype_name,
        ))
        checks = _fused_domain_checks(
            step,
            f"value_{index}",
            f"value_{index + 1}",
        )
        if checks:
            validate_division = True
            for condition, code in checks:
                body.append(
                    f"if ({condition}) {{ atomicExch(error, {code}); }}"
                )
        body.append(_fused_output_statement(
            index,
            f"value_{index + 1}",
            storage_type=storage_type,
        ))

    signature = repr((
        "forward",
        steps,
        dtype_name,
        input_shapes,
        output_shape,
    )).encode("utf-8")
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
    return cupy.RawKernel(source, name), validate_division

def _fused_derivative_expressions(
    step: FusedElementwiseStep,
    value: str,
    result: str,
) -> tuple[str, str | None]:
    """Return derivatives with respect to current and an external operand."""
    operation, scalar, reverse, operand_index = step
    operand = _fused_operand_expression(step, value)
    if operation == "identity":
        return "1.0", None
    if operation == "negate":
        return "-1.0", None
    if operation == "abs":
        return (
            f"(isnan({value}) ? ({value}) : "
            f"(({value}) > 0.0 ? 1.0 : (({value}) < 0.0 ? -1.0 : 0.0)))",
            None,
        )
    if operation == "sqrt":
        return f"(0.5 / ({result}))", None
    if operation == "exp":
        return f"exp({value})", None
    if operation == "log":
        return f"(1.0 / ({value}))", None
    if operation == "sin":
        return f"cos({value})", None
    if operation == "cos":
        return f"-sin({value})", None
    if operation == "tan":
        cosine = f"cos({value})"
        return f"(1.0 / (({cosine}) * ({cosine})))", None
    if operation == "arcsin":
        return f"(1.0 / sqrt(1.0 - ({value}) * ({value})))", None
    if operation == "arccos":
        return f"(-1.0 / sqrt(1.0 - ({value}) * ({value})))", None
    if operation == "arctan":
        reciprocal = f"(1.0 / fabs({value}))"
        return (
            f"(isinf({value}) ? 0.0 : (fabs({value}) <= 1.0 "
            f"? 1.0 / (1.0 + ({value}) * ({value})) "
            f": (({reciprocal}) * ({reciprocal})) / "
            f"(1.0 + ({reciprocal}) * ({reciprocal}))))",
            None,
        )
    if operation == "sinh":
        return f"cosh({value})", None
    if operation == "cosh":
        return f"sinh({value})", None
    if operation == "arcsinh":
        reciprocal = f"(1.0 / fabs({value}))"
        return (
            f"(isinf({value}) ? 0.0 : (fabs({value}) <= 1.0 "
            f"? 1.0 / sqrt(1.0 + ({value}) * ({value})) "
            f": ({reciprocal}) / sqrt(1.0 + ({reciprocal}) * ({reciprocal}))))",
            None,
        )
    if operation == "arccosh":
        return (
            f"(isinf({value}) ? 0.0 : "
            f"1.0 / (sqrt(({value}) - 1.0) * sqrt(({value}) + 1.0)))",
            None,
        )
    if operation == "arctanh":
        return f"(1.0 / (1.0 - ({value}) * ({value})))", None
    if operation == "sign":
        return f"(isnan({value}) ? ({value}) : 0.0)", None
    if operation == "relu":
        return (
            f"(isnan({value}) ? ({value}) : "
            f"(({value}) > 0.0 ? 1.0 : 0.0))",
            None,
        )
    if operation in {"sigmoid", "softplus"}:
        z = (
            f"(({value}) >= 0.0 ? exp(-({value})) : exp({value}))"
        )
        sigmoid_derivative = f"(({z}) / ((1.0 + ({z})) * (1.0 + ({z}))))"
        if operation == "sigmoid":
            return sigmoid_derivative, None
        sigmoid_value = _fused_unary_expression("sigmoid", value)
        return sigmoid_value, None
    if operation == "tanh":
        z = f"exp(-2.0 * fabs({value}))"
        return f"(4.0 * ({z}) / ((1.0 + ({z})) * (1.0 + ({z}))))", None

    if operation == "power" and operand is not None:
        if reverse:
            return f"(({result}) * log({operand}))", None
        if scalar == 0:
            return "0.0", None
        return f"(({operand}) * pow(({value}), ({operand}) - 1.0))", None

    if operation not in _FUSED_BINARY_OPERATIONS or operand is None:
        raise ValueError(f"Unsupported fused backward operation {operation!r}")
    if operand_index == -1:
        derivatives = {
            "add": "2.0",
            "subtract": "0.0",
            "multiply": f"(2.0 * ({value}))",
        }
        if operation == "divide":
            return "0.0", None
        derivative = derivatives.get(operation)
        if derivative is None:
            raise ValueError(f"Self-{operation} is not fused in backward")
        return derivative, None
    if operation == "add":
        return "1.0", None if scalar is not None else "1.0"
    if operation == "subtract":
        current = "-1.0" if reverse else "1.0"
        external = "1.0" if reverse else "-1.0"
        return current, None if scalar is not None else external
    if operation == "multiply":
        return operand, None if scalar is not None else value
    if operation == "divide":
        if reverse:
            current = f"(-({result}) / ({value}))"
            external = f"(1.0 / ({value}))"
        else:
            current = f"(1.0 / ({operand}))"
            external = f"(-({result}) / ({operand}))"
        return current, None if scalar is not None else external
    raise ValueError(f"Unsupported fused backward operation {operation!r}")

def _fused_backward_checks(
    step: FusedElementwiseStep,
    value: str,
) -> tuple[tuple[str, int], ...]:
    """Return derivative-domain checks for one fused operation."""
    operation, scalar, reverse, _ = step
    if operation == "sqrt":
        return ((f"({value}) == 0.0", 10),)
    if operation in {"arcsin", "arccos"}:
        return ((f"({value}) == -1.0 || ({value}) == 1.0", 11),)
    if operation == "arccosh":
        return ((f"({value}) == 1.0", 12),)
    if operation == "sign":
        return ((f"({value}) == 0.0", 13),)
    if operation == "power" and not reverse:
        if scalar is not None:
            if scalar != 0 and scalar < 1:
                return ((f"({value}) == 0.0", 14),)
            return ()
        operand = _fused_operand_expression(step, value)
        if operand is not None:
            # A fractional exponent below one has no derivative at a zero base.
            return ((
                f"({value}) == 0.0 && ({operand}) != 0.0 "
                f"&& ({operand}) < 1.0",
                14,
            ),)
    return ()

def _fused_external_gradient_available(step: FusedElementwiseStep) -> bool:
    """Whether a fused step can produce its external operand's gradient.

    A power step's compact form records no base or exponent VJP, so a chain
    whose caller wants that derivative must fall back to ordinary operation
    execution rather than read an unwritten row.
    """
    operation, scalar, _, operand_index = step
    if scalar is not None or operand_index is None or operand_index < 0:
        return False
    return operation != "power"

def _fused_vjp_expressions(
    step: FusedElementwiseStep,
    value: str,
    result: str,
    upstream: str,
) -> tuple[str, str | None]:
    """Return range-stable VJP contributions for one fused step."""
    operation, scalar, reverse, operand_index = step
    operand = _fused_operand_expression(step, value)
    if operation == "power" and operand is not None:
        if reverse:
            logarithm = f"log({operand})"
            contribution = (
                f"(({upstream}) == 0.0 || ({logarithm}) == 0.0 ? 0.0 : "
                f"copysign(exp(log(fabs({upstream})) + ({value}) * ({logarithm}) "
                f"+ log(fabs({logarithm}))), ({upstream}) * ({logarithm})))"
            )
            return contribution, None

        exponent = operand
        magnitude = (
            f"copysign(exp(log(fabs({upstream})) + log(fabs({exponent})) "
            f"+ (({exponent}) - 1.0) * log(fabs({value}))), {{sign}})"
        )
        if scalar is not None:
            # A literal exponent resolves its parity while building the kernel.
            sign = f"({upstream}) * ({exponent})"
            if float(scalar).is_integer() and (int(scalar) - 1) % 2:
                sign = f"({sign}) * (({value}) < 0.0 ? -1.0 : 1.0)"
            if scalar == 0:
                return "0.0", None
            if scalar == 1:
                return upstream, None
            contribution = (
                f"(({upstream}) == 0.0 || ({value}) == 0.0 ? 0.0 : "
                + magnitude.format(sign=sign)
                + ")"
            )
            return contribution, None

        # An operand exponent resolves the same parity rule at runtime. An
        # even integer exponent leaves an odd power in the derivative, so the
        # result follows the sign of the base.
        even_integer = (
            f"(({exponent}) == floor({exponent}) "
            f"&& fmod({exponent}, 2.0) == 0.0)"
        )
        sign = (
            f"(({upstream}) * ({exponent}) * ({even_integer} "
            f"? (({value}) < 0.0 ? -1.0 : 1.0) : 1.0))"
        )
        contribution = (
            f"(({exponent}) == 0.0 ? 0.0 "
            f": (({exponent}) == 1.0 ? ({upstream}) "
            f": ((({upstream}) == 0.0 || ({value}) == 0.0) ? 0.0 : "
            + magnitude.format(sign=sign)
            + ")))"
        )
        return contribution, None

    if operation == "divide" and operand is not None:
        if operand_index == -1:
            return "0.0", None
        if reverse:
            zero = f"(({upstream}) == 0.0 || ({operand}) == 0.0)"
            sign = (
                f"(signbit({upstream}) != signbit({operand}) ? 1.0 : -1.0)"
            )
            stable = (
                f"({zero} ? 0.0 : ({sign}) * exp(log(fabs({upstream})) "
                f"+ log(fabs({operand})) - 2.0 * log(fabs({value}))))"
            )
            direct = f"(-({upstream}) * ({operand}) / (({value}) * ({value})))"
            square = f"(({value}) * ({value}))"
            current = (
                f"(isfinite({square}) && ({square}) != 0.0 && "
                f"({zero} || (({direct}) != 0.0 && isfinite({direct}))) "
                f"? ({direct}) : ({stable}))"
            )
            external = f"(({upstream}) / ({value}))"
        else:
            current = f"(({upstream}) / ({operand}))"
            zero = f"(({upstream}) == 0.0 || ({value}) == 0.0)"
            sign = f"(signbit({upstream}) != signbit({value}) ? 1.0 : -1.0)"
            stable = (
                f"({zero} ? 0.0 : ({sign}) * exp(log(fabs({upstream})) "
                f"+ log(fabs({value})) - 2.0 * log(fabs({operand}))))"
            )
            direct = f"(-({upstream}) * ({value}) / (({operand}) * ({operand})))"
            square = f"(({operand}) * ({operand}))"
            external = (
                f"(isfinite({square}) && ({square}) != 0.0 && "
                f"({zero} || (({direct}) != 0.0 && isfinite({direct}))) "
                f"? ({direct}) : ({stable}))"
            )
        return current, None if scalar is not None else external

    current_derivative, operand_derivative = _fused_derivative_expressions(
        step,
        value,
        result,
    )
    current = f"({upstream}) * ({current_derivative})"
    external = (
        None
        if operand_derivative is None
        else f"({upstream}) * ({operand_derivative})"
    )
    return current, external

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

def _fused_arrays(
    values: Sequence[Tensor],
    dtype: DataType,
    cupy: Any,
) -> tuple[Any, ...]:
    """Return contiguous device arrays cast to the fused storage dtype."""
    provider_dtype = cupy.dtype(dtype.name)
    return tuple(
        _view(value, cupy).astype(provider_dtype, copy=False).reshape(-1)
        for value in values
    )

def fused_elementwise(
    values: tuple[Tensor, ...],
    steps: tuple[FusedElementwiseStep, ...],
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> tuple[Storage, ...] | None:
    """Evaluate a typed expression chain without public Tensor dispatch."""
    from ..config import get_backend

    backend = get_backend()
    if (
        dtype.kind != "floating"
        or not values
        or len(steps) < 2
    ):
        return None
    if backend == "numpy":
        return _numpy_fused_elementwise(
            values,
            steps,
            dtype=dtype,
            output_shape=output_shape,
        )
    if backend != "cuda":
        return None
    cupy = _numpy()
    size = 1
    for dimension in output_shape:
        size *= dimension
    result = cupy.empty((len(steps), size), dtype=cupy.dtype(dtype.name))
    if not size:
        return tuple(CudaStorage(result[index], dtype) for index in range(len(steps)))
    try:
        arrays = _fused_arrays(values, dtype, cupy)
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
    return tuple(
        CudaStorage(result[index], dtype)
        for index in range(len(steps))
    )

def _numpy_fused_elementwise(
    values: tuple[Tensor, ...],
    steps: tuple[FusedElementwiseStep, ...],
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> tuple[Storage, ...] | None:
    """Evaluate a simple same-shape chain directly on NumPy arrays."""
    supported = {"add", "subtract", "multiply", "negate"}
    if (
        any(step[0] not in supported for step in steps)
        or any(value.shape != output_shape for value in values)
    ):
        return None
    numpy = _numpy()
    provider_dtype = numpy.dtype(dtype.name)
    sources = tuple(
        _view(value, numpy).astype(provider_dtype, copy=False)
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
    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        for operation, scalar, reverse, operand_index in steps:
            if operation == "negate":
                result = functions[operation](current)
            else:
                operand = (
                    scalar
                    if scalar is not None
                    else current
                    if operand_index == -1
                    else sources[operand_index]
                )
                left, right = (operand, current) if reverse else (current, operand)
                result = functions[operation](left, right)
            current = numpy.asarray(result, dtype=provider_dtype)
            storage = NumPyStorage(current.reshape(-1), dtype)
            storages.append(storage)
            current = storage.buffer.reshape(output_shape)
    return tuple(storages)

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
    from ..config import get_backend

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
