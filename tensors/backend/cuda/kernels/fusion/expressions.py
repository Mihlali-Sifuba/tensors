"""Fused step, operand, and derivative expressions.

The lowest fusion layer: it turns compact step tuples into the C expression
text that the generated kernels evaluate, forward and reverse.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from tensors.strides import Strides

if TYPE_CHECKING:
    from tensors.backend.types import FusedElementwiseStep
_FUSED_BINARY_OPERATIONS = frozenset({"add", "subtract", "multiply", "divide", "power"})
_FUSED_UNARY_OPERATIONS = frozenset(
    {
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
    }
)


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Return row-major element strides for a shape."""
    return Strides.contiguous(shape)


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


def _broadcast_offset_expression(
    input_shape: tuple[int, ...], output_shape: tuple[int, ...]
) -> str:
    """Return CUDA code mapping one flat output index to a broadcast input."""
    if len(input_shape) > len(output_shape):
        raise ValueError("Fused input rank exceeds the output rank")
    aligned = (1,) * (len(output_shape) - len(input_shape)) + input_shape
    input_strides = _contiguous_strides(aligned)
    output_strides = _contiguous_strides(output_shape)
    terms = []
    for input_size, output_size, input_stride, output_stride in zip(
        aligned, output_shape, input_strides, output_strides
    ):
        if input_size not in {1, output_size}:
            raise ValueError("Fused input is not broadcast-compatible")
        if input_size == 1:
            continue
        terms.append(
            f"((index / {output_stride}ULL) % {output_size}ULL) * {input_stride}ULL"
        )
    return " + ".join(terms) if terms else "0ULL"


def _fused_operand_expression(
    step: FusedElementwiseStep, current: str, *, storage_type: str
) -> str | None:
    """Return the scalar, current, or tensor operand expression for a step.

    An external tensor operand is read from storage here, so it crosses into
    the binary64 working precision exactly as the initial input does and takes
    the same conversion. A plain ``(double)`` cast flushes a binary32
    subnormal, which would make a fused ``0.0 ** smallest_subnormal`` evaluate
    ``0.0 ** 0.0`` and return one where the specified result is ``+0.0``.
    """
    _, scalar, _, operand_index = step
    if scalar is not None:
        literal = format(float(scalar), ".17g")
        return (
            literal if any((marker in literal for marker in ".eE")) else f"{literal}.0"
        )
    if operand_index == -1:
        return current
    if operand_index is not None:
        return _widen(
            f"input_{operand_index}[offset_{operand_index}]",
            storage_type=storage_type,
        )
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
        return f"(isnan({value}) ? ({value}) : (({value}) > 0.0 ? 1.0 : (({value}) < 0.0 ? -1.0 : 0.0)))"
    if operation == "relu":
        return f"(isnan({value}) ? ({value}) : (({value}) > 0.0 ? ({value}) : 0.0))"
    if operation == "sigmoid":
        return f"(({value}) >= 0.0 ? 1.0 / (1.0 + exp(-({value}))) : exp({value}) / (1.0 + exp({value})))"
    if operation == "tanh":
        return f"tanh({value})"
    if operation == "softplus":
        return f"(log1p(exp(-fabs({value}))) + fmax(({value}), 0.0))"
    raise ValueError(f"Unsupported fused unary operation {operation!r}")


def _fused_step_expression(
    step: FusedElementwiseStep, current: str, *, storage_type: str
) -> tuple[str, str | None]:
    """Return a step expression and an optional denominator to validate."""
    operation, _, reverse, _ = step
    if operation in _FUSED_UNARY_OPERATIONS:
        return (_fused_unary_expression(operation, current), None)
    if operation not in _FUSED_BINARY_OPERATIONS:
        raise ValueError(f"Unsupported fused operation {operation!r}")
    operand = _fused_operand_expression(step, current, storage_type=storage_type)
    if operand is None:
        raise ValueError("A fused binary step requires an operand")
    left, right = (operand, current) if reverse else (current, operand)
    operators = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/"}
    expression = (
        f"pow(({left}), ({right}))"
        if operation == "power"
        else f"({left}) {operators[operation]} ({right})"
    )
    return (expression, right if operation == "divide" else None)


def _fused_derivative_expressions(
    step: FusedElementwiseStep, value: str, result: str, *, storage_type: str
) -> tuple[str, str | None]:
    """Return derivatives with respect to current and an external operand."""
    operation, scalar, reverse, operand_index = step
    operand = _fused_operand_expression(step, value, storage_type=storage_type)
    if operation == "identity":
        return ("1.0", None)
    if operation == "negate":
        return ("-1.0", None)
    if operation == "abs":
        return (
            f"(isnan({value}) ? ({value}) : (({value}) > 0.0 ? 1.0 : (({value}) < 0.0 ? -1.0 : 0.0)))",
            None,
        )
    if operation == "sqrt":
        return (f"(0.5 / ({result}))", None)
    if operation == "exp":
        return (f"exp({value})", None)
    if operation == "log":
        return (f"(1.0 / ({value}))", None)
    if operation == "sin":
        return (f"cos({value})", None)
    if operation == "cos":
        return (f"-sin({value})", None)
    if operation == "tan":
        cosine = f"cos({value})"
        return (f"(1.0 / (({cosine}) * ({cosine})))", None)
    if operation == "arcsin":
        return (f"(1.0 / sqrt(1.0 - ({value}) * ({value})))", None)
    if operation == "arccos":
        return (f"(-1.0 / sqrt(1.0 - ({value}) * ({value})))", None)
    if operation == "arctan":
        reciprocal = f"(1.0 / fabs({value}))"
        return (
            f"(isinf({value}) ? 0.0 : (fabs({value}) <= 1.0 ? 1.0 / (1.0 + ({value}) * ({value})) : (({reciprocal}) * ({reciprocal})) / (1.0 + ({reciprocal}) * ({reciprocal}))))",
            None,
        )
    if operation == "sinh":
        return (f"cosh({value})", None)
    if operation == "cosh":
        return (f"sinh({value})", None)
    if operation == "arcsinh":
        reciprocal = f"(1.0 / fabs({value}))"
        return (
            f"(isinf({value}) ? 0.0 : (fabs({value}) <= 1.0 ? 1.0 / sqrt(1.0 + ({value}) * ({value})) : ({reciprocal}) / sqrt(1.0 + ({reciprocal}) * ({reciprocal}))))",
            None,
        )
    if operation == "arccosh":
        return (
            f"(isinf({value}) ? 0.0 : 1.0 / (sqrt(({value}) - 1.0) * sqrt(({value}) + 1.0)))",
            None,
        )
    if operation == "arctanh":
        return (f"(1.0 / (1.0 - ({value}) * ({value})))", None)
    if operation == "sign":
        return (f"(isnan({value}) ? ({value}) : 0.0)", None)
    if operation == "relu":
        return (f"(isnan({value}) ? ({value}) : (({value}) > 0.0 ? 1.0 : 0.0))", None)
    if operation in {"sigmoid", "softplus"}:
        z = f"(({value}) >= 0.0 ? exp(-({value})) : exp({value}))"
        sigmoid_derivative = f"(({z}) / ((1.0 + ({z})) * (1.0 + ({z}))))"
        if operation == "sigmoid":
            return (sigmoid_derivative, None)
        sigmoid_value = _fused_unary_expression("sigmoid", value)
        return (sigmoid_value, None)
    if operation == "tanh":
        z = f"exp(-2.0 * fabs({value}))"
        return (f"(4.0 * ({z}) / ((1.0 + ({z})) * (1.0 + ({z}))))", None)
    if operation == "power" and operand is not None:
        if reverse:
            return (f"(({result}) * log({operand}))", None)
        if scalar == 0:
            return ("0.0", None)
        return (f"(({operand}) * pow(({value}), ({operand}) - 1.0))", None)
    if operation not in _FUSED_BINARY_OPERATIONS or operand is None:
        raise ValueError(f"Unsupported fused backward operation {operation!r}")
    if operand_index == -1:
        derivatives = {
            "add": "2.0",
            "subtract": "0.0",
            "multiply": f"(2.0 * ({value}))",
        }
        if operation == "divide":
            return ("0.0", None)
        derivative = derivatives.get(operation)
        if derivative is None:
            raise ValueError(f"Self-{operation} is not fused in backward")
        return (derivative, None)
    if operation == "add":
        return ("1.0", None if scalar is not None else "1.0")
    if operation == "subtract":
        current = "-1.0" if reverse else "1.0"
        external = "1.0" if reverse else "-1.0"
        return (current, None if scalar is not None else external)
    if operation == "multiply":
        return (operand, None if scalar is not None else value)
    if operation == "divide":
        if reverse:
            current = f"(-({result}) / ({value}))"
            external = f"(1.0 / ({value}))"
        else:
            current = f"(1.0 / ({operand}))"
            external = f"(-({result}) / ({operand}))"
        return (current, None if scalar is not None else external)
    raise ValueError(f"Unsupported fused backward operation {operation!r}")


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
    *,
    storage_type: str,
) -> tuple[str, str | None]:
    """Return range-stable VJP contributions for one fused step."""
    operation, scalar, reverse, operand_index = step
    operand = _fused_operand_expression(step, value, storage_type=storage_type)
    if operation == "power" and operand is not None:
        if reverse:
            logarithm = f"log({operand})"
            contribution = f"(({upstream}) == 0.0 || ({logarithm}) == 0.0 ? 0.0 : copysign(exp(log(fabs({upstream})) + ({value}) * ({logarithm}) + log(fabs({logarithm}))), ({upstream}) * ({logarithm})))"
            return (contribution, None)
        exponent = operand
        magnitude = f"copysign(exp(log(fabs({upstream})) + log(fabs({exponent})) + (({exponent}) - 1.0) * log(fabs({value}))), {{sign}})"
        if scalar is not None:
            sign = f"({upstream}) * ({exponent})"
            if float(scalar).is_integer() and (int(scalar) - 1) % 2:
                sign = f"({sign}) * (({value}) < 0.0 ? -1.0 : 1.0)"
            if scalar == 0:
                return ("0.0", None)
            if scalar == 1:
                return (upstream, None)
            contribution = (
                f"(({upstream}) == 0.0 || ({value}) == 0.0 ? 0.0 : "
                + magnitude.format(sign=sign)
                + ")"
            )
            return (contribution, None)
        even_integer = (
            f"(({exponent}) == floor({exponent}) && fmod({exponent}, 2.0) == 0.0)"
        )
        sign = f"(({upstream}) * ({exponent}) * ({even_integer} ? (({value}) < 0.0 ? -1.0 : 1.0) : 1.0))"
        contribution = (
            f"(({exponent}) == 0.0 ? 0.0 : (({exponent}) == 1.0 ? ({upstream}) : ((({upstream}) == 0.0 || ({value}) == 0.0) ? 0.0 : "
            + magnitude.format(sign=sign)
            + ")))"
        )
        return (contribution, None)
    if operation == "divide" and operand is not None:
        if operand_index == -1:
            return ("0.0", None)
        if reverse:
            zero = f"(({upstream}) == 0.0 || ({operand}) == 0.0)"
            sign = f"(signbit({upstream}) != signbit({operand}) ? 1.0 : -1.0)"
            stable = f"({zero} ? 0.0 : ({sign}) * exp(log(fabs({upstream})) + log(fabs({operand})) - 2.0 * log(fabs({value}))))"
            direct = f"(-({upstream}) * ({operand}) / (({value}) * ({value})))"
            square = f"(({value}) * ({value}))"
            current = f"(isfinite({square}) && ({square}) != 0.0 && ({zero} || (({direct}) != 0.0 && isfinite({direct}))) ? ({direct}) : ({stable}))"
            external = f"(({upstream}) / ({value}))"
        else:
            current = f"(({upstream}) / ({operand}))"
            zero = f"(({upstream}) == 0.0 || ({value}) == 0.0)"
            sign = f"(signbit({upstream}) != signbit({value}) ? 1.0 : -1.0)"
            stable = f"({zero} ? 0.0 : ({sign}) * exp(log(fabs({upstream})) + log(fabs({value})) - 2.0 * log(fabs({operand}))))"
            direct = f"(-({upstream}) * ({value}) / (({operand}) * ({operand})))"
            square = f"(({operand}) * ({operand}))"
            external = f"(isfinite({square}) && ({square}) != 0.0 && ({zero} || (({direct}) != 0.0 && isfinite({direct}))) ? ({direct}) : ({stable}))"
        return (current, None if scalar is not None else external)
    current_derivative, operand_derivative = _fused_derivative_expressions(
        step, value, result, storage_type=storage_type
    )
    current = f"({upstream}) * ({current_derivative})"
    external = (
        None if operand_derivative is None else f"({upstream}) * ({operand_derivative})"
    )
    return (current, external)
