"""Shared NumPy/CuPy kernels backed by native array storage."""

from __future__ import annotations

import hashlib
import importlib
import math
from collections.abc import Sequence
from contextlib import nullcontext
from functools import lru_cache
from typing import TYPE_CHECKING, Any, cast

from ..shape import Shape
from ..storage import CudaStorage, NumPyStorage, Storage, StorageKind
from ..strides import Strides

if TYPE_CHECKING:
    from .._typing import Scalar, TensorIndex
    from ..dtype import DataType
    from ..tensor import Tensor
    from . import (
        ArgExtremumOperation,
        BinaryOperation,
        ComparisonOperation,
        DifferentiableReductionOperation,
        ExtremumOperation,
        FusedElementwiseStep,
        LossReduction,
        NormalizationOperation,
        ReductionOperation,
        UnaryOperation,
    )


def _view(tensor: Tensor, numpy: Any) -> Any:
    """Return compact logical values as a backend-native array.

    Provider kernels operate on compact arrays. Tensor metadata remains the
    source of truth, so a future non-compact layout is gathered before crossing
    this boundary rather than being reshaped as if logical and physical
    positions were identical.
    """
    storage = tensor._logical_storage_for(_array_kind(numpy))
    return storage.buffer.reshape(tensor.shape)


def _array_kind(numpy: Any) -> StorageKind:
    """Return the storage kind associated with an imported array provider."""
    return "cuda" if numpy.__name__.split(".", 1)[0] == "cupy" else "numpy"


def _cuda_integer(dtype: DataType) -> bool:
    """Return whether exact integer semantics require the Python fallback."""
    from . import get_backend

    return get_backend() == "cuda" and dtype.kind == "integer"


def _errstate(numpy: Any, **settings: str) -> Any:
    """Use NumPy warning controls when the selected array module provides them."""
    factory = getattr(numpy, "errstate", None)
    if factory is None:
        return nullcontext()
    return factory(**settings)


def _operand(value: Tensor | Scalar, dtype: DataType, numpy: Any) -> Any:
    """Return an array operand with Python-reference working precision."""
    from ..tensor import Tensor

    if _array_kind(numpy) == "cuda" and dtype.kind == "integer":
        # CuPy has no object dtype with Python's unbounded intermediate integer
        # semantics. Let the reference backend handle these operations.
        raise TypeError("CUDA integer kernels require the Python fallback")

    if isinstance(value, Tensor):
        result = _view(value, numpy)
    else:
        result = value
    working_dtype = numpy.float64 if dtype.kind == "floating" else object
    return numpy.asarray(result, dtype=working_dtype)


def _storage(
    result: Any,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
    numpy: Any,
) -> Storage | None:
    """Retain a native array result without changing tensor semantics."""
    backend = _array_kind(numpy)
    flattened = numpy.asarray(result).reshape(-1)
    if dtype.kind == "integer":
        if backend == "cuda":
            return None
        try:
            contiguous = numpy.asarray(flattened, dtype=numpy.dtype(dtype.name))
        except (OverflowError, TypeError, ValueError):
            return None
    else:
        try:
            flattened = numpy.asarray(flattened, dtype=numpy.float64)
        except (OverflowError, TypeError, ValueError):
            return None
        target_dtype = numpy.dtype(dtype.name)
        if target_dtype.itemsize < numpy.dtype(numpy.float64).itemsize:
            finite = numpy.isfinite(flattened)
            outside_range = numpy.abs(flattened) > numpy.finfo(target_dtype).max
            if bool(numpy.any(finite & outside_range)):
                return None
        with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
            contiguous = numpy.asarray(flattened, dtype=target_dtype)
    storage: Storage
    if backend == "cuda":
        storage = CudaStorage(contiguous, dtype)
    else:
        storage = NumPyStorage(contiguous, dtype)
    if storage.size != _shape_size(output_shape):
        raise RuntimeError("Array kernel returned an unexpected result size")
    return storage


def _optimizer_batch_values(
    tensors: Sequence[Tensor],
    numpy: Any,
) -> Any | None:
    """Concatenate compatible optimizer tensors into one native flat buffer."""
    if not tensors:
        return None
    dtype = tensors[0].dtype
    if any(tensor.dtype != dtype for tensor in tensors):
        return None
    arrays = tuple(
        _view(tensor, numpy).astype(numpy.float64, copy=False).reshape(-1)
        for tensor in tensors
    )
    return numpy.concatenate(arrays)


def _optimizer_batch_compatible(
    *groups: Sequence[Tensor],
) -> bool:
    """Return whether optimizer tensor groups can share one native update."""
    if not groups or not groups[0]:
        return False
    count = len(groups[0])
    if any(len(group) != count for group in groups):
        return False
    dtype = groups[0][0].dtype
    for tensors in zip(*groups):
        shape = tensors[0].shape
        if any(
            tensor.shape != shape or tensor.dtype != dtype
            for tensor in tensors
        ):
            return False
    return True


def _split_optimizer_storage(
    result: Any,
    references: Sequence[Tensor],
    numpy: Any,
) -> tuple[Storage, ...] | None:
    """Retain slices of one batched result without copying them again."""
    if not references:
        return ()
    dtype = references[0].dtype
    total = sum(reference.size for reference in references)
    storage = _storage(
        result,
        dtype=dtype,
        output_shape=(total,),
        numpy=numpy,
    )
    if storage is None:
        return None
    storages: list[Storage] = []
    offset = 0
    for reference in references:
        end = offset + reference.size
        buffer = storage.buffer[offset:end]
        if storage.kind == "cuda":
            storages.append(CudaStorage(buffer, dtype))
        else:
            storages.append(NumPyStorage(buffer, dtype))
        offset = end
    return tuple(storages)


def binary(
    operation: BinaryOperation,
    left: Tensor | Scalar,
    right: Tensor | Scalar,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Run a broadcasting NumPy binary kernel."""
    numpy = _numpy()
    try:
        left_array = _operand(left, dtype, numpy)
        right_array = _operand(right, dtype, numpy)
    except (OverflowError, TypeError, ValueError):
        return None
    functions = {
        "add": numpy.add,
        "subtract": numpy.subtract,
        "multiply": numpy.multiply,
        "divide": numpy.true_divide,
        "power": numpy.power,
    }
    if operation == "divide" and bool(numpy.any(right_array == 0)):
        raise ZeroDivisionError("Division by zero")
    with _errstate(
        numpy,
        divide="ignore",
        over="ignore",
        under="ignore",
        invalid="ignore",
    ):
        result = functions[operation](left_array, right_array)

    if operation == "power" and dtype.kind == "floating" and _finite_operands(
        left_array,
        right_array,
        numpy=numpy,
    ) and not bool(numpy.all(numpy.isfinite(result))):
        return None
    return _storage(
        result,
        dtype=dtype,
        output_shape=output_shape,
        numpy=numpy,
    )


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
    if operation == "power" and scalar is not None and not reverse:
        if scalar != 0 and scalar < 1:
            return ((f"({value}) == 0.0", 14),)
    return ()


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

        assert scalar is not None
        if scalar == 0:
            return "0.0", None
        if scalar == 1:
            return upstream, None
        scalar_literal = _fused_operand_expression(step, value)
        assert scalar_literal is not None
        sign = f"({upstream}) * ({scalar_literal})"
        if float(scalar).is_integer() and (int(scalar) - 1) % 2:
            sign = f"({sign}) * (({value}) < 0.0 ? -1.0 : 1.0)"
        contribution = (
            f"(({upstream}) == 0.0 || ({value}) == 0.0 ? 0.0 : "
            f"copysign(exp(log(fabs({upstream})) + log(fabs({scalar_literal})) "
            f"+ (({scalar_literal}) - 1.0) * log(fabs({value}))), {sign})"
        )
        contribution += ")"
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

    external_rows = {}
    next_row = len(steps) + 1
    for index, (_, scalar, _, operand_index) in enumerate(steps):
        if scalar is None and operand_index is not None and operand_index >= 0:
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
        if operand_contribution is not None:
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
    from . import get_backend

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
) -> tuple[Storage, ...] | None:
    """Evaluate safe same-shape VJPs for a fused expression chain."""
    from . import get_backend

    if (
        get_backend() != "cuda"
        or dtype.kind != "floating"
        or not values
        or len(steps) < 2
        or grad.shape != output_shape
    ):
        return None
    cupy = _numpy()
    size = grad.size
    external_count = sum(
        scalar is None and operand_index is not None and operand_index >= 0
        for _, scalar, _, operand_index in steps
    )
    row_count = len(steps) + 1 + external_count
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


def negate(value: Tensor, *, dtype: DataType) -> Storage | None:
    """Run elementwise NumPy negation."""
    numpy = _numpy()
    try:
        operand = _operand(value, dtype, numpy)
    except (TypeError, ValueError):
        return None
    result = numpy.negative(operand)
    return _storage(
        result,
        dtype=dtype,
        output_shape=value.shape,
        numpy=numpy,
    )


def unary(
    operation: UnaryOperation,
    value: Tensor,
    *,
    dtype: DataType,
) -> Storage | None:
    """Run an elementwise unary kernel while preserving public domains."""
    if dtype.kind == "integer":
        return None
    numpy = _numpy()
    try:
        values = _view(value, numpy).astype(numpy.float64, copy=False)
    except (TypeError, ValueError):
        return None

    if operation == "sqrt" and bool(numpy.any(values < 0.0)):
        raise ValueError("sqrt is only defined for non-negative values")
    if operation == "log" and bool(numpy.any(values <= 0.0)):
        raise ValueError("log is only defined for positive values")
    if operation in {"arcsin", "arccos"} and bool(
        numpy.any((values < -1.0) | (values > 1.0))
    ):
        raise ValueError(
            f"{operation} is only defined for values between -1 and 1"
        )
    if operation == "arccosh" and bool(numpy.any(values < 1.0)):
        raise ValueError(
            "arccosh is only defined for values greater than or equal to 1"
        )
    if operation == "arctanh":
        outside = (~numpy.isnan(values)) & (
            (values <= -1.0) | (values >= 1.0)
        )
        if bool(numpy.any(outside)):
            raise ValueError(
                "arctanh is only defined for values strictly between -1 and 1"
            )
    if operation in {"sin", "cos", "tan"} and bool(
        numpy.any(numpy.isinf(values))
    ):
        return None

    functions = {
        "abs": numpy.abs,
        "sqrt": numpy.sqrt,
        "exp": numpy.exp,
        "log": numpy.log,
        "sin": numpy.sin,
        "cos": numpy.cos,
        "tan": numpy.tan,
        "arcsin": numpy.arcsin,
        "arccos": numpy.arccos,
        "arctan": numpy.arctan,
        "sinh": numpy.sinh,
        "cosh": numpy.cosh,
        "arcsinh": numpy.arcsinh,
        "arccosh": numpy.arccosh,
        "arctanh": numpy.arctanh,
        "sign": numpy.sign,
        "tanh": numpy.tanh,
    }
    with _errstate(
        numpy,
        divide="ignore",
        over="ignore",
        under="ignore",
        invalid="ignore",
    ):
        if operation == "relu":
            result = numpy.where(numpy.isnan(values), values, numpy.maximum(values, 0.0))
        elif operation == "sigmoid":
            magnitude = numpy.exp(-numpy.abs(values))
            result = numpy.where(
                values >= 0.0,
                1.0 / (1.0 + magnitude),
                magnitude / (1.0 + magnitude),
            )
        elif operation == "softplus":
            result = numpy.log1p(numpy.exp(-numpy.abs(values))) + numpy.maximum(
                values,
                0.0,
            )
        else:
            result = functions[operation](values)
    return _storage(
        result,
        dtype=dtype,
        output_shape=value.shape,
        numpy=numpy,
    )


def unary_gradient(
    operation: UnaryOperation,
    grad: Tensor,
    value: Tensor,
) -> Storage | None:
    """Run the vector-Jacobian product for an elementwise unary operation."""
    numpy = _numpy()
    try:
        upstream = _view(grad, numpy).astype(numpy.float64, copy=False)
        values = _view(value, numpy).astype(numpy.float64, copy=False)
    except (TypeError, ValueError):
        return None
    if upstream.shape != values.shape:
        return None

    if operation == "sqrt":
        if bool(numpy.any(values == 0.0)):
            raise ValueError("sqrt derivative is undefined at zero")
        if bool(numpy.any(values < 0.0)):
            return None
    if operation in {"arcsin", "arccos"}:
        if bool(numpy.any((values == -1.0) | (values == 1.0))):
            raise ValueError(
                f"{operation} derivative is undefined at -1 and 1"
            )
        if bool(numpy.any((values < -1.0) | (values > 1.0))):
            return None
    if operation == "arccosh":
        if bool(numpy.any(values == 1.0)):
            raise ValueError("arccosh derivative is undefined at 1")
        if bool(numpy.any(values < 1.0)):
            return None
    if operation == "sign" and bool(numpy.any(values == 0.0)):
        raise ValueError("sign derivative is undefined at zero")
    if operation in {"sin", "cos", "tan"} and bool(
        numpy.any(numpy.isinf(values))
    ):
        return None

    with _errstate(

        numpy,
        divide="ignore",
        over="ignore",
        under="ignore",
        invalid="ignore",
    ):
        if operation == "abs":
            derivative = numpy.where(
                numpy.isnan(values),
                numpy.nan,
                numpy.where(values > 0.0, 1.0, numpy.where(values < 0.0, -1.0, 0.0)),
            )
        elif operation == "sqrt":
            derivative = 1.0 / (2.0 * numpy.sqrt(values))
        elif operation == "exp":
            derivative = numpy.exp(values)
        elif operation == "log":
            derivative = 1.0 / values
        elif operation == "sin":
            derivative = numpy.cos(values)
        elif operation == "cos":
            derivative = -numpy.sin(values)
        elif operation == "tan":
            cosine = numpy.cos(values)
            if bool(numpy.any(numpy.abs(cosine) < numpy.finfo(numpy.float64).eps)):
                return None
            derivative = 1.0 / (cosine * cosine)
        elif operation == "arcsin":
            derivative = 1.0 / numpy.sqrt(1.0 - values * values)
        elif operation == "arccos":
            derivative = -1.0 / numpy.sqrt(1.0 - values * values)
        elif operation == "arctan":
            reciprocal = 1.0 / numpy.abs(values)
            derivative = numpy.where(
                numpy.isinf(values),
                0.0,
                numpy.where(
                    numpy.abs(values) <= 1.0,
                    1.0 / (1.0 + values * values),
                    (reciprocal * reciprocal) / (1.0 + reciprocal * reciprocal),
                ),
            )
        elif operation == "sinh":
            derivative = numpy.cosh(values)
        elif operation == "cosh":
            derivative = numpy.sinh(values)
        elif operation == "arcsinh":
            reciprocal = 1.0 / numpy.abs(values)
            derivative = numpy.where(
                numpy.isinf(values),
                0.0,
                numpy.where(
                    numpy.abs(values) <= 1.0,
                    1.0 / numpy.sqrt(1.0 + values * values),
                    reciprocal / numpy.sqrt(1.0 + reciprocal * reciprocal),
                ),
            )
        elif operation == "arccosh":
            derivative = numpy.where(
                numpy.isinf(values),
                0.0,
                1.0 / (numpy.sqrt(values - 1.0) * numpy.sqrt(values + 1.0)),
            )
        elif operation == "arctanh":
            derivative = 1.0 / (1.0 - values * values)
        elif operation == "sign":
            derivative = numpy.where(numpy.isnan(values), numpy.nan, 0.0)
        elif operation == "relu":
            derivative = numpy.where(
                numpy.isnan(values),
                numpy.nan,
                numpy.where(values > 0.0, 1.0, 0.0),
            )
        elif operation in {"sigmoid", "softplus"}:
            magnitude = numpy.exp(-numpy.abs(values))
            if operation == "sigmoid":
                derivative = magnitude / ((1.0 + magnitude) ** 2.0)
            else:
                derivative = numpy.where(
                    values >= 0.0,
                    1.0 / (1.0 + magnitude),
                    magnitude / (1.0 + magnitude),
                )
        else:
            magnitude = numpy.exp(-2.0 * numpy.abs(values))
            derivative = 4.0 * magnitude / ((1.0 + magnitude) ** 2.0)
        result = upstream * derivative
    return _storage(
        result,
        dtype=grad.dtype,
        output_shape=value.shape,
        numpy=numpy,
    )


def _normalization_terms(
    values: Any,
    axis: int | tuple[int, ...],
    numpy: Any,
) -> tuple[Any, Any, Any]:
    """Return stable maxima, corrections, and probabilities."""
    maximum = numpy.max(values, axis=axis, keepdims=True)
    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        deltas = values - maximum
        maxima = numpy.sum(deltas == 0.0, axis=axis, keepdims=True)
        tails = numpy.sum(
            numpy.where(deltas == 0.0, 0.0, numpy.exp(deltas)),
            axis=axis,
            keepdims=True,
        )
        correction = numpy.log(maxima) + numpy.log1p(tails / maxima)
        probabilities = numpy.exp(deltas - correction)
    return maximum, correction, probabilities


def normalization(
    operation: NormalizationOperation,
    value: Tensor,
    axis: int,
    *,
    dtype: DataType,
) -> Storage | None:
    """Run fused softmax or log-softmax on finite values."""
    numpy = _numpy()
    values = _view(value, numpy).astype(numpy.float64, copy=False)
    maximum, correction, probabilities = _normalization_terms(
        values,
        axis,
        numpy,
    )
    if operation == "softmax":
        result = probabilities
    else:
        with _errstate(numpy, over="ignore", invalid="ignore"):
            result = values - maximum - correction
    if not _finite_operands(values, probabilities, result, numpy=numpy):
        return None
    return _storage(
        result,
        dtype=dtype,
        output_shape=value.shape,
        numpy=numpy,
    )


def normalization_gradient(
    operation: NormalizationOperation,
    grad: Tensor,
    value: Tensor,
    axis: int,
) -> Storage | None:
    """Run a softmax-family VJP away from dominant cancellation."""
    numpy = _numpy()
    upstream = _view(grad, numpy).astype(numpy.float64, copy=False)
    values = _view(value, numpy).astype(numpy.float64, copy=False)
    if upstream.shape != values.shape:
        return None
    _, _, probabilities = _normalization_terms(values, axis, numpy)
    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        if operation == "softmax":
            spread = numpy.max(upstream, axis=axis, keepdims=True) - numpy.min(
                upstream,
                axis=axis,
                keepdims=True,
            )
            expectation = numpy.sum(
                upstream * probabilities,
                axis=axis,
                keepdims=True,
            )
            result = probabilities * (upstream - expectation)
            result = numpy.where(spread == 0.0, 0.0, result)
        else:
            total = numpy.sum(upstream, axis=axis, keepdims=True)
            result = upstream - probabilities * total
    valid = (
        numpy.all(numpy.isfinite(values))
        & numpy.all(numpy.isfinite(upstream))
        & numpy.all(numpy.isfinite(probabilities))
        & ~numpy.any(numpy.max(probabilities, axis=axis) > 0.95)
        & numpy.all(numpy.isfinite(result))
    )
    if not bool(valid):
        return None
    return _storage(
        result,
        dtype=grad.dtype,
        output_shape=value.shape,
        numpy=numpy,
    )


def logsumexp(
    value: Tensor,
    axes: tuple[int, ...],
    *,
    keepdims: bool,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Run a stable log-sum-exp reduction on finite values."""
    numpy = _numpy()
    values = _view(value, numpy).astype(numpy.float64, copy=False)
    maximum, correction, probabilities = _normalization_terms(
        values,
        axes,
        numpy,
    )
    with _errstate(numpy, over="ignore", invalid="ignore"):
        result = maximum + correction
    if not keepdims and axes:
        result = numpy.squeeze(result, axis=axes)
    if not _finite_operands(values, probabilities, result, numpy=numpy):
        return None
    return _storage(
        result,
        dtype=dtype,
        output_shape=output_shape,
        numpy=numpy,
    )


def logsumexp_gradient(
    grad: Tensor,
    value: Tensor,
    axes: tuple[int, ...],
    *,
    keepdims: bool,
) -> Storage | None:
    """Run a stable log-sum-exp VJP on finite values."""
    numpy = _numpy()
    values = _view(value, numpy).astype(numpy.float64, copy=False)
    upstream = _view(grad, numpy).astype(numpy.float64, copy=False)
    _, _, probabilities = _normalization_terms(values, axes, numpy)
    expanded_shape = tuple(
        1 if dimension in axes else size
        for dimension, size in enumerate(value.shape)
    )
    try:
        expanded = upstream.reshape(expanded_shape)
    except ValueError:
        return None
    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        result = expanded * probabilities
    if not _finite_operands(
        values,
        upstream,
        probabilities,
        result,
        numpy=numpy,
    ):
        return None
    return _storage(
        result,
        dtype=grad.dtype,
        output_shape=value.shape,
        numpy=numpy,
    )


def _reduce_losses(values: Any, reduction: LossReduction, numpy: Any) -> Any:
    """Reduce non-negative losses without overflowing an ordinary mean."""
    if reduction == "none":
        return values
    if reduction == "sum":
        with _errstate(numpy, over="ignore", invalid="ignore"):
            return numpy.asarray([numpy.sum(values)])
    scale = numpy.max(values)
    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        exceptional = (scale == 0.0) | ~numpy.isfinite(scale)
        safe_scale = numpy.where(exceptional, 1.0, scale)
        stable = safe_scale * numpy.mean(values / safe_scale)
        result = numpy.where(exceptional, numpy.mean(values), stable)
    return numpy.asarray(result).reshape(1)


def distributions_valid(targets: Tensor, axis: int) -> bool:
    """Return whether dense targets are finite normalized probabilities."""
    numpy = _numpy()
    values = _view(targets, numpy).astype(numpy.float64, copy=False)
    valid_values = numpy.all(
        numpy.isfinite(values) & (values >= 0.0) & (values <= 1.0)
    )
    totals = numpy.sum(values, axis=axis)
    class_count = targets.shape[axis]
    epsilon = numpy.finfo(numpy.float64).eps
    accumulated_error = (
        class_count
        * epsilon
        * numpy.sum(numpy.abs(values), axis=axis)
    )
    tolerance = numpy.maximum(
        1e-7,
        1e-7 * numpy.maximum(numpy.abs(totals), 1.0),
    )
    valid_totals = numpy.all(
        numpy.abs(totals - 1.0) + accumulated_error <= tolerance
    )
    return bool(valid_values & valid_totals)


def one_hot_targets(
    logits: Tensor,
    targets: Tensor,
    axis: int,
) -> Storage | None:
    """Expand validated class indices directly into native dense storage."""
    numpy = _numpy()
    values = _view(targets, numpy).astype(numpy.float64, copy=False)
    class_count = logits.shape[axis]
    with _errstate(numpy, invalid="ignore"):
        integral = values == numpy.floor(values)
    valid = numpy.all(
        numpy.isfinite(values)
        & integral
        & (values >= 0.0)
        & (values < class_count)
    )
    if not bool(valid):
        return None

    sample_shape = logits.shape[:axis] + logits.shape[axis + 1:]
    indices = values.astype(numpy.int64).reshape(sample_shape)
    expanded_indices = numpy.expand_dims(indices, axis=axis)
    result = numpy.zeros(logits.shape, dtype=numpy.float64)
    numpy.put_along_axis(result, expanded_indices, 1.0, axis=axis)
    return _storage(
        result,
        dtype=logits.dtype,
        output_shape=logits.shape,
        numpy=numpy,
    )


def cross_entropy(
    logits: Tensor,
    targets: Tensor,
    axis: int,
    *,
    reduction: LossReduction,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Run fused dense multiclass cross-entropy."""
    numpy = _numpy()
    values = _view(logits, numpy).astype(numpy.float64, copy=False)
    weights = _view(targets, numpy).astype(numpy.float64, copy=False)
    if values.shape != weights.shape:
        return None
    maximum, correction, probabilities = _normalization_terms(
        values,
        axis,
        numpy,
    )
    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        log_probabilities = values - maximum - correction
        contributions = numpy.where(
            weights == 0.0,
            0.0,
            -weights * log_probabilities,
        )
        losses = numpy.sum(contributions, axis=axis)
    valid = (
        numpy.all(numpy.isfinite(values))
        & numpy.all(numpy.isfinite(weights))
        & numpy.all(numpy.isfinite(probabilities))
        & ~numpy.any(numpy.isnan(losses))
    )
    if not bool(valid):
        return None
    result = _reduce_losses(losses, reduction, numpy)
    return _storage(
        result,
        dtype=dtype,
        output_shape=output_shape,
        numpy=numpy,
    )


def cross_entropy_gradient(
    grad: Tensor,
    logits: Tensor,
    targets: Tensor,
    axis: int,
    *,
    reduction: LossReduction,
) -> tuple[Storage, Storage] | None:
    """Run fused dense multiclass cross-entropy VJPs."""
    numpy = _numpy()
    values = _view(logits, numpy).astype(numpy.float64, copy=False)
    weights = _view(targets, numpy).astype(numpy.float64, copy=False)
    upstream = _view(grad, numpy).astype(numpy.float64, copy=False)
    if values.shape != weights.shape:
        return None
    maximum, correction, probabilities = _normalization_terms(
        values,
        axis,
        numpy,
    )
    sample_shape = values.shape[:axis] + values.shape[axis + 1:]
    try:
        if reduction == "none":
            expanded_upstream = upstream.reshape(sample_shape)
        else:
            scale = 1.0 / _shape_size(sample_shape) if (
                reduction == "mean" and _shape_size(sample_shape)
            ) else 1.0
            expanded_upstream = numpy.broadcast_to(
                upstream.reshape(()),
                sample_shape,
            ) * scale
        expanded_upstream = numpy.expand_dims(expanded_upstream, axis=axis)
    except (TypeError, ValueError):
        return None
    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        target_mass = numpy.sum(weights, axis=axis, keepdims=True)
        log_probabilities = values - maximum - correction
        logits_result = expanded_upstream * (
            target_mass * probabilities - weights
        )
        targets_result = -expanded_upstream * log_probabilities
        zero_upstream = expanded_upstream == 0.0
        logits_result = numpy.where(zero_upstream, 0.0, logits_result)
        targets_result = numpy.where(zero_upstream, 0.0, targets_result)
    valid = (
        numpy.all(numpy.isfinite(values))
        & numpy.all(numpy.isfinite(weights))
        & numpy.all(numpy.isfinite(upstream))
        & numpy.all(numpy.isfinite(probabilities))
        & ~numpy.any(numpy.max(probabilities, axis=axis) > 0.95)
    )
    if not bool(valid):
        return None
    logits_storage = _storage(
        logits_result,
        dtype=grad.dtype,
        output_shape=logits.shape,
        numpy=numpy,
    )
    targets_storage = _storage(
        targets_result,
        dtype=grad.dtype,
        output_shape=targets.shape,
        numpy=numpy,
    )
    if logits_storage is None or targets_storage is None:
        return None
    return logits_storage, targets_storage


def binary_cross_entropy(
    prediction: Tensor,
    target: Tensor,
    *,
    from_logits: bool,
    reduction: LossReduction,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Run fused binary cross-entropy."""
    numpy = _numpy()
    values = _view(prediction, numpy).astype(numpy.float64, copy=False)
    targets = _view(target, numpy).astype(numpy.float64, copy=False)
    if values.shape != targets.shape:
        return None
    if from_logits:
        if not _finite_operands(values, targets, numpy=numpy):
            return None
        with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
            correction = numpy.log1p(numpy.exp(-numpy.abs(values)))
            losses = numpy.where(
                values >= 0.0,
                (1.0 - targets) * values + correction,
                -targets * values + correction,
            )
    else:
        invalid = (~numpy.isfinite(values)) | (
            (values < 0.0) | (values > 1.0)
        )
        target_valid = numpy.all(numpy.isfinite(targets))
        value_valid = ~numpy.any(invalid)
        status = int(
            numpy.asarray(~target_valid, dtype=numpy.uint8)
            | (
                numpy.asarray(~value_valid, dtype=numpy.uint8)
                * numpy.uint8(2)
            )
        )
        if status & 1:
            return None
        if status & 2:
            raise ValueError(
                "binary cross-entropy probabilities must be between 0 and 1"
            )
        with _errstate(numpy, divide="ignore", invalid="ignore"):
            losses = numpy.where(
                values == 0.0,
                numpy.where(targets == 0.0, 0.0, numpy.inf),
                numpy.where(
                    values == 1.0,
                    numpy.where(targets == 1.0, 0.0, numpy.inf),
                    -targets * numpy.log(values)
                    - (1.0 - targets) * numpy.log1p(-values),
                ),
            )
    result = _reduce_losses(losses, reduction, numpy)
    return _storage(
        result,
        dtype=dtype,
        output_shape=output_shape,
        numpy=numpy,
    )


def binary_cross_entropy_gradient(
    grad: Tensor,
    prediction: Tensor,
    target: Tensor,
    *,
    from_logits: bool,
    reduction: LossReduction,
) -> tuple[Storage, Storage] | None:
    """Run fused binary cross-entropy VJPs."""
    numpy = _numpy()
    values = _view(prediction, numpy).astype(numpy.float64, copy=False)
    targets = _view(target, numpy).astype(numpy.float64, copy=False)
    upstream = _view(grad, numpy).astype(numpy.float64, copy=False)
    if values.shape != targets.shape:
        return None
    if reduction == "none":
        if upstream.shape != values.shape:
            return None
        expanded_upstream = upstream
    else:
        scale = 1.0 / values.size if reduction == "mean" and values.size else 1.0
        try:
            expanded_upstream = numpy.broadcast_to(
                upstream.reshape(()),
                values.shape,
            ) * scale
        except ValueError:
            return None

    with _errstate(

        numpy,
        divide="ignore",
        over="ignore",
        under="ignore",
        invalid="ignore",
    ):
        if from_logits:
            if not _finite_operands(
                values,
                targets,
                upstream,
                numpy=numpy,
            ):
                return None
            magnitude = numpy.exp(-numpy.abs(values))
            sigmoid = numpy.where(
                values >= 0.0,
                1.0 / (1.0 + magnitude),
                magnitude / (1.0 + magnitude),
            )
            prediction_derivative = sigmoid - targets
            target_derivative = -values
        else:
            invalid = (~numpy.isfinite(values)) | (
                (values < 0.0) | (values > 1.0)
            )
            valid = (
                numpy.all(numpy.isfinite(targets))
                & numpy.all(numpy.isfinite(upstream))
                & ~numpy.any(invalid)
            )
            if not bool(valid):
                return None
            prediction_derivative = numpy.where(
                values == 0.0,
                numpy.where(targets == 0.0, 1.0, -numpy.inf),
                numpy.where(
                    values == 1.0,
                    numpy.where(targets == 1.0, -1.0, numpy.inf),
                    (values - targets) / (values * (1.0 - values)),
                ),
            )
            target_derivative = numpy.where(
                values == 0.0,
                numpy.inf,
                numpy.where(
                    values == 1.0,
                    -numpy.inf,
                    numpy.log1p(-values) - numpy.log(values),
                ),
            )
        prediction_result = expanded_upstream * prediction_derivative
        target_result = expanded_upstream * target_derivative
        zero_upstream = expanded_upstream == 0.0
        prediction_result = numpy.where(zero_upstream, 0.0, prediction_result)
        target_result = numpy.where(zero_upstream, 0.0, target_result)
    prediction_storage = _storage(
        prediction_result,
        dtype=grad.dtype,
        output_shape=prediction.shape,
        numpy=numpy,
    )
    target_storage = _storage(
        target_result,
        dtype=grad.dtype,
        output_shape=target.shape,
        numpy=numpy,
    )
    if prediction_storage is None or target_storage is None:
        return None
    return prediction_storage, target_storage


def slice_tensor(
    value: Tensor,
    key: TensorIndex,
    *,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Run a NumPy slicing kernel after caller-side key validation."""
    numpy = _numpy()
    try:
        # Provider slicing can return a view into the input Tensor. Backend
        # results transferred into a new Tensor must own independent storage.
        result = _view(value, numpy)[key].copy()
    except ValueError:
        return None
    return _storage(
        result,
        dtype=value.dtype,
        output_shape=output_shape,
        numpy=numpy,
    )


def slice_scatter(
    value: Tensor,
    indices: list[int],
    *,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Scatter flat values into a zero NumPy tensor."""
    if _cuda_integer(value.dtype):
        return None
    numpy = _numpy()
    working_dtype = object if value.dtype.kind == "integer" else numpy.dtype(
        value.dtype.name
    )
    result = numpy.zeros(_shape_size(output_shape), dtype=working_dtype)
    try:
        values = _view(value, numpy).reshape(-1).astype(
            working_dtype,
            copy=False,
        )
    except ValueError:
        return None
    numpy.add.at(result, indices, values)
    return _storage(
        result,
        dtype=value.dtype,
        output_shape=output_shape,
        numpy=numpy,
    )


def cast_tensor(value: Tensor, *, dtype: DataType) -> Storage | None:
    """Convert tensor values with Python-compatible scalar conversion."""
    if _cuda_integer(dtype):
        return None
    numpy = _numpy()
    try:
        source = _view(value, numpy).reshape(-1)
    except ValueError:
        return None
    if dtype.kind == "integer":
        converter = numpy.frompyfunc(int, 1, 1)
        result = converter(source)
    else:
        # Same-dtype casts would otherwise retain the source buffer.
        result = source.astype(numpy.float64, copy=True)
    return _storage(
        result,
        dtype=dtype,
        output_shape=value.shape,
        numpy=numpy,
    )


def full(
    shape: tuple[int, ...],
    fill_value: int | float,
    *,
    dtype: DataType,
) -> Storage | None:
    """Create constant-filled canonical storage."""
    if _cuda_integer(dtype):
        return None
    numpy = _numpy()
    working_dtype = object if dtype.kind == "integer" else numpy.float64
    result = numpy.full(shape, fill_value, dtype=working_dtype)
    return _storage(
        result,
        dtype=dtype,
        output_shape=shape,
        numpy=numpy,
    )


def eye(
    rows: int,
    columns: int,
    k: int,
    *,
    dtype: DataType,
) -> Storage | None:
    """Create identity-like canonical storage."""
    if _cuda_integer(dtype):
        return None
    numpy = _numpy()
    working_dtype = object if dtype.kind == "integer" else numpy.float64
    result = numpy.eye(rows, columns, k=k, dtype=working_dtype)
    return _storage(
        result,
        dtype=dtype,
        output_shape=(rows, columns),
        numpy=numpy,
    )


def arange(
    start: int | float,
    step: int | float,
    count: int,
    *,
    dtype: DataType,
) -> Storage | None:
    """Create an arithmetic progression from a validated element count."""
    integer_inputs = isinstance(start, int) and isinstance(step, int)
    if _cuda_integer(dtype) or (_is_cuda() and integer_inputs):
        return None
    numpy = _numpy()
    working_dtype = object if integer_inputs else numpy.float64
    indices = numpy.arange(count, dtype=working_dtype)
    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        result = start + indices * step
    return _storage(
        result,
        dtype=dtype,
        output_shape=(count,),
        numpy=numpy,
    )


def _is_cuda() -> bool:
    from . import get_backend

    return get_backend() == "cuda"


def linspace(
    start: int | float,
    stop: int | float,
    count: int,
    *,
    dtype: DataType,
) -> Storage | None:
    """Create evenly spaced values when ordinary vector arithmetic is safe."""
    if count < 2:
        return None
    start_value = float(start)
    stop_value = float(stop)
    if start_value * stop_value < 0.0:
        return None
    numpy = _numpy()
    limit = numpy.finfo(numpy.float64).max / 4.0
    if abs(start_value) > limit or abs(stop_value) > limit:
        return None
    fractions = numpy.arange(count, dtype=numpy.float64) / (count - 1)
    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        result = start_value * (1.0 - fractions) + stop_value * fractions
    result[0] = start
    result[-1] = stop
    return _storage(
        result,
        dtype=dtype,
        output_shape=(count,),
        numpy=numpy,
    )


def transpose(
    value: Tensor,
    permutation: tuple[int, ...],
    *,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Permute tensor axes into canonical contiguous storage."""
    numpy = _numpy()
    # Transpose is a provider view operation, while the public Tensor result is
    # materialized and independently owned.
    result = numpy.transpose(
        _view(value, numpy),
        axes=permutation,
    ).copy()
    return _storage(
        result,
        dtype=value.dtype,
        output_shape=output_shape,
        numpy=numpy,
    )


def concat(
    values: Sequence[Tensor],
    axis: int,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Concatenate tensors along an existing axis."""
    numpy = _numpy()
    try:
        result = numpy.concatenate(
            [_view(value, numpy) for value in values],
            axis=axis,
        )
    except (TypeError, ValueError):
        return None
    return _storage(
        result,
        dtype=dtype,
        output_shape=output_shape,
        numpy=numpy,
    )


def stack(
    values: Sequence[Tensor],
    axis: int,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Stack tensors along a newly inserted axis."""
    numpy = _numpy()
    try:
        result = numpy.stack(
            [_view(value, numpy) for value in values],
            axis=axis,
        )
    except (TypeError, ValueError):
        return None
    return _storage(
        result,
        dtype=dtype,
        output_shape=output_shape,
        numpy=numpy,
    )


def outer(
    left: Tensor,
    right: Tensor,
    *,
    dtype: DataType,
) -> Storage | None:
    """Run a vector outer product."""
    numpy = _numpy()
    try:
        left_values = _operand(left, dtype, numpy)
        right_values = _operand(right, dtype, numpy)
    except (TypeError, ValueError):
        return None
    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        result = numpy.multiply.outer(left_values, right_values)
    return _storage(
        result,
        dtype=dtype,
        output_shape=(left.size, right.size),
        numpy=numpy,
    )


def outer_gradient(
    grad: Tensor,
    left: Tensor,
    right: Tensor,
) -> tuple[Storage, Storage] | None:
    """Run stable native outer-product VJPs when products can be summed."""
    numpy = _numpy()
    upstream = _view(grad, numpy).astype(numpy.float64, copy=False)
    left_values = _view(left, numpy).astype(numpy.float64, copy=False)
    right_values = _view(right, numpy).astype(numpy.float64, copy=False)
    if not _finite_operands(
        upstream,
        left_values,
        right_values,
        numpy=numpy,
    ):
        return None
    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        left_terms = upstream * right_values
        right_terms = upstream * left_values[:, None]
    if not _stable_sum_candidate(left_terms, (1,), numpy) or not (
        _stable_sum_candidate(right_terms, (0,), numpy)
    ):
        return None
    left_result = numpy.sum(left_terms, axis=1)
    right_result = numpy.sum(right_terms, axis=0)
    left_storage = _storage(
        left_result,
        dtype=grad.dtype,
        output_shape=left.shape,
        numpy=numpy,
    )
    right_storage = _storage(
        right_result,
        dtype=grad.dtype,
        output_shape=right.shape,
        numpy=numpy,
    )
    if left_storage is None or right_storage is None:
        return None
    return left_storage, right_storage


def sgd_update(
    parameter: Tensor,
    gradient: Tensor,
    learning_rate: float,
) -> Storage | None:
    """Apply one fused SGD update."""
    numpy = _numpy()
    values = _view(parameter, numpy).astype(numpy.float64, copy=False)
    gradients = _view(gradient, numpy).astype(numpy.float64, copy=False)
    if not _finite_operands(values, gradients, numpy=numpy):
        return None
    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        result = values - learning_rate * gradients
    if not bool(numpy.all(numpy.isfinite(result))):
        return None
    return _storage(
        result,
        dtype=parameter.dtype,
        output_shape=parameter.shape,
        numpy=numpy,
    )


def adam_update(
    parameter: Tensor,
    gradient: Tensor,
    moment: Tensor,
    scale: Tensor,
    scaled: Tensor,
    *,
    beta1: float,
    beta2: float,
    learning_rate: float,
    epsilon: float,
    first_correction: float,
    second_correction: float,
) -> tuple[Storage, Storage, Storage, Storage, Storage] | None:
    """Apply one fused Adam update on finite, non-cancelling state."""
    numpy = _numpy()
    tensors = (parameter, gradient, moment, scale, scaled)
    values = [
        _view(item, numpy).astype(numpy.float64, copy=False)
        for item in tensors
    ]
    parameter_values, gradients, moments, scales, scaled_values = values
    if not _finite_operands(*values, numpy=numpy):
        return None
    left_term = beta1 * moments
    right_term = (1.0 - beta1) * gradients
    if bool(numpy.any((left_term > 0.0) & (right_term < 0.0))) or bool(
        numpy.any((left_term < 0.0) & (right_term > 0.0))
    ):
        return None
    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        new_moments = left_term + right_term
        new_scales = numpy.maximum(scales, numpy.abs(gradients))
        safe_scales = numpy.where(new_scales == 0.0, 1.0, new_scales)
        previous_ratio = scales / safe_scales
        gradient_ratio = numpy.abs(gradients) / safe_scales
        new_scaled = (
            beta2 * scaled_values * previous_ratio * previous_ratio
            + (1.0 - beta2) * gradient_ratio * gradient_ratio
        )
        new_scaled = numpy.where(new_scales == 0.0, 0.0, new_scaled)
        root_correction = numpy.sqrt(second_correction)
        root_moment = new_scales * numpy.sqrt(new_scaled)
        denominator = first_correction * (
            root_moment + epsilon * root_correction
        )
        ratio = new_moments * root_correction / denominator
        parameter_result = parameter_values - learning_rate * ratio
        visible = new_scales * new_scales * new_scaled
    if not _finite_operands(
        new_moments,
        new_scales,
        new_scaled,
        parameter_result,
        numpy=numpy,
    ):
        return None
    specifications = (
        (parameter_result, parameter.dtype),
        (new_moments, gradient.dtype),
        (visible, gradient.dtype),
        (new_scales, gradient.dtype),
        (new_scaled, gradient.dtype),
    )
    storages = tuple(
        _storage(
            result,
            dtype=dtype,
            output_shape=gradient.shape,
            numpy=numpy,
        )
        for result, dtype in specifications
    )
    if any(storage is None for storage in storages):
        return None
    return cast(
        "tuple[Storage, Storage, Storage, Storage, Storage]",
        storages,
    )


def rmsprop_update(
    parameter: Tensor,
    gradient: Tensor,
    scale: Tensor,
    scaled: Tensor,
    *,
    rho: float,
    learning_rate: float,
    epsilon: float,
) -> tuple[Storage, Storage, Storage, Storage] | None:
    """Apply one fused RMSprop update on finite optimizer state."""
    numpy = _numpy()
    tensors = (parameter, gradient, scale, scaled)
    values = [
        _view(item, numpy).astype(numpy.float64, copy=False)
        for item in tensors
    ]
    parameter_values, gradients, scales, scaled_values = values
    if not _finite_operands(*values, numpy=numpy):
        return None
    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        new_scales = numpy.maximum(scales, numpy.abs(gradients))
        safe_scales = numpy.where(new_scales == 0.0, 1.0, new_scales)
        previous_ratio = scales / safe_scales
        gradient_ratio = numpy.abs(gradients) / safe_scales
        new_scaled = (
            rho * scaled_values * previous_ratio * previous_ratio
            + (1.0 - rho) * gradient_ratio * gradient_ratio
        )
        new_scaled = numpy.where(new_scales == 0.0, 0.0, new_scaled)
        root_moment = new_scales * numpy.sqrt(new_scaled)
        parameter_result = parameter_values - (
            learning_rate * gradients / (root_moment + epsilon)
        )
        visible = new_scales * new_scales * new_scaled
    if not _finite_operands(
        new_scales,
        new_scaled,
        parameter_result,
        numpy=numpy,
    ):
        return None
    specifications = (
        (parameter_result, parameter.dtype),
        (visible, gradient.dtype),
        (new_scales, gradient.dtype),
        (new_scaled, gradient.dtype),
    )
    storages = tuple(
        _storage(
            result,
            dtype=dtype,
            output_shape=gradient.shape,
            numpy=numpy,
        )
        for result, dtype in specifications
    )
    if any(storage is None for storage in storages):
        return None
    return cast(
        "tuple[Storage, Storage, Storage, Storage]",
        storages,
    )


def sgd_updates(
    parameters: Sequence[Tensor],
    gradients: Sequence[Tensor],
    learning_rate: float,
) -> tuple[Storage, ...] | None:
    """Apply one native SGD update to several compatible parameters."""
    if not _optimizer_batch_compatible(parameters, gradients):
        return None
    numpy = _numpy()
    values = _optimizer_batch_values(parameters, numpy)
    gradient_values = _optimizer_batch_values(gradients, numpy)
    if values is None or gradient_values is None:
        return None
    if _array_kind(numpy) == "cuda":
        result, invalid = _cuda_sgd_batch_kernel()(
            values,
            gradient_values,
            learning_rate,
        )
        if bool(numpy.any(invalid)):
            return None
    else:
        with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
            result = values - learning_rate * gradient_values
        valid = (
            numpy.all(numpy.isfinite(values))
            & numpy.all(numpy.isfinite(gradient_values))
            & numpy.all(numpy.isfinite(result))
        )
        if not bool(valid):
            return None
    return _split_optimizer_storage(result, parameters, numpy)


@lru_cache(maxsize=1)
def _cuda_sgd_batch_kernel() -> Any:
    """Return a fused CUDA SGD update and validation kernel."""
    cupy = importlib.import_module("cupy")
    return cupy.ElementwiseKernel(
        "float64 parameter, float64 gradient, float64 learning_rate",
        "float64 updated, uint8 invalid",
        """
        updated = parameter - learning_rate * gradient;
        invalid = (
            !isfinite(parameter)
            || !isfinite(gradient)
            || !isfinite(updated)
        );
        """,
        "tensors_sgd_batch",
    )


@lru_cache(maxsize=1)
def _cuda_adam_batch_kernel() -> Any:
    """Return a fused CUDA Adam update and validation kernel."""
    cupy = importlib.import_module("cupy")
    return cupy.ElementwiseKernel(
        """
        float64 parameter, float64 gradient, float64 moment,
        float64 scale, float64 normalized, float64 beta1, float64 beta2,
        float64 learning_rate, float64 epsilon,
        float64 first_correction, float64 second_correction
        """,
        """
        float64 updated, float64 new_moment, float64 visible,
        float64 new_scale, float64 new_normalized, uint8 invalid
        """,
        """
        double left = beta1 * moment;
        double right = (1.0 - beta1) * gradient;
        new_moment = left + right;
        new_scale = fmax(scale, fabs(gradient));
        double safe_scale = new_scale == 0.0 ? 1.0 : new_scale;
        double previous_ratio = scale / safe_scale;
        double gradient_ratio = fabs(gradient) / safe_scale;
        new_normalized = (
            beta2 * normalized * previous_ratio * previous_ratio
            + (1.0 - beta2) * gradient_ratio * gradient_ratio
        );
        if (new_scale == 0.0) new_normalized = 0.0;
        double root_correction = sqrt(second_correction);
        double root_moment = new_scale * sqrt(new_normalized);
        double denominator = first_correction * (
            root_moment + epsilon * root_correction
        );
        double ratio = new_moment * root_correction / denominator;
        updated = parameter - learning_rate * ratio;
        visible = new_scale * new_scale * new_normalized;
        invalid = (
            !isfinite(parameter)
            || !isfinite(gradient)
            || !isfinite(moment)
            || !isfinite(scale)
            || !isfinite(normalized)
            || (left > 0.0 && right < 0.0)
            || (left < 0.0 && right > 0.0)
            || !isfinite(updated)
            || !isfinite(new_moment)
            || !isfinite(new_scale)
            || !isfinite(new_normalized)
        );
        """,
        "tensors_adam_batch",
    )


@lru_cache(maxsize=1)
def _cuda_rmsprop_batch_kernel() -> Any:
    """Return a fused CUDA RMSprop update and validation kernel."""
    cupy = importlib.import_module("cupy")
    return cupy.ElementwiseKernel(
        """
        float64 parameter, float64 gradient, float64 scale,
        float64 normalized, float64 rho, float64 learning_rate,
        float64 epsilon
        """,
        """
        float64 updated, float64 visible, float64 new_scale,
        float64 new_normalized, uint8 invalid
        """,
        """
        new_scale = fmax(scale, fabs(gradient));
        double safe_scale = new_scale == 0.0 ? 1.0 : new_scale;
        double previous_ratio = scale / safe_scale;
        double gradient_ratio = fabs(gradient) / safe_scale;
        new_normalized = (
            rho * normalized * previous_ratio * previous_ratio
            + (1.0 - rho) * gradient_ratio * gradient_ratio
        );
        if (new_scale == 0.0) new_normalized = 0.0;
        double root_moment = new_scale * sqrt(new_normalized);
        updated = parameter - (
            learning_rate * gradient / (root_moment + epsilon)
        );
        visible = new_scale * new_scale * new_normalized;
        invalid = (
            !isfinite(parameter)
            || !isfinite(gradient)
            || !isfinite(scale)
            || !isfinite(normalized)
            || !isfinite(updated)
            || !isfinite(new_scale)
            || !isfinite(new_normalized)
        );
        """,
        "tensors_rmsprop_batch",
    )


def _optimizer_scalar_batch(
    values: Sequence[float],
    references: Sequence[Tensor],
    numpy: Any,
) -> Any:
    """Expand one scalar per parameter into its batched element layout."""
    if values and all(value == values[0] for value in values[1:]):
        return float(values[0])
    scalars = numpy.asarray(tuple(values), dtype=numpy.float64)
    counts = numpy.asarray(
        tuple(reference.size for reference in references),
        dtype=numpy.int64,
    )
    return numpy.repeat(scalars, counts)


def adam_updates(
    parameters: Sequence[Tensor],
    gradients: Sequence[Tensor],
    moments: Sequence[Tensor],
    scales: Sequence[Tensor],
    scaled_values: Sequence[Tensor],
    *,
    beta1: float,
    beta2: float,
    learning_rate: float,
    epsilon: float,
    first_corrections: Sequence[float],
    second_corrections: Sequence[float],
) -> tuple[tuple[Storage, ...], ...] | None:
    """Apply Adam to several parameters with one group of array operations."""
    if not _optimizer_batch_compatible(
        parameters,
        gradients,
        moments,
        scales,
        scaled_values,
    ):
        return None
    numpy = _numpy()
    groups = (parameters, gradients, moments, scales, scaled_values)
    optional_arrays = tuple(
        _optimizer_batch_values(group, numpy) for group in groups
    )
    if any(array is None for array in optional_arrays):
        return None
    arrays = cast("tuple[Any, ...]", optional_arrays)
    (
        parameter_values,
        gradient_values,
        moment_values,
        scale_values,
        normalized_values,
    ) = arrays
    first = _optimizer_scalar_batch(first_corrections, parameters, numpy)
    second = _optimizer_scalar_batch(second_corrections, parameters, numpy)
    if _array_kind(numpy) == "cuda":
        (
            parameter_result,
            new_moments,
            visible,
            new_scales,
            new_scaled,
            invalid,
        ) = _cuda_adam_batch_kernel()(
            parameter_values,
            gradient_values,
            moment_values,
            scale_values,
            normalized_values,
            beta1,
            beta2,
            learning_rate,
            epsilon,
            first,
            second,
        )
        if bool(numpy.any(invalid)):
            return None
    else:
        left_term = beta1 * moment_values
        right_term = (1.0 - beta1) * gradient_values
        with _errstate(
            numpy,
            divide="ignore",
            over="ignore",
            under="ignore",
            invalid="ignore",
        ):
            new_moments = left_term + right_term
            new_scales = numpy.maximum(
                scale_values,
                numpy.abs(gradient_values),
            )
            safe_scales = numpy.where(new_scales == 0.0, 1.0, new_scales)
            previous_ratio = scale_values / safe_scales
            gradient_ratio = numpy.abs(gradient_values) / safe_scales
            new_scaled = (
                beta2 * normalized_values * previous_ratio * previous_ratio
                + (1.0 - beta2) * gradient_ratio * gradient_ratio
            )
            new_scaled = numpy.where(new_scales == 0.0, 0.0, new_scaled)
            root_correction = numpy.sqrt(second)
            root_moment = new_scales * numpy.sqrt(new_scaled)
            denominator = first * (
                root_moment + epsilon * root_correction
            )
            ratio = new_moments * root_correction / denominator
            parameter_result = parameter_values - learning_rate * ratio
            visible = new_scales * new_scales * new_scaled
        finite_inputs = numpy.asarray(True)
        for array in arrays:
            finite_inputs &= numpy.all(numpy.isfinite(array))
        valid = (
            finite_inputs
            & ~numpy.any(
                ((left_term > 0.0) & (right_term < 0.0))
                | ((left_term < 0.0) & (right_term > 0.0))
            )
            & numpy.all(numpy.isfinite(new_moments))
            & numpy.all(numpy.isfinite(new_scales))
            & numpy.all(numpy.isfinite(new_scaled))
            & numpy.all(numpy.isfinite(parameter_result))
        )
        if not bool(valid):
            return None
    results = (
        _split_optimizer_storage(parameter_result, parameters, numpy),
        _split_optimizer_storage(new_moments, gradients, numpy),
        _split_optimizer_storage(visible, gradients, numpy),
        _split_optimizer_storage(new_scales, gradients, numpy),
        _split_optimizer_storage(new_scaled, gradients, numpy),
    )
    if any(result is None for result in results):
        return None
    return cast("tuple[tuple[Storage, ...], ...]", results)


def rmsprop_updates(
    parameters: Sequence[Tensor],
    gradients: Sequence[Tensor],
    scales: Sequence[Tensor],
    scaled_values: Sequence[Tensor],
    *,
    rho: float,
    learning_rate: float,
    epsilon: float,
) -> tuple[tuple[Storage, ...], ...] | None:
    """Apply RMSprop to several parameters with one group of array operations."""
    if not _optimizer_batch_compatible(
        parameters,
        gradients,
        scales,
        scaled_values,
    ):
        return None
    numpy = _numpy()
    groups = (parameters, gradients, scales, scaled_values)
    optional_arrays = tuple(
        _optimizer_batch_values(group, numpy) for group in groups
    )
    if any(array is None for array in optional_arrays):
        return None
    arrays = cast("tuple[Any, ...]", optional_arrays)
    parameter_values, gradient_values, scale_values, normalized_values = arrays
    if _array_kind(numpy) == "cuda":
        (
            parameter_result,
            visible,
            new_scales,
            new_scaled,
            invalid,
        ) = _cuda_rmsprop_batch_kernel()(
            parameter_values,
            gradient_values,
            scale_values,
            normalized_values,
            rho,
            learning_rate,
            epsilon,
        )
        if bool(numpy.any(invalid)):
            return None
    else:
        with _errstate(
            numpy,
            divide="ignore",
            over="ignore",
            under="ignore",
            invalid="ignore",
        ):
            new_scales = numpy.maximum(
                scale_values,
                numpy.abs(gradient_values),
            )
            safe_scales = numpy.where(new_scales == 0.0, 1.0, new_scales)
            previous_ratio = scale_values / safe_scales
            gradient_ratio = numpy.abs(gradient_values) / safe_scales
            new_scaled = (
                rho * normalized_values * previous_ratio * previous_ratio
                + (1.0 - rho) * gradient_ratio * gradient_ratio
            )
            new_scaled = numpy.where(new_scales == 0.0, 0.0, new_scaled)
            root_moment = new_scales * numpy.sqrt(new_scaled)
            parameter_result = parameter_values - (
                learning_rate * gradient_values / (root_moment + epsilon)
            )
            visible = new_scales * new_scales * new_scaled
        finite_inputs = numpy.asarray(True)
        for array in arrays:
            finite_inputs &= numpy.all(numpy.isfinite(array))
        valid = (
            finite_inputs
            & numpy.all(numpy.isfinite(new_scales))
            & numpy.all(numpy.isfinite(new_scaled))
            & numpy.all(numpy.isfinite(parameter_result))
        )
        if not bool(valid):
            return None
    results = (
        _split_optimizer_storage(parameter_result, parameters, numpy),
        _split_optimizer_storage(visible, gradients, numpy),
        _split_optimizer_storage(new_scales, gradients, numpy),
        _split_optimizer_storage(new_scaled, gradients, numpy),
    )
    if any(result is None for result in results):
        return None
    return cast("tuple[tuple[Storage, ...], ...]", results)


def _summation_guard(
    values: Any,
    numpy: Any,
    *,
    axes: tuple[int, ...] | None = None,
    keepdims: bool = False,
    mixed_signs: bool = True,
) -> Any:
    """Build one provider-native safety decision for ordinary summation."""
    if values.size == 0:
        return numpy.asarray(True)
    minimum = numpy.min(values, axis=axes, keepdims=keepdims)
    maximum = numpy.max(values, axis=axes, keepdims=keepdims)
    subnormal = numpy.any(
        (values != 0.0)
        & (numpy.abs(values) < numpy.finfo(numpy.float64).tiny),
        axis=axes,
        keepdims=keepdims,
    )
    valid = (
        numpy.isfinite(minimum)
        & numpy.isfinite(maximum)
        & ~subnormal
    )
    if mixed_signs:
        absolute = numpy.abs(values)
        largest = numpy.max(absolute, axis=axes, keepdims=keepdims)
        smallest = numpy.min(
            numpy.where(absolute == 0.0, numpy.inf, absolute),
            axis=axes,
            keepdims=keepdims,
        )
        comparable_magnitudes = (largest == 0.0) | (
            smallest >= largest * numpy.finfo(numpy.float64).eps
        )
        mixed = (minimum < 0.0) & (maximum > 0.0)
        valid &= ~(mixed & ~comparable_magnitudes)
    return numpy.all(valid)


def _scaled_sum(values: Any, axes: tuple[int, ...], numpy: Any) -> Any:
    """Sum comparable finite values without overflowing intermediates."""
    if not axes:
        return values
    scale = numpy.max(numpy.abs(values), axis=axes, keepdims=True)
    safe_scale = numpy.where(scale == 0.0, 1.0, scale)
    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        normalized = values / safe_scale
        result = numpy.sum(normalized, axis=axes, keepdims=True) * scale
    return numpy.where(scale == 0.0, 0.0, result)


def _scaled_product_sum(
    left: Any,
    right: Any,
    axes: tuple[int, ...],
    numpy: Any,
) -> Any:
    """Evaluate a product reduction through normalized device operands."""
    reduction_axes: tuple[int, ...] | None = axes if axes else None
    left_scale = numpy.max(numpy.abs(left), axis=reduction_axes, keepdims=True)
    right_scale = numpy.max(numpy.abs(right), axis=reduction_axes, keepdims=True)
    safe_left = numpy.where(left_scale == 0.0, 1.0, left_scale)
    safe_right = numpy.where(right_scale == 0.0, 1.0, right_scale)
    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        normalized_terms = (left / safe_left) * (right / safe_right)
        normalized_sum = (
            numpy.sum(normalized_terms, axis=axes, keepdims=True)
            if axes
            else normalized_terms
        )
        log_magnitude = (
            numpy.log(numpy.abs(normalized_sum))
            + numpy.log(safe_left)
            + numpy.log(safe_right)
        )
        restored = numpy.copysign(numpy.exp(log_magnitude), normalized_sum)
    zero = (
        (left_scale == 0.0)
        | (right_scale == 0.0)
        | (normalized_sum == 0.0)
    )
    return numpy.where(zero, 0.0, restored)


def reduction(
    operation: ReductionOperation,
    value: Tensor,
    axes: tuple[int, ...],
    *,
    keepdims: bool,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Run a numerically guarded NumPy reduction."""
    if value.size == 0:
        return None

    numpy = _numpy()
    axis = axes
    if operation in {"min", "max"}:
        values = _view(value, numpy)
        function = numpy.min if operation == "min" else numpy.max
        result = function(values, axis=axis, keepdims=keepdims)
    elif operation == "prod":
        values = _view(value, numpy)
        working = (
            values.astype(object)
            if dtype.kind == "integer"
            else values.astype(numpy.float64, copy=False)
        )
        with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
            result = numpy.prod(working, axis=axis, keepdims=keepdims)
    else:
        values = _view(value, numpy).astype(numpy.float64, copy=False)

    if operation in {"sum", "mean"}:
        if operation == "sum" and dtype.kind == "integer":
            return None
        with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
            direct = numpy.sum(values, axis=axis, keepdims=True)
            if operation == "mean":
                count = math.prod(value.shape[item] for item in axis)
                direct = direct / count

        # A finite same-sign reduction cannot lose values through
        # cancellation, and a finite direct result proves that no partial
        # overflow escaped the provider.  This common NumPy case only needs
        # min/max plus the sum; mixed-sign or extreme data retains the full
        # scaled guard below.
        from . import get_backend

        ordinary = False
        if get_backend() == "numpy":
            minimum = numpy.min(values, axis=axis, keepdims=True)
            maximum = numpy.max(values, axis=axis, keepdims=True)
            ordinary = bool(numpy.all(
                numpy.isfinite(minimum)
                & numpy.isfinite(maximum)
                & numpy.isfinite(direct)
                & ((minimum >= 0.0) | (maximum <= 0.0))
            ))
        if ordinary:
            result = direct
        else:
            with _errstate(
                numpy,
                over="ignore",
                under="ignore",
                invalid="ignore",
            ):
                scaled = _scaled_sum(values, axis, numpy)
                if operation == "mean":
                    scaled = scaled / count
            safe = _summation_guard(
                values,
                numpy,
                axes=axis,
                keepdims=True,
            )
            direct_safe = safe & numpy.all(numpy.isfinite(direct))
            result = numpy.where(direct_safe, direct, scaled)
            valid = safe & ~numpy.any(numpy.isnan(result))
            if not bool(valid):
                return None
        if not keepdims and axis:
            result = numpy.squeeze(result, axis=axis)
    elif operation in {"variance", "std"}:
        with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
            center = numpy.mean(values, axis=axis, keepdims=True)
            centered = values - center
            scale = numpy.max(
                numpy.abs(centered),
                axis=axis,
                keepdims=True,
            )
            safe_scale = numpy.where(scale == 0.0, 1.0, scale)
            normalized = centered / safe_scale
            normalized_variance = numpy.mean(
                normalized * normalized,
                axis=axis,
                keepdims=keepdims,
            )
            output_scale = (
                scale if keepdims else numpy.squeeze(scale, axis=axis)
            )
            deviation = output_scale * numpy.sqrt(normalized_variance)
            result = (
                deviation * deviation
                if operation == "variance"
                else deviation
            )
        valid = (
            numpy.all(numpy.isfinite(values))
            & numpy.all(numpy.isfinite(centered))
            & numpy.all(numpy.isfinite(result))
        )
        if not bool(valid):
            return None
    elif operation == "norm":
        with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
            absolute = numpy.abs(values)
            scale = numpy.max(absolute, axis=axis, keepdims=True)
            safe_scale = numpy.where(scale == 0.0, 1.0, scale)
            normalized = values / safe_scale
            normalized_magnitude = numpy.sqrt(
                numpy.sum(
                    normalized * normalized,
                    axis=axis,
                    keepdims=keepdims,
                )
            )
            output_scale = (
                scale if keepdims else numpy.squeeze(scale, axis=axis)
            )
            result = output_scale * normalized_magnitude
        valid = numpy.all(numpy.isfinite(values)) & numpy.all(
            numpy.isfinite(result)
        )
        if not bool(valid):
            return None

    return _storage(
        result,
        dtype=dtype,
        output_shape=output_shape,
        numpy=numpy,
    )


def reduction_gradient(
    operation: DifferentiableReductionOperation,
    grad: Tensor,
    value: Tensor,
    axes: tuple[int, ...],
    *,
    keepdims: bool,
) -> Storage | None:
    """Run fused VJPs for reductions with regular native fast paths."""
    numpy = _numpy()
    values = _view(value, numpy).astype(numpy.float64, copy=False)
    upstream = _view(grad, numpy).astype(numpy.float64, copy=False)
    expanded_shape = tuple(
        1 if dimension in axes else size
        for dimension, size in enumerate(value.shape)
    )
    try:
        expanded = upstream.reshape(expanded_shape)
    except ValueError:
        return None
    count = 1
    for axis in axes:
        count *= value.shape[axis]

    with _errstate(

        numpy,
        divide="ignore",
        over="ignore",
        under="ignore",
        invalid="ignore",
    ):
        if operation == "sum":
            # `broadcast_to` returns a view of the upstream gradient.
            result = numpy.broadcast_to(expanded, value.shape).copy()
        elif operation == "mean":
            if count == 0:
                return None
            result = numpy.broadcast_to(expanded / count, value.shape)
        elif operation == "variance":
            if count == 0:
                return None
            center = numpy.mean(values, axis=axes, keepdims=True)
            centered = values - center
            scale = numpy.max(
                numpy.abs(centered),
                axis=axes,
                keepdims=True,
            )
            safe_scale = numpy.where(scale == 0.0, 1.0, scale)
            normalized = centered / safe_scale
            result = expanded * normalized * scale * (2.0 / count)
            valid = (
                numpy.all(numpy.isfinite(values))
                & numpy.all(numpy.isfinite(centered))
                & numpy.all(numpy.isfinite(result))
            )
            if not bool(valid):
                return None
        elif operation == "std":
            if not bool(numpy.all(numpy.isfinite(values))) or count == 0:
                return None
            scale = numpy.max(numpy.abs(values), axis=axes, keepdims=True)
            safe_scale = numpy.where(scale == 0.0, 1.0, scale)
            normalized = values / safe_scale
            center = numpy.mean(normalized, axis=axes, keepdims=True)
            centered = normalized - center
            deviation = numpy.sqrt(
                numpy.mean(centered * centered, axis=axes, keepdims=True)
            )
            derivative = numpy.where(
                deviation == 0.0,
                0.0,
                centered / (count * deviation),
            )
            result = expanded * derivative
        elif operation == "prod":
            if not bool(numpy.all(numpy.isfinite(values))):
                return None
            zero_count = numpy.sum(values == 0.0, axis=axes, keepdims=True)
            product = numpy.prod(values, axis=axes, keepdims=True)
            nonzero_product = numpy.prod(
                numpy.where(values == 0.0, 1.0, values),
                axis=axes,
                keepdims=True,
            )
            unsafe = (
                ((zero_count == 0) & ((product == 0.0) | ~numpy.isfinite(product)))
                | (
                    (zero_count == 1)
                    & (
                        (nonzero_product == 0.0)
                        | ~numpy.isfinite(nonzero_product)
                    )
                )
            )
            if bool(numpy.any(unsafe)):
                return None
            derivative = numpy.where(
                zero_count == 0,
                product / values,
                numpy.where(
                    (zero_count == 1) & (values == 0.0),
                    nonzero_product,
                    0.0,
                ),
            )
            result = expanded * derivative
        else:
            has_nan = numpy.any(numpy.isnan(values), axis=axes, keepdims=True)
            function = numpy.min if operation == "min" else numpy.max
            extreme = function(values, axis=axes, keepdims=True)
            selected = values == extreme
            ties = numpy.sum(selected, axis=axes, keepdims=True)
            result = numpy.where(
                has_nan,
                numpy.nan,
                expanded * selected / ties,
            )
    return _storage(
        result,
        dtype=grad.dtype,
        output_shape=value.shape,
        numpy=numpy,
    )


def arg_extremum(
    operation: ArgExtremumOperation,
    value: Tensor,
    axis: int | None,
    *,
    keepdims: bool,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Run a first-occurrence argmin or argmax reduction."""
    if value.size == 0:
        return None
    from ..dtype import int64

    numpy = _numpy()
    values = _view(value, numpy)
    function = numpy.argmin if operation == "argmin" else numpy.argmax
    result = function(values, axis=axis, keepdims=keepdims)
    return _storage(
        result,
        dtype=int64,
        output_shape=output_shape,
        numpy=numpy,
    )


def comparison(
    operation: ComparisonOperation,
    left: Tensor,
    right: Tensor,
    *,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Run a broadcasting elementwise comparison."""
    from ..dtype import uint8

    numpy = _numpy()
    functions = {
        "equal": numpy.equal,
        "not_equal": numpy.not_equal,
        "less": numpy.less,
        "less_equal": numpy.less_equal,
        "greater": numpy.greater,
        "greater_equal": numpy.greater_equal,
    }
    try:
        result = functions[operation](_view(left, numpy), _view(right, numpy))
    except (TypeError, ValueError):
        return None
    return _storage(
        result,
        dtype=uint8,
        output_shape=output_shape,
        numpy=numpy,
    )


def where(
    condition: Tensor,
    left: Tensor,
    right: Tensor,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Run broadcasting elementwise selection."""
    numpy = _numpy()
    try:
        result = numpy.where(
            _view(condition, numpy) != 0,
            _view(left, numpy),
            _view(right, numpy),
        )
    except (TypeError, ValueError):
        return None
    return _storage(
        result,
        dtype=dtype,
        output_shape=output_shape,
        numpy=numpy,
    )


def where_gradient(
    grad: Tensor,
    condition: Tensor,
) -> tuple[Storage, Storage] | None:
    """Split a selection VJP into its expanded left and right terms."""
    numpy = _numpy()
    try:
        selected = numpy.broadcast_to(_view(condition, numpy), grad.shape) != 0
    except ValueError:
        return None
    upstream = _view(grad, numpy).astype(numpy.float64, copy=False)
    left = _storage(
        numpy.where(selected, upstream, 0.0),
        dtype=grad.dtype,
        output_shape=grad.shape,
        numpy=numpy,
    )
    right = _storage(
        numpy.where(selected, 0.0, upstream),
        dtype=grad.dtype,
        output_shape=grad.shape,
        numpy=numpy,
    )
    if left is None or right is None:
        return None
    return left, right


def clip(
    value: Tensor,
    min_value: int | float | None,
    max_value: int | float | None,
    *,
    dtype: DataType,
) -> Storage | None:
    """Clip tensor values to optional scalar bounds."""
    numpy = _numpy()
    values = _view(value, numpy)
    result = numpy.clip(values, min_value, max_value)
    return _storage(
        result,
        dtype=dtype,
        output_shape=value.shape,
        numpy=numpy,
    )


def clip_gradient(
    grad: Tensor,
    value: Tensor,
    min_value: int | float | None,
    max_value: int | float | None,
) -> Storage | None:
    """Run the clipping VJP with zero boundary subgradients."""
    numpy = _numpy()
    values = _view(value, numpy).astype(numpy.float64, copy=False)
    upstream = _view(grad, numpy).astype(numpy.float64, copy=False)
    mask = numpy.ones(value.shape, dtype=bool)
    if min_value is not None:
        mask &= values > min_value
    if max_value is not None:
        mask &= values < max_value
    result = numpy.where(numpy.isnan(values), numpy.nan, numpy.where(mask, upstream, 0.0))
    return _storage(
        result,
        dtype=grad.dtype,
        output_shape=value.shape,
        numpy=numpy,
    )


def extremum(
    operation: ExtremumOperation,
    left: Tensor,
    right: Tensor,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Run a broadcasting elementwise minimum or maximum."""
    numpy = _numpy()
    function = numpy.minimum if operation == "minimum" else numpy.maximum
    try:
        result = function(_view(left, numpy), _view(right, numpy))
    except (TypeError, ValueError):
        return None
    return _storage(
        result,
        dtype=dtype,
        output_shape=output_shape,
        numpy=numpy,
    )


def extremum_gradient(
    operation: ExtremumOperation,
    grad: Tensor,
    left: Tensor,
    right: Tensor,
) -> tuple[Storage, Storage] | None:
    """Split an elementwise-extremum VJP, sharing exact ties."""
    numpy = _numpy()
    try:
        left_values, right_values = numpy.broadcast_arrays(
            _view(left, numpy),
            _view(right, numpy),
        )
    except ValueError:
        return None
    upstream = _view(grad, numpy).astype(numpy.float64, copy=False)
    has_nan = numpy.isnan(left_values) | numpy.isnan(right_values)
    ties = left_values == right_values
    left_selected = (
        left_values > right_values
        if operation == "maximum"
        else left_values < right_values
    )
    left_weight = numpy.where(
        has_nan,
        numpy.nan,
        numpy.where(ties, 0.5, numpy.where(left_selected, 1.0, 0.0)),
    )
    right_weight = numpy.where(
        has_nan,
        numpy.nan,
        numpy.where(ties, 0.5, numpy.where(left_selected, 0.0, 1.0)),
    )
    left_storage = _storage(
        upstream * left_weight,
        dtype=grad.dtype,
        output_shape=grad.shape,
        numpy=numpy,
    )
    right_storage = _storage(
        upstream * right_weight,
        dtype=grad.dtype,
        output_shape=grad.shape,
        numpy=numpy,
    )
    if left_storage is None or right_storage is None:
        return None
    return left_storage, right_storage


def _sum_axes(
    source_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    """Return padded target shape and axes reduced after broadcasting."""
    if len(target_shape) > len(source_shape):
        return None
    padded = (1,) * (len(source_shape) - len(target_shape)) + target_shape
    axes = []
    for axis, (source, target) in enumerate(zip(source_shape, padded)):
        if source == target:
            continue
        if target != 1:
            return None
        axes.append(axis)
    return padded, tuple(axes)


def _stable_sum_candidate(values: Any, axes: tuple[int, ...], numpy: Any) -> Any:
    """Return a provider-native guard for ordinary reduction summation."""
    if axes:
        return _summation_guard(
            values,
            numpy,
            axes=axes,
            keepdims=True,
        )
    return _summation_guard(values, numpy, mixed_signs=False)


def sum_to_shape(
    gradient: Tensor,
    shape: tuple[int, ...],
) -> Storage | None:
    """Reduce a broadcast gradient using guarded native summation."""
    if gradient.dtype.kind == "integer":
        return None
    layout = _sum_axes(gradient.shape, shape)
    if layout is None:
        return None
    _, axes = layout
    numpy = _numpy()
    values = _view(gradient, numpy).astype(numpy.float64, copy=False)
    safe = _stable_sum_candidate(values, axes, numpy)
    result = _scaled_sum(values, axes, numpy)
    valid = safe & ~numpy.any(numpy.isnan(result))
    if not bool(valid):
        return None
    return _storage(
        numpy.asarray(result).reshape(shape),
        dtype=gradient.dtype,
        output_shape=shape,
        numpy=numpy,
    )


def sum_products_to_shape(
    gradient: Tensor,
    factor: Tensor,
    shape: tuple[int, ...],
) -> Storage | None:
    """Multiply and reduce broadcast VJP terms in one guarded kernel."""
    if gradient.dtype.kind == "integer":
        return None
    numpy = _numpy()
    try:
        left, right = numpy.broadcast_arrays(
            _view(gradient, numpy).astype(numpy.float64, copy=False),
            _view(factor, numpy).astype(numpy.float64, copy=False),
        )
    except ValueError:
        return None
    layout = _sum_axes(tuple(left.shape), shape)
    if layout is None:
        return None
    _, axes = layout
    finite = numpy.all(numpy.isfinite(left)) & numpy.all(numpy.isfinite(right))
    nonzero = (left != 0.0) & (right != 0.0)
    reduction_axes: tuple[int, ...] | None = axes if axes else None
    with _errstate(numpy, divide="ignore", invalid="ignore"):
        log_terms = numpy.log(numpy.abs(left)) + numpy.log(numpy.abs(right))
    largest_log = numpy.max(
        numpy.where(nonzero, log_terms, -numpy.inf),
        axis=reduction_axes,
        keepdims=True,
    )
    smallest_log = numpy.min(
        numpy.where(nonzero, log_terms, numpy.inf),
        axis=reduction_axes,
        keepdims=True,
    )
    any_nonzero = numpy.any(nonzero, axis=reduction_axes, keepdims=True)
    comparable = (~any_nonzero) | (
        largest_log - smallest_log
        <= -math.log(numpy.finfo(numpy.float64).eps)
    )
    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        products = left * right
        direct = (
            numpy.sum(products, axis=axes, keepdims=True)
            if axes
            else products
        )
    lost_range = (~numpy.isfinite(products)) | (
        (products == 0.0) & nonzero
    )
    direct_safe = (
        ~numpy.any(lost_range)
        & _stable_sum_candidate(products, axes, numpy)
        & ~numpy.any(numpy.isnan(direct))
    )
    stable = _scaled_product_sum(left, right, axes, numpy)
    result = numpy.where(direct_safe, direct, stable)
    valid = (
        finite
        & (direct_safe | numpy.all(comparable))
        & ~numpy.any(numpy.isnan(result))
    )
    if not bool(valid):
        return None
    return _storage(
        numpy.asarray(result).reshape(shape),
        dtype=gradient.dtype,
        output_shape=shape,
        numpy=numpy,
    )


def division_denominator_gradient(
    grad: Tensor,
    numerator: Tensor,
    denominator: Tensor,
) -> Storage | None:
    """Calculate ``-grad * numerator / denominator**2`` when range-safe."""
    numpy = _numpy()
    try:
        upstream = _operand(grad, grad.dtype, numpy)
        values = _operand(numerator, grad.dtype, numpy)
        divisors = _operand(denominator, grad.dtype, numpy)
    except (OverflowError, TypeError, ValueError):
        return None
    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        squares = numpy.square(divisors)
    finite_inputs = _finite_operands(
        upstream,
        values,
        divisors,
        numpy=numpy,
    )
    if not finite_inputs or bool(numpy.any(divisors == 0.0)):
        return None
    with _errstate(
        numpy,
        divide="ignore",
        over="ignore",
        under="ignore",
        invalid="ignore",
    ):
        direct = -upstream * values / squares
        zero = (upstream == 0.0) | (values == 0.0)
        log_magnitude = (
            numpy.log(numpy.abs(upstream))
            + numpy.log(numpy.abs(values))
            - 2.0 * numpy.log(numpy.abs(divisors))
        )
        sign = numpy.where(
            numpy.signbit(upstream) ^ numpy.signbit(values),
            1.0,
            -1.0,
        )
        stable = numpy.where(zero, 0.0, sign * numpy.exp(log_magnitude))
    unsafe = (
        (squares == 0.0)
        | ~numpy.isfinite(squares)
        | (~zero & ((direct == 0.0) | ~numpy.isfinite(direct)))
    )
    result = numpy.where(unsafe, stable, direct)
    return _storage(
        result,
        dtype=grad.dtype,
        output_shape=grad.shape,
        numpy=numpy,
    )


def power_base_gradient(
    grad: Tensor,
    base: Tensor,
    exponent: Tensor,
) -> Storage | None:
    """Calculate the power gradient with respect to its base when safe."""
    numpy = _numpy()
    try:
        upstream = _operand(grad, grad.dtype, numpy)
        bases = _operand(base, grad.dtype, numpy)
        powers = _operand(exponent, grad.dtype, numpy)
    except (OverflowError, TypeError, ValueError):
        return None
    if not _finite_operands(upstream, bases, powers, numpy=numpy):
        return None
    if bool(numpy.any(bases <= 0.0)):
        return None
    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        power_term = numpy.power(bases, powers - 1.0)
        direct = upstream * powers * power_term
        zero = (upstream == 0.0) | (powers == 0.0)
        log_magnitude = (
            numpy.log(numpy.abs(upstream))
            + numpy.log(numpy.abs(powers))
            + (powers - 1.0) * numpy.log(bases)
        )
        stable = numpy.where(
            zero,
            0.0,
            numpy.copysign(numpy.exp(log_magnitude), upstream * powers),
        )
    unsafe = ~zero & (
        (power_term == 0.0)
        | ~numpy.isfinite(power_term)
        | (direct == 0.0)
        | ~numpy.isfinite(direct)
    )
    result = numpy.where(unsafe, stable, direct)
    return _storage(
        result,
        dtype=grad.dtype,
        output_shape=grad.shape,
        numpy=numpy,
    )


def power_exponent_gradient(
    grad: Tensor,
    base: Tensor,
    exponent: Tensor,
) -> Storage | None:
    """Calculate the power gradient with respect to its exponent when safe."""
    numpy = _numpy()
    try:
        upstream = _operand(grad, grad.dtype, numpy)
        bases = _operand(base, grad.dtype, numpy)
        powers = _operand(exponent, grad.dtype, numpy)
    except (OverflowError, TypeError, ValueError):
        return None
    if not _finite_operands(upstream, bases, powers, numpy=numpy):
        return None
    if bool(numpy.any(bases <= 0.0)):
        return None
    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        logarithm = numpy.log(bases)
        outputs = numpy.power(bases, powers)
        direct = upstream * outputs * logarithm
        zero = (upstream == 0.0) | (logarithm == 0.0)
        log_magnitude = (
            numpy.log(numpy.abs(upstream))
            + powers * logarithm
            + numpy.log(numpy.abs(logarithm))
        )
        stable = numpy.where(
            zero,
            0.0,
            numpy.copysign(numpy.exp(log_magnitude), upstream * logarithm),
        )
    unsafe = ~zero & (
        (outputs == 0.0)
        | ~numpy.isfinite(outputs)
        | (direct == 0.0)
        | ~numpy.isfinite(direct)
    )
    result = numpy.where(unsafe, stable, direct)
    return _storage(
        result,
        dtype=grad.dtype,
        output_shape=grad.shape,
        numpy=numpy,
    )


def _finite_operands(*operands: Any, numpy: Any) -> bool:
    finite = numpy.asarray(True)
    for operand in operands:
        finite = finite & numpy.all(numpy.isfinite(operand))
    return bool(finite)


def _unsafe_finite_result(
    result: Any,
    *operands: Any,
    numpy: Any,
) -> bool:
    return _finite_operands(*operands, numpy=numpy) and not bool(
        numpy.all(numpy.isfinite(result))
    )


@lru_cache(maxsize=2)
def _import_array_module(module_name: str) -> Any:
    """Import and retain one array provider module."""
    try:
        return importlib.import_module(module_name)
    except ImportError as error:
        from . import BackendUnavailableError

        raise BackendUnavailableError(
            f"The {module_name} backend became unavailable after it was selected."
        ) from error


def _numpy() -> Any:
    """Return the cached array module selected by the active backend."""
    from . import get_backend

    backend = get_backend()
    return _import_array_module("cupy" if backend == "cuda" else "numpy")


def _comparable_finite_values(values: Any, numpy: Any) -> bool:
    """Return whether global scaling can retain every nonzero magnitude."""
    if not bool(numpy.all(numpy.isfinite(values))):
        return False
    absolute = numpy.abs(values)
    largest = numpy.max(absolute)
    smallest = numpy.min(numpy.where(absolute == 0.0, numpy.inf, absolute))
    return bool(
        (largest == 0.0)
        | (smallest >= largest * numpy.finfo(numpy.float64).eps)
    )


def _scaled_matmul(left: Any, right: Any, numpy: Any) -> Any:
    """Calculate a matrix product without overflowing temporary products."""
    left_scale = numpy.max(numpy.abs(left))
    right_scale = numpy.max(numpy.abs(right))
    safe_left = numpy.where(left_scale == 0.0, 1.0, left_scale)
    safe_right = numpy.where(right_scale == 0.0, 1.0, right_scale)
    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        normalized = numpy.matmul(left / safe_left, right / safe_right)
        log_magnitude = (
            numpy.log(numpy.abs(normalized))
            + numpy.log(safe_left)
            + numpy.log(safe_right)
        )
        restored = numpy.copysign(numpy.exp(log_magnitude), normalized)
    zero = (
        (left_scale == 0.0)
        | (right_scale == 0.0)
        | (normalized == 0.0)
    )
    return numpy.where(zero, 0.0, restored)


def matmul(
    left: Tensor,
    right: Tensor,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Return a NumPy matrix product or defer to the reference implementation."""
    if dtype.kind != "floating":
        return None

    numpy = _numpy()

    # The reference implementation accumulates float32 products in Python's
    # double precision and casts only the final result. Match that behavior by
    # using float64 as the NumPy working dtype for every supported float result.
    try:
        left_array = _view(left, numpy).astype(numpy.float64, copy=False)
        right_array = _view(right, numpy).astype(numpy.float64, copy=False)
    except ValueError:
        return None
    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        result = numpy.matmul(left_array, right_array)

    # Recover device-side when comparable finite operands overflow temporary
    # products. Highly disparate magnitudes still use the exact reference path.
    if not bool(numpy.all(numpy.isfinite(result))):
        if not (
            _comparable_finite_values(left_array, numpy)
            and _comparable_finite_values(right_array, numpy)
        ):
            return None
        result = _scaled_matmul(left_array, right_array, numpy)
        if bool(numpy.any(numpy.isnan(result))):
            return None
    return _storage(
        result,
        dtype=dtype,
        output_shape=output_shape,
        numpy=numpy,
    )


def _matrix_view(value: Any, vector: bool, *, left: bool) -> Any:
    """Promote a vector to the matrix shape used by matmul differentiation."""
    if not vector:
        return value
    if left:
        return value.reshape((1, value.shape[0]))
    return value.reshape((value.shape[0], 1))


def _matrix_gradient_view(
    gradient: Any,
    left_vector: bool,
    right_vector: bool,
) -> Any:
    """Restore the two matrix axes omitted from a public matmul result."""
    if left_vector and right_vector:
        return gradient.reshape((1, 1))
    if left_vector:
        return gradient.reshape(gradient.shape[:-1] + (1, gradient.shape[-1]))
    if right_vector:
        return gradient.reshape(gradient.shape + (1,))
    return gradient


def _reduce_matrix_gradient(
    values: Any,
    shape: tuple[int, ...],
    numpy: Any,
) -> Any | None:
    """Reduce broadcast batch axes back to one operand's matrix shape."""
    layout = _sum_axes(tuple(values.shape), shape)
    if layout is None:
        return None
    _, axes = layout
    safe = _stable_sum_candidate(values, axes, numpy)
    if not bool(safe):
        return None
    if axes:
        with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
            direct = numpy.sum(values, axis=axes, keepdims=True)
        scaled = _scaled_sum(values, axes, numpy)
        values = numpy.where(numpy.isfinite(direct), direct, scaled)
        if bool(numpy.any(numpy.isnan(values))):
            return None
    return values.reshape(shape)


def matmul_gradient(
    grad: Tensor,
    left: Tensor,
    right: Tensor,
) -> tuple[Storage, Storage] | None:
    """Run native vector-Jacobian products for general floating matmul."""
    if grad.dtype.kind != "floating":
        return None
    numpy = _numpy()
    left_vector = left.ndim == 1
    right_vector = right.ndim == 1
    try:
        upstream = _view(grad, numpy).astype(numpy.float64, copy=False)
        left_values = _view(left, numpy).astype(numpy.float64, copy=False)
        right_values = _view(right, numpy).astype(numpy.float64, copy=False)
    except ValueError:
        return None
    if not _finite_operands(
        upstream,
        left_values,
        right_values,
        numpy=numpy,
    ):
        return None

    left_matrix = _matrix_view(
        left_values,
        left_vector,
        left=True,
    )
    right_matrix = _matrix_view(
        right_values,
        right_vector,
        left=False,
    )
    matrix_grad = _matrix_gradient_view(
        upstream,
        left_vector,
        right_vector,
    )
    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        left_result = numpy.matmul(
            matrix_grad,
            numpy.swapaxes(right_matrix, -1, -2),
        )
        right_result = numpy.matmul(
            numpy.swapaxes(left_matrix, -1, -2),
            matrix_grad,
        )
    if not bool(numpy.all(numpy.isfinite(left_result))):
        if not (
            _comparable_finite_values(matrix_grad, numpy)
            and _comparable_finite_values(right_matrix, numpy)
        ):
            return None
        left_result = _scaled_matmul(
            matrix_grad,
            numpy.swapaxes(right_matrix, -1, -2),
            numpy,
        )
    if not bool(numpy.all(numpy.isfinite(right_result))):
        if not (
            _comparable_finite_values(left_matrix, numpy)
            and _comparable_finite_values(matrix_grad, numpy)
        ):
            return None
        right_result = _scaled_matmul(
            numpy.swapaxes(left_matrix, -1, -2),
            matrix_grad,
            numpy,
        )
    if bool(numpy.any(numpy.isnan(left_result)) | numpy.any(numpy.isnan(right_result))):
        return None

    left_shape = (1, left.shape[0]) if left_vector else left.shape
    right_shape = (right.shape[0], 1) if right_vector else right.shape
    left_result = _reduce_matrix_gradient(left_result, left_shape, numpy)
    right_result = _reduce_matrix_gradient(right_result, right_shape, numpy)
    if left_result is None or right_result is None:
        return None
    if left_vector:
        left_result = left_result.reshape(left.shape)
    if right_vector:
        right_result = right_result.reshape(right.shape)

    left_storage = _storage(
        left_result,
        dtype=grad.dtype,
        output_shape=left.shape,
        numpy=numpy,
    )
    right_storage = _storage(
        right_result,
        dtype=grad.dtype,
        output_shape=right.shape,
        numpy=numpy,
    )
    if left_storage is None or right_storage is None:
        return None
    return left_storage, right_storage


def _shape_size(shape: tuple[int, ...]) -> int:
    return Shape.from_iterable(shape).size
