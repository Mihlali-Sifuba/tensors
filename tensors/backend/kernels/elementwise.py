"""Elementwise kernels and their vector-Jacobian products."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..storage import Storage
from .core import _errstate, _finite_operands, _numpy, _operand, _storage, _view

if TYPE_CHECKING:
    from ..._typing import Scalar
    from ...dtype import DataType
    from ...tensor import Tensor
    from ..types import (
        BinaryOperation,
        ComparisonOperation,
        ExtremumOperation,
        UnaryOperation,
    )

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

def comparison(
    operation: ComparisonOperation,
    left: Tensor,
    right: Tensor,
    *,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Run a broadcasting elementwise comparison."""
    from ...dtype import uint8

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
    *,
    needs_input_grad: tuple[bool, ...] = (True, True),
) -> tuple[Storage | None, Storage | None] | None:
    """Split a selection VJP into the requested left and right terms."""
    numpy = _numpy()
    need_left, need_right = needs_input_grad
    try:
        selected = numpy.broadcast_to(_view(condition, numpy), grad.shape) != 0
    except ValueError:
        return None
    upstream = _view(grad, numpy).astype(numpy.float64, copy=False)
    left = None
    if need_left:
        left = _storage(
            numpy.where(selected, upstream, 0.0),
            dtype=grad.dtype,
            output_shape=grad.shape,
            numpy=numpy,
        )
        if left is None:
            return None
    right = None
    if need_right:
        right = _storage(
            numpy.where(selected, 0.0, upstream),
            dtype=grad.dtype,
            output_shape=grad.shape,
            numpy=numpy,
        )
        if right is None:
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
    *,
    needs_input_grad: tuple[bool, ...] = (True, True),
) -> tuple[Storage | None, Storage | None] | None:
    """Split an elementwise-extremum VJP for the requested operands."""
    numpy = _numpy()
    need_left, need_right = needs_input_grad
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
    left_storage = None
    if need_left:
        left_weight = numpy.where(
            has_nan,
            numpy.nan,
            numpy.where(ties, 0.5, numpy.where(left_selected, 1.0, 0.0)),
        )
        left_storage = _storage(
            upstream * left_weight,
            dtype=grad.dtype,
            output_shape=grad.shape,
            numpy=numpy,
        )
        if left_storage is None:
            return None
    right_storage = None
    if need_right:
        right_weight = numpy.where(
            has_nan,
            numpy.nan,
            numpy.where(ties, 0.5, numpy.where(left_selected, 0.0, 1.0)),
        )
        right_storage = _storage(
            upstream * right_weight,
            dtype=grad.dtype,
            output_shape=grad.shape,
            numpy=numpy,
        )
        if right_storage is None:
            return None
    return left_storage, right_storage

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
