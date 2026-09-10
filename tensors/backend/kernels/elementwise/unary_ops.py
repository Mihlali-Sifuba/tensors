"""Unary transform kernels and their vector-Jacobian products."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...storage import Storage
from ..core import _errstate, _numpy, _operand, _storage, _view

if TYPE_CHECKING:
    from ....dtype import DataType
    from ....tensor import Tensor
    from ...types import UnaryOperation

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
