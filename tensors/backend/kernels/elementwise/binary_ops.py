"""Binary arithmetic kernels and their vector-Jacobian products."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...storage import Storage
from ..core import _errstate, _finite_operands, _numpy, _operand, _storage

if TYPE_CHECKING:
    from ...._typing import Scalar
    from ....dtype import DataType
    from ....tensor import Tensor
    from ...types import BinaryOperation

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
