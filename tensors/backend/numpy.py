"""Optional NumPy kernels backed by canonical ``array.array`` storage."""

from __future__ import annotations

import importlib
from array import array
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from .._typing import Scalar, TensorIndex
    from ..dtype import DataType
    from ..tensor import Tensor
    from . import BinaryOperation, ReductionOperation


def _view(tensor: Tensor, numpy: Any) -> Any:
    """Return a zero-copy NumPy view of a tensor's canonical storage."""
    storage = cast(array[Any], getattr(tensor, "_data"))
    return numpy.frombuffer(
        storage,
        dtype=numpy.dtype(tensor.dtype.name),
        count=tensor.size,
    ).reshape(tensor.shape)


def _operand(value: Tensor | Scalar, dtype: DataType, numpy: Any) -> Any:
    """Return a NumPy operand with Python-reference working precision."""
    from ..tensor import Tensor

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
) -> array[Any] | None:
    """Convert a NumPy result to canonical storage without changing semantics."""
    flattened = numpy.asarray(result).reshape(-1)
    if dtype.kind == "integer":
        storage = array(dtype.typecode, (int(value) for value in flattened))
    else:
        target_dtype = numpy.dtype(dtype.name)
        finite = numpy.isfinite(flattened)
        outside_range = numpy.abs(flattened) > numpy.finfo(target_dtype).max
        if bool(numpy.any(finite & outside_range)):
            return None
        with numpy.errstate(over="ignore", under="ignore", invalid="ignore"):
            contiguous = numpy.asarray(flattened, dtype=target_dtype)
        storage = array(dtype.typecode)
        storage.frombytes(contiguous.tobytes())
    if len(storage) != _shape_size(output_shape):
        raise RuntimeError("NumPy kernel returned an unexpected result size")
    return storage


def binary(
    operation: BinaryOperation,
    left: Tensor | Scalar,
    right: Tensor | Scalar,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> array[Any] | None:
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
    with numpy.errstate(
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


def negate(value: Tensor, *, dtype: DataType) -> array[Any] | None:
    """Run elementwise NumPy negation."""
    numpy = _numpy()
    try:
        operand = _operand(value, dtype, numpy)
    except ValueError:
        return None
    result = numpy.negative(operand)
    return _storage(
        result,
        dtype=dtype,
        output_shape=value.shape,
        numpy=numpy,
    )


def slice_tensor(
    value: Tensor,
    key: TensorIndex,
    *,
    output_shape: tuple[int, ...],
) -> array[Any] | None:
    """Run a NumPy slicing kernel after caller-side key validation."""
    numpy = _numpy()
    try:
        result = _view(value, numpy)[key]
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
) -> array[Any] | None:
    """Scatter flat values into a zero NumPy tensor."""
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


def cast_tensor(value: Tensor, *, dtype: DataType) -> array[Any] | None:
    """Convert tensor values with Python-compatible scalar conversion."""
    numpy = _numpy()
    try:
        source = _view(value, numpy).reshape(-1)
    except ValueError:
        return None
    if dtype.kind == "integer":
        converter = numpy.frompyfunc(int, 1, 1)
        result = converter(source)
    else:
        result = source.astype(numpy.float64, copy=False)
    return _storage(
        result,
        dtype=dtype,
        output_shape=value.shape,
        numpy=numpy,
    )


def reduction(
    operation: ReductionOperation,
    value: Tensor,
    axes: tuple[int, ...],
    *,
    keepdims: bool,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> array[Any] | None:
    """Run a numerically guarded NumPy reduction."""
    if value.size == 0:
        return None

    numpy = _numpy()
    values = _view(value, numpy).astype(numpy.float64, copy=False)
    if not bool(numpy.all(numpy.isfinite(values))):
        return None

    axis = axes
    if operation in {"sum", "mean"}:
        if operation == "sum" and dtype.kind == "integer":
            return None
        has_positive = bool(numpy.any(values > 0.0))
        has_negative = bool(numpy.any(values < 0.0))
        has_subnormal = bool(
            numpy.any(
                (values != 0.0)
                & (numpy.abs(values) < numpy.finfo(numpy.float64).tiny)
            )
        )
        if (has_positive and has_negative) or has_subnormal:
            return None
        with numpy.errstate(over="ignore", under="ignore", invalid="ignore"):
            if operation == "sum":
                result = numpy.sum(values, axis=axis, keepdims=keepdims)
            else:
                result = numpy.mean(values, axis=axis, keepdims=keepdims)
        if not bool(numpy.all(numpy.isfinite(result))):
            return None
    elif operation == "variance":
        with numpy.errstate(over="ignore", under="ignore", invalid="ignore"):
            center = numpy.mean(values, axis=axis, keepdims=True)
            centered = values - center
        if not bool(numpy.all(numpy.isfinite(centered))):
            return None
        scale = numpy.max(numpy.abs(centered), axis=axis, keepdims=True)
        safe_scale = numpy.where(scale == 0.0, 1.0, scale)
        normalized = centered / safe_scale
        normalized_variance = numpy.mean(
            normalized * normalized,
            axis=axis,
            keepdims=keepdims,
        )
        output_scale = scale if keepdims else numpy.squeeze(scale, axis=axis)
        with numpy.errstate(over="ignore", under="ignore", invalid="ignore"):
            deviation = output_scale * numpy.sqrt(normalized_variance)
            result = deviation * deviation
    else:
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
        output_scale = scale if keepdims else numpy.squeeze(scale, axis=axis)
        with numpy.errstate(over="ignore", under="ignore", invalid="ignore"):
            result = output_scale * normalized_magnitude

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
) -> array[Any] | None:
    """Calculate ``-grad * numerator / denominator**2`` when range-safe."""
    numpy = _numpy()
    try:
        upstream = _operand(grad, grad.dtype, numpy)
        values = _operand(numerator, grad.dtype, numpy)
        divisors = _operand(denominator, grad.dtype, numpy)
    except (OverflowError, TypeError, ValueError):
        return None
    with numpy.errstate(over="ignore", under="ignore", invalid="ignore"):
        squares = numpy.square(divisors)
    if _finite_operands(
        upstream,
        values,
        divisors,
        numpy=numpy,
    ) and bool(numpy.any((squares == 0.0) | ~numpy.isfinite(squares))):
        return None
    with numpy.errstate(
        divide="ignore",
        over="ignore",
        under="ignore",
        invalid="ignore",
    ):
        result = -upstream * values / squares
    if _unsafe_finite_result(
        result,
        upstream,
        values,
        divisors,
        numpy=numpy,
    ):
        return None
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
) -> array[Any] | None:
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
    with numpy.errstate(over="ignore", under="ignore", invalid="ignore"):
        power_term = numpy.power(bases, powers - 1.0)
    if bool(numpy.any((power_term == 0.0) | ~numpy.isfinite(power_term))):
        return None
    with numpy.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = upstream * powers * power_term
    if not bool(numpy.all(numpy.isfinite(result))):
        return None
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
) -> array[Any] | None:
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
    with numpy.errstate(over="ignore", under="ignore", invalid="ignore"):
        outputs = numpy.power(bases, powers)
    if bool(numpy.any((outputs == 0.0) | ~numpy.isfinite(outputs))):
        return None
    with numpy.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = upstream * outputs * numpy.log(bases)
    if not bool(numpy.all(numpy.isfinite(result))):
        return None
    return _storage(
        result,
        dtype=grad.dtype,
        output_shape=grad.shape,
        numpy=numpy,
    )


def _finite_operands(*operands: Any, numpy: Any) -> bool:
    return all(
        bool(numpy.all(numpy.isfinite(operand)))
        for operand in operands
    )


def _unsafe_finite_result(
    result: Any,
    *operands: Any,
    numpy: Any,
) -> bool:
    return _finite_operands(*operands, numpy=numpy) and not bool(
        numpy.all(numpy.isfinite(result))
    )


def _numpy() -> Any:
    """Import NumPy only after its backend has been selected."""
    try:
        return importlib.import_module("numpy")
    except ImportError as error:
        from . import BackendUnavailableError

        raise BackendUnavailableError(
            "The NumPy backend became unavailable after it was selected."
        ) from error


def matmul(
    left: Tensor,
    right: Tensor,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> array[Any] | None:
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
    with numpy.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = numpy.matmul(left_array, right_array)

    # NumPy may overflow an intermediate product that the Python reference can
    # recover through exact-ratio summation. Preserve those semantics by asking
    # the caller to use the reference implementation whenever the fast result is
    # non-finite or cannot be represented by the requested output dtype.
    if not bool(numpy.all(numpy.isfinite(result))):
        return None
    return _storage(
        result,
        dtype=dtype,
        output_shape=output_shape,
        numpy=numpy,
    )


def _shape_size(shape: tuple[int, ...]) -> int:
    size = 1
    for dimension in shape:
        size *= dimension
    return size
