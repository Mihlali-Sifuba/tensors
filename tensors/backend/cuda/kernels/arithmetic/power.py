"""CuPy implementation of exponentiation."""

from __future__ import annotations
import cupy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.cuda.conversion import _errstate
from tensors.backend.cuda.conversion import _operand
from tensors.backend.cuda.conversion import _arithmetic_operand
from tensors.backend.cuda.conversion import _arithmetic_storage
from tensors.backend.cuda.conversion import _storage
from tensors.backend.cuda.conversion import _finite_operands

if TYPE_CHECKING:
    from tensors._typing import Scalar
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


#: The unsigned dtype of each width. Modular arithmetic is defined for
#: unsigned integers, and the bit pattern of a two's-complement signed value
#: is the same, so the whole computation runs unsigned and is reinterpreted at
#: the end rather than relying on signed overflow, which C leaves undefined.
_UNSIGNED_OF_WIDTH = {1: "uint8", 2: "uint16", 4: "uint32", 8: "uint64"}


def _integer_power(left, right, native):
    """Exponentiation by squaring in a fixed width (section 12.4.1).

    The exponent is non-negative: section 12.4.2 rejects a negative one before
    evaluation. Each iteration handles one bit of the exponent, so the work is
    O(log n) elementwise multiplications rather than n of them, and every
    product wraps at the declared width because it is computed unsigned.
    """
    unsigned = cupy.dtype(_UNSIGNED_OF_WIDTH[native.itemsize])

    base = cupy.ascontiguousarray(cupy.asarray(left, dtype=native))
    exponent = cupy.ascontiguousarray(cupy.asarray(right, dtype=native))
    base, exponent = cupy.broadcast_arrays(base, exponent)

    # `.view` reinterprets the bits and is exact for both signs; `.astype`
    # would be a value conversion, and out-of-range signed conversion is
    # implementation-defined.
    accumulator = cupy.ascontiguousarray(base).view(unsigned).copy()
    remaining = cupy.ascontiguousarray(exponent).view(unsigned).copy()
    result = cupy.ones(accumulator.shape, dtype=unsigned)

    one = unsigned.type(1)
    largest = int(remaining.max()) if remaining.size else 0
    for _ in range(max(largest.bit_length(), 1)):
        odd = (remaining & one).astype(bool)
        result = cupy.where(odd, result * accumulator, result)
        accumulator = accumulator * accumulator
        remaining = remaining >> one
    return result.view(native)


def power(
    left: Tensor | Scalar,
    right: Tensor | Scalar,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Return native storage, or decline when reference semantics require it."""
    if dtype.kind == "integer":
        # Native fixed-width exponentiation; see _integer_power. This is what
        # closes the gap where integer operands were declined and the Python
        # reference answered instead.
        native = cupy.dtype(dtype.name)
        with _errstate(over="ignore", under="ignore", invalid="ignore"):
            result = _integer_power(
                _arithmetic_operand(left, dtype),
                _arithmetic_operand(right, dtype),
                native,
            )
        return _arithmetic_storage(result, dtype=dtype, output_shape=output_shape)

    try:
        left_array = _operand(left, dtype)
        right_array = _operand(right, dtype)
    except (OverflowError, TypeError, ValueError):
        return None
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        result = cupy.power(left_array, right_array)
    if (
        dtype.kind == "floating"
        and _finite_operands(left_array, right_array)
        and (not bool(cupy.all(cupy.isfinite(result))))
    ):
        return None
    return _storage(result, dtype=dtype, output_shape=output_shape)
