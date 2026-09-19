"""Data type definitions for tensors.

Usage::

    import tensors as ts

    t = ts.Tensor([1, 2, 3], dtype=ts.float64)
    t.dtype          # DataType.float64
    t.dtype.name     # 'float64'
    t.dtype.typecode # 'd'
    t.dtype.size     # 8
"""

from array import array
from typing import Any

_SUPPORTED_TYPECODES = {"d", "f", "q", "i", "h", "b", "B"}
_FLOAT_CODES = {"f", "d"}
_INTEGER_CODES = {"b", "B", "h", "i", "q"}


class DataType:
    """Represents a tensor data type.

    Wraps Python's ``array`` module type codes into a clean interface,
    similar to ``np.float64`` or ``tf.float32``.
    """

    def __init__(self, name: str, typecode: str, byte_size: int):
        """
        Args:
            name: Human-readable name (e.g. ``'float64'``).
            typecode: Corresponding ``array`` typecode (e.g. ``'d'``).
            byte_size: Number of bytes per element.
        """
        if not isinstance(name, str) or not name:
            raise ValueError("dtype name must be a non-empty string")
        if not isinstance(typecode, str) or typecode not in _SUPPORTED_TYPECODES:
            raise ValueError(f"Unsupported array typecode: {typecode!r}")
        if isinstance(byte_size, bool) or not isinstance(byte_size, int):
            raise TypeError("dtype byte_size must be an integer")
        actual_size = array(typecode).itemsize
        if byte_size != actual_size:
            raise ValueError(
                f"dtype {typecode!r} has byte size {actual_size}, not {byte_size}"
            )
        self._name = name
        self._typecode = typecode
        self._byte_size = byte_size

    # -- public read-only properties -------------------------------------------

    @property
    def name(self) -> str:
        """Human-readable name, e.g. ``'float64'``."""
        return self._name

    @property
    def typecode(self) -> str:
        """The ``array`` module typecode, e.g. ``'d'``."""
        return self._typecode

    @property
    def size(self) -> int:
        """Number of bytes per element."""
        return self._byte_size

    @property
    def kind(self) -> str:
        """General numeric category: ``integer`` or ``floating``."""
        if self.typecode in _INTEGER_CODES:
            return "integer"
        if self.typecode in _FLOAT_CODES:
            return "floating"
        raise TypeError(f"Unsupported dtype: {self.name}")

    # -- dunder methods --------------------------------------------------------

    def __repr__(self) -> str:
        return f"dtype('{self._name}')"

    def __str__(self) -> str:
        return self._name

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, DataType):
            return self._typecode == other._typecode
        if isinstance(other, str):
            return self._typecode == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._typecode)


# ======================================================================
#  Predefined data types  —  the public API
# ======================================================================

float64 = DataType("float64", "d", 8)
float32 = DataType("float32", "f", 4)
int64 = DataType("int64", "q", 8)
int32 = DataType("int32", "i", 4)
int16 = DataType("int16", "h", 2)
int8 = DataType("int8", "b", 1)
uint8 = DataType("uint8", "B", 1)

# Default dtype used when none is specified
default = float64


# ======================================================================
#  Lookup helpers
# ======================================================================

_TYPE_CODE_MAP = {
    "d": float64,
    "f": float32,
    "q": int64,
    "i": int32,
    "h": int16,
    "b": int8,
    "B": uint8,
}

_NAME_MAP = {
    "float64": float64,
    "float32": float32,
    "int64": int64,
    "int32": int32,
    "int16": int16,
    "int8": int8,
    "uint8": uint8,
}

_INTEGER_LIMITS = {
    "B": (0, 2**8 - 1),
    "b": (-(2**7), 2**7 - 1),
    "h": (-(2**15), 2**15 - 1),
    "i": (-(2**31), 2**31 - 1),
    "q": (-(2**63), 2**63 - 1),
}


def _integer_scalar_result_dtype(a_dtype: DataType, value: int) -> DataType:
    """Promote an integer dtype enough to represent its domain and value."""
    lower, upper = _INTEGER_LIMITS[a_dtype.typecode]
    required_lower = min(lower, value)
    required_upper = max(upper, value)
    for candidate in (uint8, int8, int16, int32, int64):
        candidate_lower, candidate_upper = _INTEGER_LIMITS[candidate.typecode]
        if candidate_lower <= required_lower and required_upper <= candidate_upper:
            return candidate
    return float64


def from_typecode(code: str) -> DataType:
    """Look up a :class:`DataType` by its typecode or human-readable name."""
    dt = _TYPE_CODE_MAP.get(code)
    if dt is None:
        dt = _NAME_MAP.get(code)
    if dt is None:
        raise ValueError(f"Unknown typecode or dtype name: {code!r}")

    return dt


# ======================================================================
#  Dtype promotion helpers
# ======================================================================


def result_dtype(
    a_dtype: DataType,
    b: Any = None,
    *,
    division: bool = False,
) -> DataType:
    """Choose a predictable result dtype for the supported numeric types."""
    b_dtype = getattr(b, "dtype", None)

    if division:
        if a_dtype.typecode == "d" or getattr(b_dtype, "typecode", None) == "d":
            return float64
        if a_dtype.typecode == "f" and (
            b_dtype is None or b_dtype.typecode in _FLOAT_CODES
        ):
            return float32
        return float64

    if b_dtype is not None:
        if a_dtype == b_dtype:
            return a_dtype
        codes = {a_dtype.typecode, b_dtype.typecode}
        if "d" in codes:
            return float64
        if "f" in codes:
            integer_dtype = b_dtype if a_dtype.typecode == "f" else a_dtype
            if integer_dtype.typecode in {"i", "q"}:
                return float64
            return float32
        if "B" in codes:
            signed_dtype = b_dtype if a_dtype.typecode == "B" else a_dtype
            return int16 if signed_dtype.typecode == "b" else signed_dtype
        return a_dtype if a_dtype.size >= b_dtype.size else b_dtype

    if a_dtype.typecode in _INTEGER_CODES:
        if isinstance(b, float):
            return float64
        if isinstance(b, int) and not isinstance(b, bool):
            return _integer_scalar_result_dtype(a_dtype, b)
    return a_dtype


def negation_dtype(a_dtype: DataType) -> DataType:
    """Unsigned bytes need a signed type to represent negative values."""
    return int16 if a_dtype.typecode == "B" else a_dtype


# ======================================================================
#  Arithmetic promotion and scalar conversion
#
#  docs/arithmetic-semantics.md section 6 is the specification for this
#  section. It governs +, -, * and / only; ``result_dtype`` above continues
#  to serve the operations outside that contract.
# ======================================================================

_INFINITY = float("inf")

#: Precision, in significand bits, of each floating dtype (section 3.2).
_FLOAT_PRECISION = {"f": 24, "d": 53}

#: Largest finite value of each floating dtype, as an exact integer, so
#: representability questions need no floating arithmetic (section 3.2).
_FLOAT_MAX = {
    "f": (2**24 - 1) * 2 ** (127 - 23),
    "d": (2**53 - 1) * 2 ** (1023 - 52),
}

#: Exponent of the largest finite value, used for the overflow threshold.
_MAX_EXPONENT = {"f": 127, "d": 1023}

#: Integer dtypes narrowest first, so promotion picks the smallest that fits.
_INTEGER_ORDER = (uint8, int8, int16, int32, int64)

#: Floating dtypes narrowest first, for the same reason.
_FLOAT_ORDER = (float32, float64)


class DtypePromotionError(TypeError):
    """No supported dtype preserves both operand domains (principle P-e).

    A ``TypeError``, because the operands are the wrong types for an implicit
    combination; the caller is expected to cast explicitly.
    """


def integer_limits(dtype: DataType) -> tuple[int, int]:
    """Return the inclusive representable range of an integer dtype."""
    return _INTEGER_LIMITS[dtype.typecode]


def exactly_representable(value: int, dtype: DataType) -> bool:
    """Whether one integer is exact in a floating dtype (section 6.5, S4).

    Writing ``|n| = m * 2**k`` with ``m`` odd, ``n`` is exact when ``m`` fits
    in the format's precision and ``|n|`` is within its finite range. The
    interval ``[-2**p, 2**p]`` is sufficient but not necessary: ``2**25`` is
    exact in float32 while ``2**24 + 1`` is not.
    """
    if value == 0:
        return True
    odd = abs(value)
    while odd % 2 == 0:
        odd //= 2
    precision = _FLOAT_PRECISION[dtype.typecode]
    return odd.bit_length() <= precision and abs(value) <= _FLOAT_MAX[dtype.typecode]


def _rounds_to_finite(value: float, dtype: DataType) -> bool:
    """Whether a finite literal survives conversion (section 6.5, S3).

    Round-to-nearest delivers infinity at or above the midpoint between the
    largest finite value and the first unrepresentable power of two. The
    comparison is exact, using the literal's rational value.
    """
    if value != value or value == _INFINITY or value == -_INFINITY:
        return True
    code = dtype.typecode
    threshold = _FLOAT_MAX[code] + 2 ** (_MAX_EXPONENT[code] + 1)
    numerator, denominator = abs(value).as_integer_ratio()
    return 2 * numerator < threshold * denominator


def _round_to_dtype(value: float, dtype: DataType) -> float:
    """Round a Python float to a floating dtype (section 6.5, S3).

    ``array`` applies the same C conversion the storage layer applies to
    every element, which is round-to-nearest, ties-to-even. Reading the
    element back widens exactly, so the literal rounds once and only once.

    Overflow is not this function's decision: ``array`` would quietly return
    an infinity, so :func:`_rounds_to_finite` gates the call and S3 raises
    instead. A literal infinity or NaN passes through unchanged.
    """
    return array(dtype.typecode, (value,))[0]


def _domain_fits(integer_dtype: DataType, float_dtype: DataType) -> bool:
    """Whether *every* value of an integer dtype is exact (section 3.3).

    This is the range question promotion asks. It is deliberately not the
    per-value question :func:`exactly_representable` answers.
    """
    lower, upper = integer_limits(integer_dtype)
    limit = 2 ** _FLOAT_PRECISION[float_dtype.typecode]
    return -limit <= lower and upper <= limit


def arithmetic_result_dtype(left: DataType, right: DataType) -> DataType:
    """Return the result dtype of ``+``, ``-`` or ``*`` (section 6.2).

    Raises :class:`DtypePromotionError` for a combination no public dtype can
    carry without losing a domain, which the caller resolves with a cast.
    """
    if left == right:
        return left
    left_integer = left.kind == "integer"
    right_integer = right.kind == "integer"

    if left_integer and right_integer:
        low = min(integer_limits(left)[0], integer_limits(right)[0])
        high = max(integer_limits(left)[1], integer_limits(right)[1])
        for candidate in _INTEGER_ORDER:
            candidate_low, candidate_high = integer_limits(candidate)
            if candidate_low <= low and high <= candidate_high:
                return candidate
        raise DtypePromotionError(
            "no supported integer dtype represents both "
            + left.name
            + " and "
            + right.name
            + "; cast one operand explicitly"
        )

    if not left_integer and not right_integer:
        return float64

    integer, floating = (left, right) if left_integer else (right, left)
    precision = _FLOAT_PRECISION[floating.typecode]
    for candidate in _FLOAT_ORDER:
        if _FLOAT_PRECISION[candidate.typecode] < precision:
            continue  # never narrow the floating operand
        if _domain_fits(integer, candidate):
            return candidate
    raise DtypePromotionError(
        "no supported floating dtype represents every "
        + integer.name
        + " value exactly, so combining it with "
        + floating.name
        + " would lose integer precision; cast explicitly, for example "
        + "x.astype(float64)"
    )


def division_result_dtype(left: DataType, right: DataType) -> DataType:
    """Return the result dtype of true division (section 7, rule D).

    Division never yields an integer. Integer operands promote first, and the
    promoted integer dtype must then be exact in a floating dtype.
    """
    promoted = arithmetic_result_dtype(left, right)
    if promoted.kind != "integer":
        return promoted
    for candidate in _FLOAT_ORDER:
        if _domain_fits(promoted, candidate):
            return candidate
    raise DtypePromotionError(
        "true division of "
        + left.name
        + " by "
        + right.name
        + " promotes to "
        + promoted.name
        + ", which no supported floating dtype represents exactly; cast "
        + "explicitly, for example x.astype(float64) / y.astype(float64)"
    )


def convert_scalar(value, dtype: DataType):
    """Convert a Python scalar to a tensor's dtype (section 6.5, S1 to S4).

    Returns the converted value, or raises :class:`TypeError`. The scalar's
    value is consulted because a Python literal carries no dtype; this is not
    the value-dependent promotion section 6.4 forbids, which concerns tensor
    operands, whose elements vary.
    """
    if isinstance(value, bool):
        raise TypeError(
            "bool is not a supported numeric scalar; there is no public "
            "boolean dtype"
        )
    if dtype.kind == "integer":
        low, high = integer_limits(dtype)
        if isinstance(value, int):
            if low <= value <= high:  # S1
                return value
            raise TypeError(
                str(value)
                + " is outside the range of "
                + dtype.name
                + "; arithmetic wraps, but an operand must be representable"
            )
        if isinstance(value, float):  # S2
            if value != value or value == _INFINITY or value == -_INFINITY:
                raise TypeError(
                    repr(value)
                    + " has no "
                    + dtype.name
                    + " value; cast the tensor to a floating dtype"
                )
            if not value.is_integer():
                raise TypeError(
                    repr(value)
                    + " is not integral, so it has no "
                    + dtype.name
                    + " value; cast the tensor to a floating dtype, for "
                    + "example x.astype(float64)"
                )
            if low <= int(value) <= high:
                return int(value)
            raise TypeError(repr(value) + " is outside the range of " + dtype.name)
        raise TypeError("Unsupported scalar type: " + type(value).__name__)

    if isinstance(value, int):  # S4
        if exactly_representable(value, dtype):
            return float(value)
        raise TypeError(
            str(value)
            + " is not exactly representable in "
            + dtype.name
            + "; cast explicitly to accept the rounding"
        )
    if isinstance(value, float):  # S3
        if _rounds_to_finite(value, dtype):
            # The *converted* value, not the literal. Returning the literal
            # left the Python backend evaluating a binary64 operand where the
            # array backends had already rounded it to the tensor's dtype, so
            # the same expression gave different results per backend.
            return _round_to_dtype(value, dtype)
        raise TypeError(
            repr(value)
            + " exceeds the range of "
            + dtype.name
            + "; cast explicitly to accept the overflow"
        )
    raise TypeError("Unsupported scalar type: " + type(value).__name__)


def resolve_binary(left_dtype: DataType, right, *, division: bool = False):
    """Return ``(result_dtype, right_operand)`` for one arithmetic call.

    ``right`` is either a Tensor, whose dtype takes part in promotion, or a
    Python scalar, which converts to ``left_dtype`` under S1 to S4 and leaves
    the result dtype to the tensor.
    """
    right_dtype = getattr(right, "dtype", None)
    if isinstance(right_dtype, DataType):
        if division:
            return division_result_dtype(left_dtype, right_dtype), right
        return arithmetic_result_dtype(left_dtype, right_dtype), right
    converted = convert_scalar(right, left_dtype)
    if division:
        return division_result_dtype(left_dtype, left_dtype), converted
    return left_dtype, converted


def resolve_power(base_dtype: DataType, exponent):
    """Return ``(result_dtype, exponent_operand)`` for ``base ** exponent``.

    Implements sections 12.5.1 and 12.5.2. ``exponent`` is either a typed
    operand — a Tensor or a Variable, whose declared dtype takes part in the
    section 6.2 promotion table — or a Python scalar, which converts to
    ``base_dtype`` under S1 to S4 and leaves the result dtype to the base.

    The exponent is not exempted from promotion on the grounds that it plays a
    different mathematical role: its value affects the result, so converting it
    to a dtype that cannot represent it exactly would change the computation.

    **No element value is ever inspected.** The result dtype follows from the
    declared dtypes and, for a scalar, from the scalar's Python type and its
    own value — never from the contents of a tensor, as section 6.4 requires.
    """
    exponent_dtype = getattr(exponent, "dtype", None)
    if isinstance(exponent_dtype, DataType):
        return arithmetic_result_dtype(base_dtype, exponent_dtype), exponent
    return base_dtype, convert_scalar(exponent, base_dtype)


def resolve_power_scalar_base(base, exponent_dtype: DataType):
    """Return ``(result_dtype, base_operand)`` for a Python scalar base.

    Implements rule S-p, section 12.5.3. S1 to S4 cannot be applied literally
    to the reflected form: they assume both operands play the same role, and
    forcing a *base* into the *exponent's* dtype would make ``2.5 ** int32_t``
    raise, refusing an ordinary real result because the exponent happens to be
    an integer tensor.

    Reflected exponentiation does not thereby bypass the promotion policy:
    a Python float base over an ``int64`` exponent still reaches the section
    6.2 ``cast`` cell and raises.
    """
    if exponent_dtype.kind == "floating":
        # The exponent supplies the target; S3 for a float, S4 for an int.
        return exponent_dtype, convert_scalar(base, exponent_dtype)
    if isinstance(base, int):
        # Including bool, which convert_scalar rejects on its own terms.
        return exponent_dtype, convert_scalar(base, exponent_dtype)
    # A Python float base is taken as float64, the default floating dtype, and
    # the ordinary promotion restrictions then apply against the exponent.
    result = arithmetic_result_dtype(float64, exponent_dtype)
    return result, convert_scalar(base, result)
