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
int64   = DataType("int64",   "q", 8)
int32   = DataType("int32",   "i", 4)
int16   = DataType("int16",   "h", 2)
int8    = DataType("int8",    "b", 1)
uint8   = DataType("uint8",   "B", 1)

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
    "B": (0, 2 ** 8 - 1),
    "b": (-(2 ** 7), 2 ** 7 - 1),
    "h": (-(2 ** 15), 2 ** 15 - 1),
    "i": (-(2 ** 31), 2 ** 31 - 1),
    "q": (-(2 ** 63), 2 ** 63 - 1),
}


def _integer_scalar_result_dtype(a_dtype: DataType, value: int) -> DataType:
    """Promote an integer dtype enough to represent its domain and value."""
    lower, upper = _INTEGER_LIMITS[a_dtype.typecode]
    required_lower = min(lower, value)
    required_upper = max(upper, value)
    for candidate in (uint8, int8, int16, int32, int64):
        candidate_lower, candidate_upper = _INTEGER_LIMITS[candidate.typecode]
        if (
            candidate_lower <= required_lower
            and required_upper <= candidate_upper
        ):
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
