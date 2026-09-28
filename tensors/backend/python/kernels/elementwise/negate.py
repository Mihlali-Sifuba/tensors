"""Reference negation for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


def negate(value: Tensor, *, dtype: DataType) -> Storage | None:
    """Return the additive inverse of every element.

    The result is retained as arithmetic rather than as construction, so an
    integer that leaves the declared width wraps instead of raising. Only one
    value can: a signed dtype's minimum has no positive counterpart, and
    section 10.1 rule B1 makes ``-(-128)`` in ``int8`` be ``-128`` rather than
    an error, as ``127 + 1`` already is.
    """
    data = [-x for x in value._data]
    return PythonStorage.from_arithmetic(data, dtype)
