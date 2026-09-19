"""Canonical storage for the dependency-free Python backend."""

from __future__ import annotations
from array import array
from collections.abc import Iterable
from tensors._typing import Scalar
from tensors.dtype import DataType
from tensors.utils.integers import wrap
from tensors.backend.storage import Storage


class PythonStorage(Storage):
    """Own a flat :class:`array.array` buffer."""

    kind = "python"

    def __init__(self, buffer: array, dtype: DataType, *, copy: bool = False) -> None:
        super().__init__(dtype)
        if buffer.typecode != dtype.typecode:
            raise TypeError(
                f"storage typecode {buffer.typecode!r} does not match dtype {dtype.name!r}"
            )
        self._buffer = array(buffer.typecode, buffer) if copy else buffer

    @classmethod
    def from_arithmetic(
        cls, values: Iterable[Scalar], dtype: DataType
    ) -> PythonStorage:
        """Store an arithmetic result, reducing integers to the declared width.

        Construction and arithmetic differ here, and deliberately: an
        out-of-range literal is a caller error and :meth:`from_values` raises
        for it, while an out-of-range arithmetic result is the defined
        consequence of a finite width and wraps. See
        docs/arithmetic-semantics.md sections 4.2 and 4.5.
        """
        if dtype.kind != "integer":
            return cls.from_values(values, dtype)
        if isinstance(values, array) and values.typecode == dtype.typecode:
            return cls(values, dtype, copy=True)
        wrapped = (wrap(int(value), dtype) for value in values)
        return cls(array(dtype.typecode, wrapped), dtype)

    @classmethod
    def from_values(cls, values: Iterable[Scalar], dtype: DataType) -> PythonStorage:
        """Convert Python numeric values using Tensor construction rules."""
        if isinstance(values, array) and values.typecode == dtype.typecode:
            return cls(values, dtype, copy=True)
        if dtype.kind == "integer":
            converted = (int(value) for value in values)
        else:
            converted = (float(value) for value in values)
        return cls(array(dtype.typecode, converted), dtype)

    @property
    def buffer(self) -> array:
        return self._buffer

    def copy(self) -> PythonStorage:
        return PythonStorage(self.buffer, self.dtype, copy=True)
