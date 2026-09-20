"""Native storage for the optional NumPy backend."""

from __future__ import annotations
import importlib
from collections.abc import Iterable
from typing import Any
from tensors._typing import Scalar
from tensors.dtype import DataType
from tensors.backend.storage import Storage


class NumPyStorage(Storage):
    """Own or retain a flat contiguous ``numpy.ndarray``."""

    kind = "numpy"

    def __init__(self, buffer: Any, dtype: DataType, *, copy: bool = False) -> None:
        super().__init__(dtype)
        numpy = importlib.import_module("numpy")
        values = numpy.asarray(buffer, dtype=numpy.dtype(dtype.name)).reshape(-1)
        if not values.flags.c_contiguous:
            values = numpy.ascontiguousarray(values)
        self._buffer = values.copy() if copy else values

    @classmethod
    def from_values(cls, values: Iterable[Scalar], dtype: DataType) -> NumPyStorage:
        """Construct NumPy-native storage directly from host values."""
        numpy = importlib.import_module("numpy")
        buffer = numpy.fromiter(values, dtype=numpy.dtype(dtype.name))
        return cls(buffer, dtype)

    @property
    def buffer(self) -> Any:
        return self._buffer

    def copy(self) -> NumPyStorage:
        return NumPyStorage(self.buffer, self.dtype, copy=True)
