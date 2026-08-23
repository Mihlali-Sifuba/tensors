"""Native storage for the optional NumPy backend."""

from __future__ import annotations

import importlib
from typing import Any

from ..dtype import DataType
from ._base import Storage


class NumPyStorage(Storage):
    """Own or retain a flat contiguous ``numpy.ndarray``."""

    kind = "numpy"

    def __init__(
        self,
        buffer: Any,
        dtype: DataType,
        *,
        copy: bool = False,
    ) -> None:
        super().__init__(dtype)
        numpy = importlib.import_module("numpy")
        values = numpy.asarray(buffer, dtype=numpy.dtype(dtype.name)).reshape(-1)
        if not values.flags.c_contiguous:
            values = numpy.ascontiguousarray(values)
        self._buffer = values.copy() if copy else values

    @property
    def buffer(self) -> Any:
        return self._buffer

    def copy(self) -> NumPyStorage:
        return NumPyStorage(self.buffer, self.dtype, copy=True)
