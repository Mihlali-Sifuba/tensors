"""Device-resident storage for the optional CUDA backend."""

from __future__ import annotations

import importlib
from typing import Any

from ..dtype import DataType
from ._base import Storage


class CudaStorage(Storage):
    """Own a flat contiguous ``cupy.ndarray`` on its current CUDA device."""

    kind = "cuda"

    def __init__(
        self,
        buffer: Any,
        dtype: DataType,
        *,
        copy: bool = False,
    ) -> None:
        super().__init__(dtype)
        cupy = importlib.import_module("cupy")
        values = cupy.asarray(buffer, dtype=cupy.dtype(dtype.name)).reshape(-1)
        if not values.flags.c_contiguous:
            values = cupy.ascontiguousarray(values)
        self._buffer = values.copy() if copy else values

    @property
    def buffer(self) -> Any:
        return self._buffer

    @property
    def device_id(self) -> int:
        """CUDA device that owns the allocation."""
        return int(self.buffer.device.id)

    def copy(self) -> CudaStorage:
        return CudaStorage(self.buffer, self.dtype, copy=True)
