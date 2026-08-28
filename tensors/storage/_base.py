"""Backend-native tensor storage contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal, TypeAlias

from ..dtype import DataType


StorageKind: TypeAlias = Literal["python", "numpy", "cuda"]


class Storage(ABC):
    """Own a flat, contiguous numeric buffer for one execution backend."""

    kind: StorageKind

    def __init__(self, dtype: DataType) -> None:
        self._dtype = dtype

    @property
    def dtype(self) -> DataType:
        """Data type represented by this storage."""
        return self._dtype

    @property
    @abstractmethod
    def buffer(self) -> Any:
        """Return the backend-native flat buffer."""

    @property
    def size(self) -> int:
        """Number of stored elements."""
        return len(self.buffer)

    @abstractmethod
    def copy(self) -> Storage:
        """Return independent storage with the same values and dtype."""
