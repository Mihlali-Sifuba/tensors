"""Selected-backend storage construction from host values."""

from __future__ import annotations

from collections.abc import Iterable

from tensors._typing import Scalar
from tensors.backend import config
from tensors.backend.cuda.storage import CudaStorage
from tensors.backend.numpy.storage import NumPyStorage
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import DataType


def storage_from_values(values: Iterable[Scalar], dtype: DataType) -> Storage:
    """Construct authoritative storage directly for the selected backend."""
    selected = config.get_backend()
    if selected == "python":
        return PythonStorage.from_values(values, dtype)
    if selected == "numpy":
        return NumPyStorage.from_values(values, dtype)
    return CudaStorage.from_values(values, dtype)


__all__ = ["storage_from_values"]
