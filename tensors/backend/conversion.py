"""Lazy conversion between backend-native storage representations."""

from __future__ import annotations
import importlib
from array import array
from tensors.backend.storage import Storage
from tensors.backend.storage import StorageKind
from tensors.backend.cuda.storage import CudaStorage
from tensors.backend.numpy.storage import NumPyStorage
from tensors.backend.python.storage import PythonStorage


def convert_storage(storage: Storage, kind: StorageKind) -> Storage:
    """Return an equivalent storage representation for ``kind``."""
    if storage.kind == kind:
        return storage
    dtype = storage.dtype
    if kind == "python":
        if isinstance(storage, CudaStorage):
            cupy = importlib.import_module("cupy")
            values = cupy.asnumpy(storage.buffer)
        else:
            values = storage.buffer
        result = array(dtype.typecode)
        result.frombytes(values.astype(dtype.name, copy=False).tobytes())
        return PythonStorage(result, dtype)
    if kind == "numpy":
        numpy = importlib.import_module("numpy")
        if isinstance(storage, PythonStorage):
            values = numpy.frombuffer(storage.buffer, dtype=numpy.dtype(dtype.name))
        elif isinstance(storage, CudaStorage):
            cupy = importlib.import_module("cupy")
            values = cupy.asnumpy(storage.buffer)
        else:
            values = storage.buffer
        return NumPyStorage(values, dtype)
    cupy = importlib.import_module("cupy")
    if isinstance(storage, PythonStorage):
        numpy = importlib.import_module("numpy")
        values = numpy.frombuffer(storage.buffer, dtype=numpy.dtype(dtype.name))
    else:
        values = storage.buffer
    return CudaStorage(cupy.asarray(values), dtype)
