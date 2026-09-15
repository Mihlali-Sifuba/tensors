"""Construct the cuda arithmetic provider lazily."""

from ..kernels.core import _import_array_module
from . import ArrayBackend, bind_array_backend


def create_backend() -> ArrayBackend:
    return bind_array_backend(_import_array_module("cupy"))
