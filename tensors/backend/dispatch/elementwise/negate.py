"""Dispatch for elementwise operations and their VJPs."""

from __future__ import annotations
from typing import TYPE_CHECKING, Any
from tensors.backend import config
from tensors.backend.config import BackendOperationUnsupportedError
from tensors.backend.loading import load_backend
from tensors.backend.storage import Storage
from tensors.backend.validation import validate_backend_residency

if TYPE_CHECKING:
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


def execute_negate(value: Tensor, *, dtype: DataType) -> Storage:
    """Negate on the selected backend, or report that it cannot run there.

    Negation is one operation whether it is written as ``-x`` or reached as
    the gradient of ``a - b`` with respect to ``b``, so it obeys the
    execution contract of `docs/backends.md` in both: the selection decides
    where it runs, at every size, and a backend that cannot negate at a dtype
    reports that rather than letting the Python reference answer.

    A second strict entry point used to sit beside this one, carrying the
    contract for the subtraction VJP while forward negation kept a
    workload-size policy. There is nothing left for it to differ about.
    """
    selected = config.get_backend()
    validate_backend_residency((value,), selected)
    backend: Any = load_backend(selected)
    result = backend.negate(value, dtype=dtype)
    if result is None:
        raise BackendOperationUnsupportedError(
            f"The {selected} backend cannot execute negate at dtype "
            f"{dtype.name} conformingly. Negation runs on the selected "
            f"backend; select another backend to run it elsewhere."
        )
    validate_backend_residency((result,), selected)
    return result
