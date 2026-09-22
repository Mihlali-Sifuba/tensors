"""Dispatch for shape, layout, indexing, and representation changes."""

from __future__ import annotations
from typing import TYPE_CHECKING, Any
from tensors.backend import config
from tensors.backend.config import BackendOperationUnsupportedError
from tensors.backend.loading import load_backend
from tensors.backend.storage import Storage
from tensors.backend.validation import validate_backend_residency

if TYPE_CHECKING:
    from tensors._typing import TensorIndex
    from tensors.tensor import Tensor


def execute_slice(
    value: Tensor, key: TensorIndex, *, output_shape: tuple[int, ...]
) -> Storage:
    """Select elements on the selected backend, whatever the selection size.

    Splitting a stacked gradient back to the contributions it came from is a
    slice, so a reverse pass runs this, and it carries the execution contract
    of `docs/backends.md`: the selection decides where it runs and a backend
    that cannot select conformingly reports that rather than letting the
    Python reference answer. Under the workload-size policy this carried, a
    small selection came back in Python storage, and the accumulated gradient
    built from it then met the next strict boundary residing elsewhere.

    Nothing is calculated here. The key is validated and the output shape
    resolved by the caller, and each backend's kernel copies the same
    selected elements in the same logical order, so no numerical decision
    depends on where this runs.
    """
    selected = config.get_backend()
    validate_backend_residency((value,), selected)
    backend: Any = load_backend(selected)
    result = backend.slice_tensor(value, key, output_shape=output_shape)
    if result is None:
        raise BackendOperationUnsupportedError(
            f"The {selected} backend cannot execute slice_tensor at dtype "
            f"{value.dtype.name} conformingly. This computation runs on the "
            f"selected backend; select another backend to run it elsewhere."
        )
    validate_backend_residency((result,), selected)
    return result
