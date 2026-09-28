"""Dispatch for reductions, extrema indices, and shape summation."""

from __future__ import annotations
from typing import TYPE_CHECKING, Any
from tensors.backend import config
from tensors.backend.config import BackendOperationUnsupportedError
from tensors.backend.loading import load_backend
from tensors.backend.storage import Storage
from tensors.backend.validation import validate_backend_residency

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def execute_sum_products_to_shape(
    gradient: Tensor, factor: Tensor, shape: tuple[int, ...]
) -> Storage:
    """Run the multiply-and-broadcast reduction on the selected backend.

    This is the whole vector-Jacobian product of multiplication, and
    ``Mul.backward`` is its only caller, so it carries the arithmetic
    execution contract directly: the selection decides where it runs, at
    every size, and a backend that declines is reported rather than replaced.
    """
    selected = config.get_backend()
    validate_backend_residency((gradient, factor), selected)
    backend: Any = load_backend(selected)
    result = backend.sum_products_to_shape(gradient, factor, shape)
    if result is None:
        raise BackendOperationUnsupportedError(
            f"The {selected} backend cannot execute sum_products_to_shape at "
            f"dtype {gradient.dtype.name} conformingly. This computation runs "
            f"on the selected backend; select another backend to run it "
            f"elsewhere."
        )
    validate_backend_residency((result,), selected)
    return result
