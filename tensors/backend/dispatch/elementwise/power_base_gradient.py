"""Dispatch for the power-base VJP.

`docs/arithmetic-semantics.md` rule G6: power gradients execute on the
selected backend at every tensor size, or report that they cannot. There is no
workload threshold, no silent fallback to another backend, and no decline that
reads operand values.

All three used to be here. A NumPy gradient of fewer than 32 elements ran the
Python reference; and a kernel that met a negative base, a zero base or a
single infinity anywhere in the tensor returned ``None``, which sent the whole
gradient to Python — a decline that had to read the operands to make, which
rule G3 forbids as well. The kernels now answer every region themselves, so a
``None`` here means a genuine capability gap and is reported as one.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any
from tensors.backend.config import (
    BackendOperationUnsupportedError,
    get_backend,
)
from tensors.backend.loading import load_backend
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def execute_power_base_gradient(
    grad: Tensor, base: Tensor, exponent: Tensor
) -> Storage:
    """Run the power-base VJP on the selected backend, whatever its size."""
    selected = get_backend()
    if selected == "python":
        from tensors.backend.python.kernels.elementwise.power_base_gradient import (
            power_base_gradient as reference,
        )

        return reference(grad, base, exponent)

    backend: Any = load_backend(selected)
    result = backend.power_base_gradient(grad, base, exponent)
    if result is None:
        raise BackendOperationUnsupportedError(
            f"The {selected} backend cannot execute the power base gradient at "
            f"dtype {base.dtype.name} conformingly. Power gradients run on the "
            f"selected backend; select another backend to run them elsewhere."
        )
    return result
