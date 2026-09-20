"""Dispatch for forward power; resolve the selection once per call.

Where forward power executes is decided by the backend selection and by
nothing else. `docs/backends.md`, *Execution requirements*, makes that a
requirement rather than a preference: a one-element power runs where a
million-element power runs, and a provider that cannot execute the operation
says so instead of handing the work to another backend.

``"auto"`` needs no case of its own. It resolves to a concrete backend when it
is selected — NumPy when NumPy is installed, Python otherwise — so by the time
a call arrives the selection names one backend.

A decline now raises, as it does for ``+``, ``-``, ``*`` and ``/``. It used to
fall back to the Python reference, for three reasons that no longer hold:

- **Domain errors.** The array kernels declined on a non-finite result from
  finite operands so the reference could raise ``ValueError`` or
  ``OverflowError``. Sections 12.2 and 12.3 replaced those errors with values:
  a negative base with a non-integral exponent is NaN, a zero base with a
  negative exponent is a signed infinity, and an overflowing result is an
  infinity. Nothing declines for them any more.
- **CUDA integer exponentiation.** The CUDA kernel has had a native
  fixed-width implementation since section 12.4 was implemented; it runs
  device-resident for all five integer dtypes and wraps in the declared width.
- **A narrowing integer result.** Section 12.4.1 makes wraparound the
  specified result, so there is nothing to narrow and nothing to decline.

The one numerical error that remains, ``ValueError`` for a negative integer
exponent under section 12.4.2, is raised by :class:`Pow.forward` before this
dispatcher is reached. It is a numerical-domain error and stays distinct from
a capability failure.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any
from tensors.backend import config
from tensors.backend.config import BackendOperationUnsupportedError
from tensors.backend.loading import load_backend
from tensors.backend.storage import Storage
from tensors.backend.validation import validate_backend_residency

if TYPE_CHECKING:
    from tensors._typing import Scalar
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


def execute_power(
    left: Tensor | Scalar,
    right: Tensor | Scalar,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Run power on the selected backend, or report that it cannot run there."""
    selected = config.get_backend()
    validate_backend_residency((left, right), selected)
    if selected == "python":
        from tensors.backend.python.kernels.arithmetic.power import (
            power as reference,
        )

        result = reference(left, right, dtype=dtype, output_shape=output_shape)
        validate_backend_residency((result,), selected)
        return result

    backend: Any = load_backend(selected)
    result = backend.power(left, right, dtype=dtype, output_shape=output_shape)
    if result is None:
        raise BackendOperationUnsupportedError(
            f"The {selected} backend cannot execute power at dtype "
            f"{dtype.name} conformingly. Arithmetic runs on the selected "
            f"backend; select another backend to run it elsewhere."
        )
    validate_backend_residency((result,), selected)
    return result
