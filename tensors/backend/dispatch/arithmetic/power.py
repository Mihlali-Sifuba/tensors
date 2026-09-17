"""Dispatch for forward power; resolve the selection once per call.

Where forward power executes is decided by the backend selection and by
nothing else. `docs/backends.md`, *Execution requirements*, makes that a
requirement rather than a preference, so the workload-size policy that used to
send small exponentiations to Python no longer has a say here: a one-element
power runs where a million-element power runs.

``"auto"`` needs no case of its own. It resolves to a concrete backend when it
is selected — NumPy when NumPy is installed, Python otherwise — so by the time
a call arrives the selection names one backend.

This does not use :func:`execute_arithmetic`, which raises when a provider
declines. Power's kernels use a decline to mean three different things, and
two of them are behaviour this dispatcher must preserve:

- **A domain error.** ``0 ** -1`` and ``(-2.0) ** 0.5`` are undefined, and an
  overflowing result is an ``OverflowError``. The array kernels detect these
  as a non-finite result from finite operands and decline; the reference is
  what turns that into the documented ``ValueError`` or ``OverflowError``.
- **A capability gap.** CuPy integer exponentiation is refused by the CUDA
  conversion helper, so ``2 ** int32_tensor`` declines on CUDA and the
  reference answers it correctly.
- **A narrowing result**, where the exact integer power leaves the declared
  dtype.

Raising on every decline would replace those documented errors with an
unsupported-operation error and would stop CUDA integer power working, which
is why the reference still answers a decline. Closing that properly means
teaching the array kernels to raise their own domain errors and giving CUDA a
native integer path; until then this dispatcher guarantees execution location
for the supported cases only, and `docs/backends.md` records the gap.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from tensors.backend.config import get_backend
from tensors.backend.loading import load_backend
from tensors.backend.storage import Storage

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
    """Run forward power on the selected backend, whatever the workload size."""
    from tensors.backend.python.kernels.arithmetic.power import power as reference

    selected = get_backend()
    if selected == "python":
        return reference(left, right, dtype=dtype, output_shape=output_shape)

    backend = load_backend(selected)
    result = backend.power(left, right, dtype=dtype, output_shape=output_shape)
    if result is not None:
        return result
    # A decline is one of the three cases in the module docstring. Each needs
    # the reference to produce the documented result or error.
    return reference(left, right, dtype=dtype, output_shape=output_shape)
