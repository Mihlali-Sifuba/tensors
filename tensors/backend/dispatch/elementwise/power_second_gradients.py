"""Dispatch for the three second partial derivatives of a power.

`docs/arithmetic-semantics.md` rule G6 requires power's gradients to execute
on the selected backend at every tensor size, or to report that they cannot.
These carry the same requirement, for the same reason: a second derivative
that ran a host loop and handed the value back would put the whole of a
reverse-over-reverse pass on the host, whatever backend was selected.

They are backend primitives because their computation cannot be assembled
from the operations already available. Each is a few small factors times a
power, and the power is the part that leaves the representable range while
the whole expression stays inside it, so the factors have to be grouped
before rounding, in one kernel. Writing the same expression with ``log`` and
``*`` would also lose the selected backend, because ``log`` still answers to
the workload policy and returns host storage for a small operand.

Nothing here raises on a numerical condition and nothing reads an operand to
decide a region (rules G2 and G3); each kernel answers every region itself,
so a ``None`` means a genuine capability gap and is reported as one.
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
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


def _unsupported(
    backend: str, name: str, dtype: DataType
) -> BackendOperationUnsupportedError:
    return BackendOperationUnsupportedError(
        f"The {backend} backend cannot execute {name} at dtype {dtype.name} "
        f"conformingly. Power gradients run on the selected backend; select "
        f"another backend to run them elsewhere."
    )


def execute_power_base_base_gradient(
    outer: Tensor, grad: Tensor, base: Tensor, exponent: Tensor
) -> Storage:
    """Run ``d2(base ** exponent)/d(base)2``, weighted by both gradients."""
    selected = get_backend()
    if selected == "python":
        from tensors.backend.python.kernels.elementwise.power_base_base_gradient import (
            power_base_base_gradient as reference,
        )

        return reference(outer, grad, base, exponent)

    backend: Any = load_backend(selected)
    result = backend.power_base_base_gradient(outer, grad, base, exponent)
    if result is None:
        raise _unsupported(selected, "power_base_base_gradient", base.dtype)
    return result


def execute_power_mixed_gradient(
    outer: Tensor,
    grad: Tensor,
    base: Tensor,
    exponent: Tensor,
    *,
    dtype: DataType,
) -> Storage:
    """Run ``d2(base ** exponent)/d(base)d(exponent)``, weighted by both.

    The dtype is the caller's because this one partial is two gradients: the
    exponent's when the base VJP is differentiated and the base's when the
    exponent VJP is. Rule G5 gives each the dtype of the operand it belongs
    to, and only the caller knows which one it is asking for.
    """
    selected = get_backend()
    if selected == "python":
        from tensors.backend.python.kernels.elementwise.power_mixed_gradient import (
            power_mixed_gradient as reference,
        )

        return reference(outer, grad, base, exponent, dtype=dtype)

    backend: Any = load_backend(selected)
    result = backend.power_mixed_gradient(outer, grad, base, exponent, dtype=dtype)
    if result is None:
        raise _unsupported(selected, "power_mixed_gradient", dtype)
    return result


def execute_power_exponent_exponent_gradient(
    outer: Tensor, grad: Tensor, base: Tensor, exponent: Tensor
) -> Storage:
    """Run ``d2(base ** exponent)/d(exponent)2``, weighted by both gradients."""
    selected = get_backend()
    if selected == "python":
        from tensors.backend.python.kernels.elementwise.power_exponent_exponent_gradient import (
            power_exponent_exponent_gradient as reference,
        )

        return reference(outer, grad, base, exponent)

    backend: Any = load_backend(selected)
    result = backend.power_exponent_exponent_gradient(outer, grad, base, exponent)
    if result is None:
        raise _unsupported(selected, "power_exponent_exponent_gradient", exponent.dtype)
    return result
