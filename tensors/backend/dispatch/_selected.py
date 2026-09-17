"""Execution on the selected backend, with no policy and no fallback.

`docs/backends.md`, *Execution requirements*, makes the backend selection an
execution requirement rather than a preference. An entry point built on this
helper consults no workload-size policy and never answers with another
backend's kernel: it runs where the selection says, or it reports that it
cannot.

The four arithmetic operations use this through
:mod:`tensors.backend.dispatch.arithmetic._execution`; their vector-Jacobian
products use it through the entry points named ``execute_vjp_*``, which exist
alongside the older entry points rather than replacing them. The older ones
still serve operations outside the arithmetic contract, which keep the
workload policy and the reference fallback until that contract is extended to
them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from tensors.backend.config import BackendOperationUnsupportedError, get_backend
from tensors.backend.loading import _backend_kernel

if TYPE_CHECKING:
    from tensors.backend.storage import Storage


def run_on_selected_backend(
    operation: str,
    reference: Callable[..., Any],
    *args: Any,
    detail: str = "",
    **kwargs: Any,
) -> Storage:
    """Run one operation on the selected backend, or say it cannot run there.

    ``reference`` is the Python implementation, which is what the Python
    selection asks for. It is never used to answer for another backend: a
    kernel that returns ``None`` has declined, and a decline under an explicit
    or resolved selection is an error rather than an invitation to compute the
    answer somewhere else.
    """
    selected = get_backend()
    if selected == "python":
        return reference(*args, **kwargs)

    result = _backend_kernel(operation)(*args, **kwargs)
    if result is None:
        raise BackendOperationUnsupportedError(
            f"The {selected} backend cannot execute {operation}"
            f"{f' {detail}' if detail else ''} conformingly. This computation "
            f"runs on the selected backend; select another backend to run it "
            f"elsewhere."
        )
    return result


__all__ = ["run_on_selected_backend"]
