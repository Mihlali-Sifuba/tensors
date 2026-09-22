"""Dispatch for shape, layout, indexing, and representation changes."""

from __future__ import annotations
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any
from tensors.backend.config import BackendOperationUnsupportedError
from tensors.backend.loading import load_backend
from tensors.backend.storage import Storage
from tensors.backend.validation import validate_backend_residency

if TYPE_CHECKING:
    from tensors._typing import Scalar


def execute_assign_indices(
    destination: Storage,
    indices: Sequence[int],
    values: Storage | Scalar,
    *,
    source_indices: Sequence[int] | None = None,
) -> None:
    """Write values into a tensor's own storage, where that storage lives.

    An in-place write is the one boundary that answers to the tensor rather
    than to the selection. Every other dispatcher runs where the active
    backend says because it is producing a new value; this one is changing a
    value that already exists, and that value has a backend of its own. A
    tensor keeps its backend for its whole life, so the write goes to it.
    Dispatching on the selection instead would either migrate the tensor or
    refuse the write, and `docs/backend-storage-architecture.md` section 5.4
    requires neither: mutation updates the tensor's own storage, on its own
    backend.

    A backend that cannot write conformingly says so rather than letting
    another one write in its place, which is the execution contract of
    `docs/backends.md` applied to mutation.

    Nothing is calculated here. The positions are resolved and the values
    converted by the caller, and each backend stores the same values at the
    same positions, so no numerical decision depends on where this runs.
    Values carrying no backend — a host scalar, already converted to the
    destination's dtype — are one value for every position, and that single
    transfer is the host-facing write section 5.4 allows.
    """
    resident = destination.kind
    validate_backend_residency((destination, values), resident)
    backend: Any = load_backend(resident)
    written = backend.assign_indices(
        destination, indices, values, source_indices=source_indices
    )
    if written is None:
        raise BackendOperationUnsupportedError(
            f"The {resident} backend cannot execute assign_indices at dtype "
            f"{destination.dtype.name} conformingly. This write happens on the "
            f"backend the tensor lives on; no other backend performs it."
        )
