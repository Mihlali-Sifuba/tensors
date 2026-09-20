"""Backend-residency validation at numerical execution boundaries."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from tensors.backend.config import BackendMismatchError
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.backend.types import BackendName


def validate_backend_residency(
    values: Iterable[Any],
    expected_backend: BackendName,
) -> None:
    """Require every backend-resident value to live on the expected backend.

    One invariant, stated once and independently of direction: the operands
    entering a selected-backend execution boundary and the storage coming back
    out are all values that must already reside where the selection says. A
    result is a backend-resident value like any other, so it is checked here
    rather than by a second validator that would restate the same rule.

    The caller selects the backend and passes it in. This never selects one,
    never moves or converts storage, never invokes a kernel, and never looks
    at a dtype, a shape or an element.

    A value carrying no residency — a Python scalar, a shape tuple, a dtype, an
    axis, a fill value — belongs to no backend and is skipped rather than
    rejected. Residency is decided by the repository's own types, never by
    probing for an attribute, so an unrelated object holding a ``kind`` is not
    mistaken for backend-resident data.

    Args:
        values: The values crossing the boundary, in order. The position of a
            failing value in this iterable is what the error reports.
        expected_backend: The backend the caller has already selected.

    Raises:
        BackendMismatchError: If a Tensor or Storage value resides elsewhere.
    """
    from tensors.tensor import Tensor

    for index, value in enumerate(values):
        if isinstance(value, Tensor):
            resident = value._storage.kind
        elif isinstance(value, Storage):
            resident = value.kind
        else:
            continue
        if resident != expected_backend:
            raise BackendMismatchError(
                f"Value {index} resides on the {resident} backend, but the "
                f"expected backend is {expected_backend}"
            )


__all__ = ["validate_backend_residency"]
