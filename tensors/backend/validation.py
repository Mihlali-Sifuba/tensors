"""Backend-residency validation at numerical execution boundaries."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from tensors.backend.config import BackendMismatchError
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.backend.types import BackendName


def validate_operands(
    operands: Iterable[Any],
    expected_backend: BackendName,
    *,
    context: str,
) -> None:
    """Require Tensor and Storage operands to reside on the expected backend."""
    from tensors.tensor import Tensor

    for index, operand in enumerate(operands):
        if isinstance(operand, Tensor):
            resident = operand._storage.kind
        elif isinstance(operand, Storage):
            resident = operand.kind
        else:
            continue
        if resident != expected_backend:
            raise BackendMismatchError(
                f"The {context} operation received operand {index} on the "
                f"{resident} backend while the active backend is "
                f"{expected_backend}"
            )


def validate_result(
    storage: Storage,
    expected_backend: BackendName,
    *,
    context: str,
) -> Storage:
    """Require a numerical result to reside on the expected backend."""
    if storage.kind != expected_backend:
        raise BackendMismatchError(
            f"The {context} operation selected the {expected_backend} backend but "
            f"returned {storage.kind} storage"
        )
    return storage


__all__ = ["validate_operands", "validate_result"]
