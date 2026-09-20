"""Backend-residency validation at numerical execution boundaries."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from tensors.backend import config
from tensors.backend.config import BackendMismatchError

if TYPE_CHECKING:
    from tensors.backend.storage import Storage


def validate_operands(operation: str, operands: Iterable[Any]) -> str:
    """Require every Tensor operand to reside on the active backend."""
    from tensors.tensor import Tensor

    active = config.get_backend()
    for index, operand in enumerate(operands):
        if not isinstance(operand, Tensor):
            continue
        resident = operand._storage.kind
        if resident != active:
            raise BackendMismatchError(
                f"The {operation} operation received operand {index} on the "
                f"{resident} backend while the active backend is {active}"
            )
    return active


def validate_result(operation: str, storage: Storage, active: str) -> Storage:
    """Require a numerical result to remain native to its active backend."""
    if storage.kind != active:
        raise RuntimeError(
            f"The {operation} operation selected the {active} backend but "
            f"returned {storage.kind} storage"
        )
    return storage


__all__ = ["validate_operands", "validate_result"]
