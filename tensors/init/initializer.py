"""Base contract for reusable parameter initializer configurations."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..tensor import Tensor
from ._utils import Shape


class Initializer(ABC):
    """Immutable callable configuration that constructs parameter tensors."""

    @abstractmethod
    def __call__(self, shape: Shape) -> Tensor:
        """Create one initialized tensor with the configured parameters."""


__all__ = ["Initializer"]
