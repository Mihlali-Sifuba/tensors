"""Tensor transpose public API."""

from typing import Any

from .dot import _transpose_impl


def transpose(value: Any) -> Any:
    """Transpose a 2D Tensor.

    Transpose has not yet gained a differentiable operation rule, so Variables
    are deliberately rejected instead of silently dropping their history.
    """
    from ..autograd.variable import Variable

    if isinstance(value, Variable):
        raise NotImplementedError("Differentiable transpose is not implemented")
    return _transpose_impl(value)


__all__ = ["transpose"]
