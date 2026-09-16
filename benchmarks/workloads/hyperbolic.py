"""Hyperbolic functions.

``tanh`` is here rather than with the activations because that is
what it is; that a model reaches for it as an activation is a use,
not a definition."""

from __future__ import annotations

from collections.abc import Sequence

from ..case import Case, Group, Unsupported
from ..inputs import FLOAT_DTYPES, SIZE_CEILING, ceiling_for
from ..profiles import selected_sizes
from ._elementwise import unary_ladder

#: Sizes at which the per-element cost of a unary function is worth a
#: separate measurement: a scalar, where fixed overhead is the whole cost,
#: then four decades over which the arithmetic takes over.
SIZES: tuple[int, ...] = (1, 100, 10_000, 1_000_000, 10_000_000)

#: The operations this domain owns.
OPERATIONS: tuple[str, ...] = ("tanh",)


def groups() -> list[Group]:
    """Return one ladder per (operation, dtype, size)."""
    result: list[Group] = []
    for operation in OPERATIONS:
        for dtype_name in FLOAT_DTYPES:
            for size in selected_sizes(SIZES):

                def factory(
                    backend: str,
                    operation: str = operation,
                    dtype_name: str = dtype_name,
                    size: int = size,
                ) -> Sequence[Case]:
                    limit = ceiling_for(backend, SIZE_CEILING)
                    if size > limit:
                        raise Unsupported(
                            f"{size} elements exceeds the {backend} ceiling "
                            f"of {limit}"
                        )
                    return unary_ladder(backend, operation, dtype_name, size)

                result.append(
                    Group(
                        name=f"hyperbolic/{operation}/{dtype_name}/{size}",
                        factory=factory,
                        suite="hyperbolic",
                    )
                )
    return result


__all__ = ["groups"]
