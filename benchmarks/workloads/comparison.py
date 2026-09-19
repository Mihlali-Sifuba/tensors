"""Elementwise comparisons, which produce a mask rather than a value.

Separated from arithmetic because the output dtype is not the input dtype
and nothing downstream differentiates through it. That makes the interesting
question a different one: what does producing a boolean mask cost relative to
producing a float of the same length, and does a backend pay for the change
of width.

Integer inputs are carried because a comparison is one of the few places an
integer dtype reaches an accelerated path at all.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..case import Case, Group, Unsupported
from ..inputs import SIZE_CEILING, ceiling_for
from ..profiles import selected_sizes
from ._elementwise import comparison_cases

#: Three decades: the fixed cost, the crossover, and the streaming regime.
SIZES: tuple[int, ...] = (1, 10_000, 1_000_000)

#: The comparisons this domain owns. The remaining three are the negations
#: of these, and measure the same path.
OPERATIONS: tuple[str, ...] = ("equal", "less", "greater_equal")

#: Both float widths, and one integer width to show the integer path.
DTYPES: tuple[str, ...] = ("float64", "float32", "int64")


def groups() -> list[Group]:
    """Return one group per (operation, dtype, size)."""
    result: list[Group] = []
    for operation in OPERATIONS:
        for dtype_name in DTYPES:
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
                    return comparison_cases(backend, operation, dtype_name, size)

                result.append(
                    Group(
                        name=f"comparison/{operation}/{dtype_name}/{size}",
                        factory=factory,
                        suite="comparison",
                    )
                )
    return result


__all__ = ["groups"]
