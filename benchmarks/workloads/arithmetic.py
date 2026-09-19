"""Addition, subtraction, multiplication, and division.

The cheapest real work the library does, which is what makes it the best
place to see everything that is not the work. A ladder over one of these
separates the provider call from the guarded kernel, the kernel from
dispatch, dispatch from the public operation, and the operation from what a
Variable and a replayed graph add on top.

Integer dtypes are carried alongside the float ones because they take a
different path: nothing accelerates integer elementwise arithmetic, so the
cost is per-element Python work on every backend, and the ladder shows that
rather than implying an acceleration that is not there.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..case import Case, Group, Unsupported
from ..inputs import FLOAT_DTYPES, INTEGER_DTYPES, SIZE_CEILING, ceiling_for, is_integer
from ..profiles import selected_sizes
from ._elementwise import INTEGER_ELEMENTWISE_CEILING, binary_ladder

#: Eight decades, because this is the curve every other curve is read
#: against: where a backend starts to pay, and where it stops improving.
SIZES: tuple[int, ...] = (
    1,
    10,
    100,
    1_000,
    10_000,
    100_000,
    1_000_000,
    10_000_000,
)

#: The operations this domain owns.
OPERATIONS: tuple[str, ...] = ("add", "subtract", "multiply", "divide")


def groups() -> list[Group]:
    """Return one ladder per (operation, dtype, size)."""
    result: list[Group] = []
    for operation in OPERATIONS:
        for dtype_name in (*FLOAT_DTYPES, *INTEGER_DTYPES):
            # Integer division promotes to float, so it is not the integer
            # operation it appears to be and belongs with the float sizes.
            if operation == "divide" and is_integer(dtype_name):
                continue
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
                            f"{size} elements exceeds the {backend} ceiling of "
                            f"{limit}; the Python backend interprets element by "
                            "element and the device has limited memory"
                        )
                    if is_integer(dtype_name) and size > INTEGER_ELEMENTWISE_CEILING:
                        raise Unsupported(
                            f"{size} integer elements exceeds the "
                            f"{INTEGER_ELEMENTWISE_CEILING} ceiling for integer "
                            "elementwise work: no backend accelerates it (NumPy "
                            "computes in object dtype, CUDA declines and uses "
                            "the host reference implementation), so cost is "
                            "per-element Python work in every case"
                        )
                    return binary_ladder(backend, operation, dtype_name, size)

                result.append(
                    Group(
                        name=f"arithmetic/{operation}/{dtype_name}/{size}",
                        factory=factory,
                        suite="arithmetic",
                    )
                )
    return result


__all__ = ["groups"]
