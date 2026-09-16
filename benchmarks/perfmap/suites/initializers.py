"""Parameter initializers across fan shapes and dtypes.

Initializers run once per parameter at model construction, so their cost
matters for startup rather than for a training step. Orthogonal
initialization is the exception: it performs a decomposition, so it scales
very differently from the sampling initializers and is measured separately.
"""

from __future__ import annotations

from collections.abc import Sequence

import tensors as ts

from ..harness import Case, Group, Unsupported
from ..workloads import FLOAT_DTYPES, dtype_of

#: The sampling initializers, which differ only in their variance scale.
SAMPLING = (
    "he_normal",
    "he_uniform",
    "xavier_normal",
    "xavier_uniform",
    "lecun_normal",
    "lecun_uniform",
    "truncated_normal",
)


def _initializer_cases(
    backend: str,
    shape: tuple[int, ...],
    dtype_name: str,
) -> list[Case]:
    """Build one case per initializer for one parameter shape."""
    dtype = dtype_of(dtype_name)
    elements = 1
    for dimension in shape:
        elements *= dimension
    cases: list[Case] = []

    # Every sampling initializer scales its variance by the fan of the shape,
    # which a one-dimensional parameter does not have.
    if len(shape) < 2:
        raise Unsupported(
            "fan-scaled initializers need a shape with at least two dimensions"
        )

    for name in SAMPLING:
        initializer = getattr(ts.init, name)
        cases.append(
            Case(
                name=f"public.{name}/{dtype_name}/"
                f"{'x'.join(str(item) for item in shape)}",
                run=lambda initializer=initializer: initializer(shape, dtype=dtype),
                layer="public",
                validate=lambda initializer=initializer: initializer(
                    shape, dtype=dtype
                ),
                description=f"{name} parameter initialization",
                family=f"init/{name}",
                dtype=dtype_name,
                shape=shape,
                elements=elements,
                work_items=elements,
                tags={
                    "ladder": f"init-{name}|{dtype_name}|{elements}",
                    "curve": f"init-{name}|{dtype_name}",
                    "dtype_pair": f"init-{name}",
                },
            )
        )

    # Orthogonal initialization decomposes a matrix, so it only applies to
    # rank-2 parameters and scales with the decomposition rather than the
    # element count.
    if len(shape) == 2:
        cases.append(
            Case(
                name=f"public.orthogonal/{dtype_name}/{shape[0]}x{shape[1]}",
                run=lambda: ts.init.orthogonal(shape, dtype=dtype),
                layer="public",
                validate=lambda: ts.init.orthogonal(shape, dtype=dtype),
                description="orthogonal initialization through a decomposition",
                family="init/orthogonal",
                dtype=dtype_name,
                shape=shape,
                elements=elements,
                work_items=elements,
                tags={
                    "ladder": f"init-orthogonal|{dtype_name}|{elements}",
                    "curve": f"init-orthogonal|{dtype_name}",
                    "dtype_pair": "init-orthogonal",
                },
            )
        )
    return cases


def groups() -> list[Group]:
    """Return initializer groups over representative parameter shapes."""
    result: list[Group] = []
    shapes: tuple[tuple[int, ...], ...] = (
        (16, 16),
        (128, 128),
        (784, 256),
        (1_024, 1_024),
        (64,),
        (32, 3, 3, 3),
    )
    for dtype_name in FLOAT_DTYPES:
        for shape in shapes:
            elements = 1
            for dimension in shape:
                elements *= dimension

            def factory(
                backend: str,
                shape: tuple[int, ...] = shape,
                dtype_name: str = dtype_name,
                elements: int = elements,
            ) -> Sequence[Case]:
                if backend == "python" and elements > 65_536:
                    raise Unsupported(
                        "exceeds the Python backend initialization ceiling"
                    )
                return _initializer_cases(backend, shape, dtype_name)

            result.append(
                Group(
                    name=(
                        f"initializers/{dtype_name}/"
                        f"{'x'.join(str(item) for item in shape)}"
                    ),
                    factory=factory,
                    suite="initializers",
                )
            )
    return result


__all__ = ["groups"]
