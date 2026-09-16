"""Random generation, seeded and unseeded.

Seeding is what decides whether generation can use the provider at all: a
seeded stream has to reproduce the reference generator's values, so it is
worth measuring both states rather than only the default.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import tensors as ts

from ..case import Case, Group, Unsupported
from ..workloads import (
    ACCELERATED,
    FLOAT_DTYPES,
    dtype_of,
    provider_module,
)


def _random_cases(
    backend: str,
    size: int,
    dtype_name: str,
    seeded: bool,
) -> list[Case]:
    """Build provider and public generation cases for one configuration."""
    dtype = dtype_of(dtype_name)
    shape = (size,)
    state = "seeded" if seeded else "unseeded"
    cases: list[Case] = []

    def common(name: str) -> dict[str, Any]:
        return {
            "family": f"random/{name}",
            "dtype": dtype_name,
            "shape": shape,
            "elements": size,
            "work_items": size,
            "tags": {
                "ladder": f"random-{name}|{dtype_name}|{size}|{state}",
                "curve": f"random-{name}|{dtype_name}|{state}",
                "dtype_pair": f"random-{name}|{state}",
                "pair": f"random-{name}|{dtype_name}|{size}",
                "seeded": state,
            },
        }

    def seed() -> None:
        ts.random.seed(12345 if seeded else None)

    if backend in ACCELERATED:
        provider = provider_module(backend)
        native = getattr(provider, dtype_name)
        provider_calls = {
            "uniform": lambda: provider.random.random(size).astype(native),
            "normal": lambda: provider.random.standard_normal(
                size
            ).astype(native),
        }
        for name, call in provider_calls.items():
            cases.append(Case(
                name=f"provider.{name}/{dtype_name}/{size}/{state}",
                run=call,
                layer="provider",
                validate=call,
                description=f"raw provider {name} generation",
                backends=ACCELERATED,
                **common(name),
            ))
        cases.append(Case(
            name=f"provider.randint/int64/{size}/{state}",
            run=lambda: provider.random.randint(0, 100, size),
            layer="provider",
            validate=lambda: provider.random.randint(0, 100, size),
            description="raw provider integer generation",
            family="random/randint",
            dtype="int64",
            shape=shape,
            elements=size,
            work_items=size,
            backends=ACCELERATED,
            tags={
                "ladder": f"random-randint|int64|{size}|{state}",
                "curve": f"random-randint|int64|{state}",
            },
        ))

    public_calls = {
        "uniform": lambda: ts.random.uniform(shape, dtype=dtype),
        "normal": lambda: ts.random.normal(shape, dtype=dtype),
    }
    for name, call in public_calls.items():
        cases.append(Case(
            name=f"public.{name}/{dtype_name}/{size}/{state}",
            run=call,
            layer="public",
            validate=call,
            setup=seed,
            description=f"public {name} generation ({state})",
            memory=name == "normal",
            **common(name),
        ))
    cases.append(Case(
        name=f"public.randint/int64/{size}/{state}",
        run=lambda: ts.random.randint(shape, 0, 100),
        layer="public",
        validate=lambda: ts.random.randint(shape, 0, 100),
        setup=seed,
        description=f"public integer generation ({state})",
        family="random/randint",
        dtype="int64",
        shape=shape,
        elements=size,
        work_items=size,
        tags={
            "ladder": f"random-randint|int64|{size}|{state}",
            "curve": f"random-randint|int64|{state}",
        },
    ))
    return cases


def groups() -> list[Group]:
    """Return generation groups over sizes, dtypes, and seeding."""
    result: list[Group] = []
    for seeded in (False, True):
        for dtype_name in FLOAT_DTYPES:
            for size in (1, 1_000, 100_000, 1_000_000):
                def factory(
                    backend: str,
                    size: int = size,
                    dtype_name: str = dtype_name,
                    seeded: bool = seeded,
                ) -> Sequence[Case]:
                    if backend == "python" and size > 100_000:
                        raise Unsupported(
                            "exceeds the Python backend generation ceiling"
                        )
                    return _random_cases(backend, size, dtype_name, seeded)

                state = "seeded" if seeded else "unseeded"
                result.append(Group(
                    name=f"random/{state}/{dtype_name}/{size}",
                    factory=factory,
                    suite="random",
                ))
    return result


__all__ = ["groups"]
