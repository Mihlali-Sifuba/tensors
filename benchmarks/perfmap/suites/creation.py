"""Creation operations: values built from parameters rather than inputs.

A creation operation has no input tensor, so its whole cost is the library's
plus the allocation. That makes it the cleanest measurement of construction
and storage-wrapping overhead, and the size curve separates the per-call
part from the per-element part.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import tensors as ts
from tensors.backend import (
    execute_arange,
    execute_eye,
    execute_full,
    execute_linspace,
)

from ..harness import Case, Group, Unsupported
from ..workloads import (
    ACCELERATED,
    FLOAT_DTYPES,
    INTEGER_DTYPES,
    dtype_of,
    kernel_module,
    provider_module,
)

#: Integer creation is per-element Python work on every backend, so its
#: curve stops where a single call would take seconds.
INTEGER_CREATION_CEILING = 1_000_000


def _creation_cases(
    backend: str,
    size: int,
    dtype_name: str,
) -> list[Case]:
    """Build provider, kernel, dispatch, and public creation rungs."""
    dtype = dtype_of(dtype_name)
    shape = (size,)
    cases: list[Case] = []

    def common(name: str) -> dict[str, Any]:
        return {
            "family": f"creation/{name}",
            "dtype": dtype_name,
            "shape": shape,
            "elements": size,
            "work_items": size,
            "tags": {
                "ladder": f"{name}|{dtype_name}|{size}",
                "curve": f"creation-{name}|{dtype_name}",
                "dtype_pair": f"creation-{name}",
            },
        }

    if backend in ACCELERATED:
        provider = provider_module(backend)
        kernels = kernel_module(backend)
        native = getattr(provider, dtype_name)

        provider_calls = {
            "zeros": lambda: provider.zeros(size, dtype=native),
            "ones": lambda: provider.ones(size, dtype=native),
            "full": lambda: provider.full(size, 3, dtype=native),
            "arange": lambda: provider.arange(size, dtype=native),
            "linspace": lambda: provider.linspace(0.0, 1.0, size, dtype=native),
        }
        for name, call in provider_calls.items():
            if name == "linspace" and dtype.kind == "integer":
                continue
            cases.append(
                Case(
                    name=f"provider.{name}/{dtype_name}/{size}",
                    run=call,
                    layer="provider",
                    validate=call,
                    description=f"raw provider {name}",
                    backends=ACCELERATED,
                    **common(name),
                )
            )

        kernel_calls = {
            "full": lambda: kernels.full(shape, 3, dtype=dtype),
            "arange": lambda: kernels.arange(0, 1, size, dtype=dtype),
        }
        if dtype.kind != "integer":
            kernel_calls["linspace"] = lambda: kernels.linspace(
                0.0, 1.0, size, dtype=dtype
            )
        for name, call in kernel_calls.items():

            def validate(call: Any = call, name: str = name) -> None:
                if call() is None:
                    raise Unsupported(
                        f"the {name} kernel declines this dtype and defers "
                        "to the Python reference implementation"
                    )

            cases.append(
                Case(
                    name=f"kernel.{name}/{dtype_name}/{size}",
                    run=call,
                    layer="kernel",
                    validate=validate,
                    description=f"internal {name} kernel",
                    backends=ACCELERATED,
                    **common(name),
                )
            )

        dispatch_calls = {
            "full": lambda: execute_full(shape, 3, dtype=dtype),
            "arange": lambda: execute_arange(0, 1, size, dtype=dtype),
        }
        if dtype.kind != "integer":
            dispatch_calls["linspace"] = lambda: execute_linspace(
                0.0, 1.0, size, dtype=dtype
            )
        for name, call in dispatch_calls.items():
            cases.append(
                Case(
                    name=f"dispatch.{name}/{dtype_name}/{size}",
                    run=call,
                    layer="dispatch",
                    validate=call,
                    description=f"execute_{name}: policy and kernel lookup",
                    backends=ACCELERATED,
                    **common(name),
                )
            )

    public_calls = {
        "zeros": lambda: ts.zeros(shape, dtype=dtype),
        "ones": lambda: ts.ones(shape, dtype=dtype),
        "full": lambda: ts.full(shape, 3, dtype=dtype),
        "arange": lambda: ts.arange(0, size, 1, dtype=dtype),
    }
    if dtype.kind != "integer":
        public_calls["linspace"] = lambda: ts.linspace(0.0, 1.0, size, dtype=dtype)
    for name, call in public_calls.items():
        cases.append(
            Case(
                name=f"public.{name}/{dtype_name}/{size}",
                run=call,
                layer="public",
                validate=call,
                description=f"public {name}",
                memory=name == "zeros",
                **common(name),
            )
        )
    return cases


def _eye_cases(backend: str, side: int) -> list[Case]:
    """Build identity-matrix cases, which are quadratic in their argument."""
    elements = side * side
    common: dict[str, Any] = {
        "family": "creation/eye",
        "dtype": "float64",
        "shape": (side, side),
        "elements": elements,
        "work_items": elements,
        "tags": {
            "ladder": f"eye|float64|{elements}",
            "curve": "creation-eye|float64",
        },
    }
    cases: list[Case] = []
    if backend in ACCELERATED:
        provider = provider_module(backend)
        kernels = kernel_module(backend)
        cases.append(
            Case(
                name=f"provider.eye/{side}",
                run=lambda: provider.eye(side, dtype=provider.float64),
                layer="provider",
                validate=lambda: provider.eye(side, dtype=provider.float64),
                description="raw provider identity matrix",
                backends=ACCELERATED,
                **common,
            )
        )

        def run_kernel() -> Any:
            return kernels.eye(side, side, 0, dtype=ts.float64)

        def validate_kernel() -> None:
            if run_kernel() is None:
                raise Unsupported("the eye kernel declines this dtype")

        cases.append(
            Case(
                name=f"kernel.eye/{side}",
                run=run_kernel,
                layer="kernel",
                validate=validate_kernel,
                description="internal identity kernel",
                backends=ACCELERATED,
                **common,
            )
        )
        cases.append(
            Case(
                name=f"dispatch.eye/{side}",
                run=lambda: execute_eye(side, side, 0, dtype=ts.float64),
                layer="dispatch",
                validate=lambda: execute_eye(side, side, 0, dtype=ts.float64),
                description="execute_eye",
                backends=ACCELERATED,
                **common,
            )
        )
    cases.append(
        Case(
            name=f"public.eye/{side}",
            run=lambda: ts.eye(side),
            layer="public",
            validate=lambda: ts.eye(side),
            description="public identity matrix",
            **common,
        )
    )
    return cases


def groups() -> list[Group]:
    """Return creation groups over a size curve and both dtype families."""
    result: list[Group] = []
    for dtype_name in (*FLOAT_DTYPES, *INTEGER_DTYPES):
        for size in (1, 100, 10_000, 1_000_000, 10_000_000):

            def factory(
                backend: str,
                size: int = size,
                dtype_name: str = dtype_name,
            ) -> Sequence[Case]:
                if backend == "python" and size > 100_000:
                    raise Unsupported("exceeds the Python backend creation ceiling")
                if backend == "cuda" and size > 10_000_000:
                    raise Unsupported("exceeds available device memory")
                # Integer creation is per-element Python work on every
                # backend. The NumPy kernels build the values in `object`
                # dtype (`numpy.full(..., dtype=object)`), so a ten-million
                # element integer array is ten million Python ints — seconds
                # per call and hundreds of megabytes of temporaries. The
                # CUDA kernels decline integers outright and fall back to the
                # host reference implementation. Both are measured up to a
                # million values, which establishes the behavior; sweeping
                # further would cost tens of minutes and add no finding.
                if (
                    dtype_of(dtype_name).kind == "integer"
                    and size > INTEGER_CREATION_CEILING
                ):
                    raise Unsupported(
                        f"integer creation of {size} values exceeds the "
                        f"{INTEGER_CREATION_CEILING} ceiling: the NumPy "
                        "kernels build integer values in object dtype and "
                        "the CUDA kernels decline them entirely, so both "
                        "construct the buffer element by element in Python"
                    )
                return _creation_cases(backend, size, dtype_name)

            result.append(
                Group(
                    name=f"creation/values/{dtype_name}/{size}",
                    factory=factory,
                    suite="creation",
                )
            )

    for side in (4, 64, 512, 2_048):

        def eye_factory(backend: str, side: int = side) -> Sequence[Case]:
            if backend == "python" and side > 256:
                raise Unsupported(
                    "exceeds the Python backend ceiling for identity " "construction"
                )
            return _eye_cases(backend, side)

        result.append(
            Group(
                name=f"creation/eye/{side}",
                factory=eye_factory,
                suite="creation",
            )
        )
    return result


__all__ = ["groups"]
