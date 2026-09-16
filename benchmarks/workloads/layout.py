"""Layout effects: materialization, gathering, and offsets.

Two separate facts are measured here.

First, what the public API does with layout. Every public shape operation
returns an independently owned, compact tensor, so ``reshape``, ``transpose``
and slicing all copy where a provider would hand back a view. Comparing them
against the provider's view operation prices that copy.

Second, what the kernels do when a layout is not compact. Non-contiguous
tensors are reachable only through the internal metadata constructor, so
they are built that way deliberately: the provider boundary gathers logical
values before crossing, and the gather is what a non-contiguous layout
actually costs.
"""

from __future__ import annotations
from collections.abc import Sequence
from typing import Any
import tensors as ts
import importlib
from benchmarks.case import Case, Group, Unsupported
from benchmarks.inputs import (
    ACCELERATED,
    provider_array,
    provider_module,
    tensor,
)

STRIDED_ELEMENT_CEILING = 70_000


def _view_tensor(
    source: ts.Tensor, shape: tuple[int, ...], strides: tuple[int, ...], offset: int = 0
) -> ts.Tensor:
    """Build a tensor whose layout is not compact, over existing storage.

    The public API never produces one, so the internal constructor is the
    only way to measure what the kernels do with a strided layout.
    """
    return ts.Tensor._from_metadata(
        source._storage, shape=shape, strides=strides, offset=offset
    )


def _materialization_cases(backend: str, side: int) -> list[Case]:
    """Price the copy each public shape operation performs."""
    shape = (side, side)
    elements = side * side
    common: dict[str, Any] = {
        "family": "layout/materialization",
        "dtype": "float64",
        "shape": shape,
        "elements": elements,
        "work_items": elements,
    }
    cases: list[Case] = []
    value = tensor(shape, dtype_name="float64", kind="ramp")
    operations = {
        "reshape": (
            lambda: ts.reshape(value, (elements,)),
            "reshape",
            lambda provider, raw: provider.reshape(raw, (elements,)),
        ),
        "transpose": (
            lambda: ts.transpose(value),
            "transpose",
            lambda provider, raw: provider.transpose(raw),
        ),
        "slice": (
            lambda: value[: side // 2, : side // 2],
            "slice",
            lambda provider, raw: raw[: side // 2, : side // 2],
        ),
        "contiguous": (
            lambda: value.contiguous(),
            "ascontiguousarray",
            lambda provider, raw: provider.ascontiguousarray(raw),
        ),
        "clone": (lambda: value.clone(), "copy", lambda provider, raw: raw.copy()),
    }
    if backend in ACCELERATED:
        provider = provider_module(backend)
        raw = provider_array(provider, shape, dtype_name="float64", kind="ramp")
        for name, (_, _, provider_call) in operations.items():
            tags = {
                "ladder": f"layout-{name}|{side}",
                "curve": f"layout-{name}",
                "pair": f"layout-{name}|{side}",
            }
            cases.append(
                Case(
                    name=f"provider.{name}/{side}x{side}",
                    run=lambda call=provider_call: call(provider, raw),
                    layer="provider",
                    validate=lambda call=provider_call: call(provider, raw),
                    description="the provider equivalent, which returns a view for reshape, transpose, and slicing",
                    backends=ACCELERATED,
                    tags=tags,
                    **common,
                )
            )
    for name, (public_call, _, _) in operations.items():
        cases.append(
            Case(
                name=f"public.{name}/{side}x{side}",
                run=public_call,
                layer="public",
                validate=public_call,
                description="public shape operation; always returns independently owned compact storage",
                tags={
                    "ladder": f"layout-{name}|{side}",
                    "curve": f"layout-{name}",
                    "pair": f"layout-{name}|{side}",
                },
                memory=True,
                **common,
            )
        )
    return cases


def _strided_cases(backend: str, side: int) -> list[Case]:
    """Measure kernels over layouts that are not compact."""
    if backend not in ACCELERATED:
        raise Unsupported(
            "the provider boundary that gathers a non-compact layout exists only for the NumPy and CUDA backends"
        )
    elements = side * side
    from tensors.backend.loading import load_backend

    kernels = load_backend(backend)
    _view = importlib.import_module(f"tensors.backend.{backend}.conversion")._view
    base = tensor((2 * side, 2 * side), dtype_name="float64", kind="ramp")
    layouts: dict[str, ts.Tensor] = {
        "contiguous": tensor((side, side), dtype_name="float64", kind="ramp"),
        "offset": _view_tensor(base, (side, side), (side, 1), offset=side),
        "transposed": _view_tensor(base, (side, side), (1, 2 * side)),
        "strided": _view_tensor(base, (side, side), (4 * side, 2)),
        "broadcast": _view_tensor(base, (side, side), (0, 1)),
    }
    cases: list[Case] = []
    for name, value in layouts.items():
        common: dict[str, Any] = {
            "family": "layout/strided",
            "dtype": "float64",
            "shape": (side, side),
            "elements": elements,
            "work_items": elements,
            "tags": {
                "layout": name,
                "pair": f"strided-view|{side}",
                "curve": f"strided-view-{name}",
            },
            "backends": ACCELERATED,
        }
        cases.append(
            Case(
                name=f"boundary.view/{name}/{side}x{side}",
                run=lambda value=value: _view(value),
                layer="kernel",
                validate=lambda value=value: _view(value),
                description="the provider boundary alone: gather logical values into a compact native array",
                **common,
            )
        )
        cases.append(
            Case(
                name=f"kernel.add/{name}/{side}x{side}",
                run=lambda value=value: kernels.add(
                    value, value, dtype=ts.float64, output_shape=(side, side)
                ),
                layer="kernel",
                validate=lambda value=value: kernels.add(
                    value, value, dtype=ts.float64, output_shape=(side, side)
                ),
                description="a kernel reading a tensor with this layout",
                **common,
            )
        )
        cases.append(
            Case(
                name=f"public.sum/{name}/{side}x{side}",
                run=lambda value=value: ts.sum(value),
                layer="public",
                validate=lambda value=value: ts.sum(value),
                description="a public reduction over a tensor with this layout",
                **common,
            )
        )
        cases.append(
            Case(
                name=f"public.contiguous/{name}/{side}x{side}",
                run=lambda value=value: value.contiguous(),
                layer="public",
                validate=lambda value=value: value.contiguous(),
                description="converting this layout to a contiguous tensor; a no-op when the layout is already contiguous",
                memory=True,
                **common,
            )
        )
    return cases


def _view_support_probe(backend: str) -> list[Case]:
    """Record that the public API offers no view-returning operation."""
    raise Unsupported(
        "the public API has no view-returning operation to benchmark: reshape, transpose, slicing, astype, and clone each return an independently owned compact Tensor, so view creation cannot be separated from materialization at the public layer. Non-compact layouts are measured through the internal metadata constructor in the layout/strided groups instead."
    )


def groups() -> list[Group]:
    """Return layout groups over materialization and strided access."""
    result: list[Group] = []
    for side in (8, 64, 256, 1_024):
        elements = side * side

        def materialization(
            backend: str, side: int = side, elements: int = elements
        ) -> Sequence[Case]:
            if backend == "python" and elements > 100_000:
                raise Unsupported(
                    f"{elements} elements exceeds the Python backend ceiling for materializing layouts"
                )
            return _materialization_cases(backend, side)

        result.append(
            Group(
                name=f"layout/materialization/{side}",
                factory=materialization,
                suite="layout",
            )
        )

        def strided(
            backend: str, side: int = side, elements: int = elements
        ) -> Sequence[Case]:
            if backend == "python" and elements > 100_000:
                raise Unsupported("exceeds the Python backend ceiling")
            if elements > STRIDED_ELEMENT_CEILING:
                raise Unsupported(
                    f"{elements} elements exceeds the {STRIDED_ELEMENT_CEILING} ceiling for non-compact layouts: gathering one is a Python-level loop over logical coordinates, measured at roughly 2.7 us per element, so a single sample here would take seconds. The 8/64/256 points establish the scaling instead."
                )
            return _strided_cases(backend, side)

        result.append(
            Group(name=f"layout/strided/{side}", factory=strided, suite="layout")
        )
    result.append(
        Group(
            name="layout/public-view-support",
            factory=_view_support_probe,
            suite="layout",
        )
    )
    return result


__all__ = ["groups"]
