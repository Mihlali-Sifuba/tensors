"""Shape manipulation, indexing, selection, and dtype conversion.

These operations move values rather than compute with them, so their cost is
almost entirely the library's: metadata work, validation, and the copy each
one performs. That makes them a clean read on fixed overhead.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import tensors as ts
from tensors.backend import (
    execute_cast,
    execute_concat,
    execute_slice,
    execute_stack,
    execute_transpose,
    execute_where,
)

from ..case import Case, Group, Unsupported
from ..inputs import (
    ACCELERATED,
    dtype_of,
    kernel_module,
    provider_array,
    provider_module,
    tensor,
)


def _manipulation_cases(backend: str, side: int) -> list[Case]:
    """Build transpose, concat, stack, cast, and selection cases."""
    shape = (side, side)
    elements = side * side
    cases: list[Case] = []
    value = tensor(shape, dtype_name="float64", kind="ramp")
    second = tensor(shape, dtype_name="float64", kind="constant", value=2.0)
    condition = ts.greater(value, ts.full(shape, 1.5))

    def common(name: str, work: int, output: tuple[int, ...]) -> dict[str, Any]:
        return {
            "family": f"shape/{name}",
            "dtype": "float64",
            "shape": output,
            "elements": elements,
            "work_items": work,
            "tags": {
                "ladder": f"{name}|float64|{elements}",
                "curve": f"shape-{name}",
            },
        }

    if backend in ACCELERATED:
        provider = provider_module(backend)
        kernels = kernel_module(backend)
        raw = provider_array(provider, shape, dtype_name="float64", kind="ramp")
        raw_second = provider_array(
            provider, shape, dtype_name="float64", kind="constant", value=2.0
        )
        raw_condition = raw > 1.5

        provider_calls: dict[str, tuple[Any, int, tuple[int, ...]]] = {
            "transpose": (
                lambda: provider.transpose(raw).copy(),
                elements,
                shape,
            ),
            "concat": (
                lambda: provider.concatenate((raw, raw_second), axis=0),
                2 * elements,
                (2 * side, side),
            ),
            "stack": (
                lambda: provider.stack((raw, raw_second), axis=0),
                2 * elements,
                (2, side, side),
            ),
            "cast": (
                lambda: raw.astype(provider.float32),
                elements,
                shape,
            ),
            "where": (
                lambda: provider.where(raw_condition, raw, raw_second),
                elements,
                shape,
            ),
            "maximum": (
                lambda: provider.maximum(raw, raw_second),
                elements,
                shape,
            ),
            "clip": (
                lambda: provider.clip(raw, 1.2, 1.8),
                elements,
                shape,
            ),
            "slice": (
                lambda: raw[: side // 2, : side // 2].copy(),
                elements // 4,
                (side // 2, side // 2),
            ),
        }
        for name, (call, work, output) in provider_calls.items():
            cases.append(
                Case(
                    name=f"provider.{name}/{side}x{side}",
                    run=call,
                    layer="provider",
                    validate=call,
                    description=f"raw provider {name}",
                    backends=ACCELERATED,
                    **common(name, work, output),
                )
            )

        kernel_calls: dict[str, tuple[Any, int, tuple[int, ...]]] = {
            "transpose": (
                lambda: kernels.transpose(value, (1, 0), output_shape=shape),
                elements,
                shape,
            ),
            "concat": (
                lambda: kernels.concat(
                    (value, second),
                    axis=0,
                    dtype=ts.float64,
                    output_shape=(2 * side, side),
                ),
                2 * elements,
                (2 * side, side),
            ),
            "stack": (
                lambda: kernels.stack(
                    (value, second),
                    axis=0,
                    dtype=ts.float64,
                    output_shape=(2, side, side),
                ),
                2 * elements,
                (2, side, side),
            ),
            "cast": (
                lambda: kernels.cast_tensor(value, dtype=ts.float32),
                elements,
                shape,
            ),
            "where": (
                lambda: kernels.where(
                    condition,
                    value,
                    second,
                    dtype=ts.float64,
                    output_shape=shape,
                ),
                elements,
                shape,
            ),
            "maximum": (
                lambda: kernels.maximum(
                    value,
                    second,
                    dtype=ts.float64,
                    output_shape=shape,
                ),
                elements,
                shape,
            ),
            "clip": (
                lambda: kernels.clip(value, 1.2, 1.8, dtype=ts.float64),
                elements,
                shape,
            ),
            "slice": (
                lambda: kernels.slice_tensor(
                    value,
                    (slice(0, side // 2), slice(0, side // 2)),
                    output_shape=(side // 2, side // 2),
                ),
                elements // 4,
                (side // 2, side // 2),
            ),
        }
        for name, (call, work, output) in kernel_calls.items():

            def validate(call: Any = call, name: str = name) -> None:
                if call() is None:
                    raise Unsupported(
                        f"the {name} kernel declines this configuration and "
                        "defers to the Python reference implementation"
                    )

            cases.append(
                Case(
                    name=f"kernel.{name}/{side}x{side}",
                    run=call,
                    layer="kernel",
                    validate=validate,
                    description=f"internal {name} kernel",
                    backends=ACCELERATED,
                    **common(name, work, output),
                )
            )

        dispatch_calls: dict[str, tuple[Any, int, tuple[int, ...]]] = {
            "transpose": (
                lambda: execute_transpose(value, (1, 0), output_shape=shape),
                elements,
                shape,
            ),
            "cast": (
                lambda: execute_cast(value, dtype=ts.float32),
                elements,
                shape,
            ),
            "where": (
                lambda: execute_where(
                    condition,
                    value,
                    second,
                    dtype=ts.float64,
                    output_shape=shape,
                ),
                elements,
                shape,
            ),
            "slice": (
                lambda: execute_slice(
                    value,
                    (slice(0, side // 2), slice(0, side // 2)),
                    output_shape=(side // 2, side // 2),
                ),
                elements // 4,
                (side // 2, side // 2),
            ),
            "concat": (
                lambda: execute_concat(
                    (value, second),
                    axis=0,
                    dtype=ts.float64,
                    output_shape=(2 * side, side),
                ),
                2 * elements,
                (2 * side, side),
            ),
            "stack": (
                lambda: execute_stack(
                    (value, second),
                    axis=0,
                    dtype=ts.float64,
                    output_shape=(2, side, side),
                ),
                2 * elements,
                (2, side, side),
            ),
        }
        for name, (call, work, output) in dispatch_calls.items():
            cases.append(
                Case(
                    name=f"dispatch.{name}/{side}x{side}",
                    run=call,
                    layer="dispatch",
                    validate=call,
                    description=f"execute_{name}: policy and kernel lookup",
                    backends=ACCELERATED,
                    **common(name, work, output),
                )
            )

    public_calls: dict[str, tuple[Any, int, tuple[int, ...]]] = {
        "transpose": (lambda: ts.transpose(value), elements, shape),
        "reshape": (
            lambda: ts.reshape(value, (elements,)),
            elements,
            (elements,),
        ),
        "concat": (
            lambda: ts.concat((value, second), axis=0),
            2 * elements,
            (2 * side, side),
        ),
        "stack": (
            lambda: ts.stack((value, second), axis=0),
            2 * elements,
            (2, side, side),
        ),
        "cast": (lambda: value.astype(ts.float32), elements, shape),
        "where": (
            lambda: ts.where(condition, value, second),
            elements,
            shape,
        ),
        "maximum": (lambda: ts.maximum(value, second), elements, shape),
        "clip": (lambda: ts.clip(value, 1.2, 1.8), elements, shape),
        "slice": (
            lambda: value[: side // 2, : side // 2],
            elements // 4,
            (side // 2, side // 2),
        ),
        "row-index": (lambda: value[0], side, (side,)),
        "scalar-index": (lambda: value[0, 0], 1, ()),
    }
    for name, (call, work, output) in public_calls.items():
        cases.append(
            Case(
                name=f"public.{name}/{side}x{side}",
                run=call,
                layer="public",
                validate=call,
                description=f"public {name}",
                memory=name in ("transpose", "reshape", "slice"),
                **common(name, work, output),
            )
        )
    return cases


def groups() -> list[Group]:
    """Return shape and indexing groups over a size curve."""
    result: list[Group] = []
    for side in (4, 32, 256, 1_024, 2_048):
        elements = side * side

        def factory(
            backend: str,
            side: int = side,
            elements: int = elements,
        ) -> Sequence[Case]:
            if backend == "python" and elements > 100_000:
                raise Unsupported(
                    f"{elements} elements exceeds the Python backend "
                    "ceiling for shape manipulation"
                )
            if backend == "cuda" and elements > 4_500_000:
                raise Unsupported(
                    "concat and stack build multi-copy temporaries that "
                    "exceed the device memory available in this run"
                )
            return _manipulation_cases(backend, side)

        result.append(
            Group(
                name=f"shape/manipulation/{side}",
                factory=factory,
                suite="shape",
            )
        )
    return result


__all__ = ["groups"]
