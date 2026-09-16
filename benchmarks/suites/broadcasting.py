"""Broadcasting shapes, and the broadcast reduction gradients use.

Broadcasting appears twice in the package. Forwards, an operation over
mismatched shapes has to agree a result shape and expand its operands.
Backwards, a gradient that flowed into a broadcast operand has to be summed
back down to that operand's shape, which is what ``sum_to_shape`` and
``sum_products_to_shape`` do. Both directions are measured against the
provider operation that performs the same expansion or reduction.
"""

from __future__ import annotations
from collections.abc import Sequence
from typing import Any
import tensors as ts
from tensors.backend.loading import load_backend
from tensors.backend import (
    execute_multiply,
    execute_sum_products_to_shape,
    execute_sum_to_shape,
)
from benchmarks.harness import Case, Group, Unsupported
from benchmarks.workloads import (
    ACCELERATED,
    provider_array,
    provider_module,
    tensor,
)

PATTERNS: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {
    "scalar-like": ((1_000_000,), ()),
    "vector-with-one": ((1_000_000,), (1,)),
    "matrix-with-row": ((1_024, 1_024), (1_024,)),
    "matrix-with-column": ((1_024, 1_024), (1_024, 1)),
    "matrix-with-scalar": ((1_024, 1_024), (1,)),
    "rank3-with-matrix": ((64, 128, 128), (128, 128)),
    "rank3-with-vector": ((64, 128, 128), (128,)),
    "rank4-middle": ((16, 32, 64, 64), (1, 32, 1, 64)),
    "equal-shapes": ((1_024, 1_024), (1_024, 1_024)),
}


def _broadcast_shape(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    """Return the result shape of broadcasting two shapes together."""
    rank = max(len(left), len(right))
    padded_left = (1,) * (rank - len(left)) + left
    padded_right = (1,) * (rank - len(right)) + right
    return tuple((max(one, other) for one, other in zip(padded_left, padded_right)))


def _elements(shape: tuple[int, ...]) -> int:
    total = 1
    for dimension in shape:
        total *= dimension
    return total


def _forward_cases(
    backend: str,
    pattern: str,
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
) -> list[Case]:
    """Measure a broadcasting multiply at provider, kernel, and public."""
    output_shape = _broadcast_shape(left_shape, right_shape)
    elements = _elements(output_shape)
    common: dict[str, Any] = {
        "family": "broadcast/forward",
        "dtype": "float64",
        "shape": output_shape,
        "elements": elements,
        "work_items": elements,
        "tags": {
            "ladder": f"broadcast-{pattern}|{elements}",
            "curve": f"broadcast-{pattern}",
            "pattern": pattern,
        },
    }
    cases: list[Case] = []
    left = tensor(left_shape, dtype_name="float64", kind="ramp")
    right = tensor(right_shape, dtype_name="float64", kind="constant", value=2.0)
    if backend in ACCELERATED:
        provider = provider_module(backend)
        kernels = load_backend(backend)
        raw_left = provider_array(
            provider, left_shape, dtype_name="float64", kind="ramp"
        )
        raw_right = provider_array(
            provider, right_shape, dtype_name="float64", kind="constant", value=2.0
        )
        cases.append(
            Case(
                name=f"provider.broadcast_multiply/{pattern}",
                run=lambda: provider.multiply(raw_left, raw_right),
                layer="provider",
                validate=lambda: provider.multiply(raw_left, raw_right),
                description="raw provider broadcasting multiply",
                backends=ACCELERATED,
                **common,
            )
        )

        def run_kernel() -> Any:
            return kernels.multiply(
                left, right, dtype=ts.float64, output_shape=output_shape
            )

        def validate_kernel() -> None:
            if run_kernel() is None:
                raise Unsupported(
                    "the binary kernel declines this broadcast and defers to the Python reference implementation"
                )

        cases.append(
            Case(
                name=f"kernel.broadcast_multiply/{pattern}",
                run=run_kernel,
                layer="kernel",
                validate=validate_kernel,
                description="internal broadcasting binary kernel",
                backends=ACCELERATED,
                **common,
            )
        )
        cases.append(
            Case(
                name=f"dispatch.broadcast_multiply/{pattern}",
                run=lambda: execute_multiply(
                    left, right, dtype=ts.float64, output_shape=output_shape
                ),
                layer="dispatch",
                validate=lambda: execute_multiply(
                    left, right, dtype=ts.float64, output_shape=output_shape
                ),
                description="execute_multiply over a broadcast",
                backends=ACCELERATED,
                **common,
            )
        )
    cases.append(
        Case(
            name=f"public.broadcast_multiply/{pattern}",
            run=lambda: left * right,
            layer="public",
            validate=lambda: left * right,
            description="public broadcasting multiply, including shape agreement and broadcast validation",
            **common,
        )
    )
    return cases


def _gradient_cases(
    backend: str,
    pattern: str,
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
) -> list[Case]:
    """Measure the broadcast reduction a reverse pass performs."""
    if backend not in ACCELERATED:
        raise Unsupported(
            "sum_to_shape and sum_products_to_shape are array kernels; the Python backend reduces broadcast gradients in its reference reduction path instead"
        )
    output_shape = _broadcast_shape(left_shape, right_shape)
    if right_shape == output_shape:
        raise Unsupported(
            "this pattern does not broadcast the right operand, so no gradient reduction is performed for it"
        )
    elements = _elements(output_shape)
    common: dict[str, Any] = {
        "family": "broadcast/gradient",
        "dtype": "float64",
        "shape": output_shape,
        "elements": elements,
        "work_items": elements,
        "tags": {
            "curve": f"broadcast-gradient-{pattern}",
            "pattern": pattern,
            "phase": "backward",
        },
        "backends": ACCELERATED,
    }
    cases: list[Case] = []
    provider = provider_module(backend)
    gradient = tensor(output_shape, dtype_name="float64", kind="ramp")
    factor = tensor(right_shape, dtype_name="float64", kind="constant", value=2.0)
    rank = len(output_shape)
    padded = (1,) * (rank - len(right_shape)) + right_shape
    axes = tuple(
        (
            index
            for index, (source, target) in enumerate(zip(output_shape, padded))
            if source != target
        )
    )
    raw_gradient = provider_array(
        provider, output_shape, dtype_name="float64", kind="ramp"
    )
    cases.append(
        Case(
            name=f"provider.sum_to_shape/{pattern}",
            run=lambda: provider.sum(raw_gradient, axis=axes).reshape(right_shape),
            layer="provider",
            validate=lambda: provider.sum(raw_gradient, axis=axes).reshape(right_shape),
            description="the provider reduction a broadcast gradient needs",
            **common,
        )
    )

    def run_sum_to_shape() -> Any:
        return execute_sum_to_shape(gradient, right_shape)

    def validate_sum_to_shape() -> None:
        if run_sum_to_shape() is None:
            raise Unsupported(
                "sum_to_shape declined this configuration and deferred to the Python reference implementation"
            )

    cases.append(
        Case(
            name=f"dispatch.sum_to_shape/{pattern}",
            run=run_sum_to_shape,
            layer="dispatch",
            validate=validate_sum_to_shape,
            description="the guarded broadcast-gradient reduction used by reverse passes",
            **common,
        )
    )

    def run_sum_products() -> Any:
        return execute_sum_products_to_shape(gradient, factor, right_shape)

    def validate_sum_products() -> None:
        if run_sum_products() is None:
            raise Unsupported(
                "sum_products_to_shape declined this configuration and deferred to the Python reference implementation"
            )

    cases.append(
        Case(
            name=f"dispatch.sum_products_to_shape/{pattern}",
            run=run_sum_products,
            validate=validate_sum_products,
            layer="dispatch",
            description="the fused multiply-and-reduce broadcast VJP",
            **common,
        )
    )
    return cases


def groups() -> list[Group]:
    """Return forward and backward broadcasting groups."""
    result: list[Group] = []
    for pattern, (left_shape, right_shape) in PATTERNS.items():
        elements = _elements(_broadcast_shape(left_shape, right_shape))

        def forward(
            backend: str,
            pattern: str = pattern,
            left_shape: tuple[int, ...] = left_shape,
            right_shape: tuple[int, ...] = right_shape,
            elements: int = elements,
        ) -> Sequence[Case]:
            if backend == "python" and elements > 100_000:
                raise Unsupported(
                    f"{elements} broadcast elements exceeds the Python backend ceiling"
                )
            return _forward_cases(backend, pattern, left_shape, right_shape)

        result.append(
            Group(
                name=f"broadcasting/forward/{pattern}",
                factory=forward,
                suite="broadcasting",
            )
        )

        def gradient(
            backend: str,
            pattern: str = pattern,
            left_shape: tuple[int, ...] = left_shape,
            right_shape: tuple[int, ...] = right_shape,
        ) -> Sequence[Case]:
            return _gradient_cases(backend, pattern, left_shape, right_shape)

        result.append(
            Group(
                name=f"broadcasting/gradient/{pattern}",
                factory=gradient,
                suite="broadcasting",
            )
        )
    for rows in (8, 64, 512, 4_096):
        columns = 512

        def curve(
            backend: str, rows: int = rows, columns: int = columns
        ) -> Sequence[Case]:
            if backend == "python" and rows * columns > 100_000:
                raise Unsupported("exceeds the Python backend ceiling")
            return _forward_cases(
                backend, f"bias-row-{rows}x{columns}", (rows, columns), (columns,)
            )

        result.append(
            Group(
                name=f"broadcasting/bias-row/{rows}x{columns}",
                factory=curve,
                suite="broadcasting",
            )
        )
    return result


__all__ = ["groups"]
