"""Matrix products and other linear algebra, at every isolatable depth.

Matrix multiplication is the case where native compute should dominate, so
it is the clearest test of whether the library's fixed cost is being paid
per call or per element. Square sides sweep from a size where framework
overhead is everything up to one where the provider's GEMM is, and the
rectangular shapes cover the degenerate aspect ratios a matrix-vector
product produces.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import tensors as ts
from tensors.backend import execute_matmul, execute_matmul_gradient
from tensors.graph import Computation

from ..case import Case, Group, Unsupported
from ..workloads import (
    ACCELERATED,
    FLOAT_DTYPES,
    MATRIX_SIDES,
    RECTANGULAR_SHAPES,
    dtype_of,
    kernel_module,
    label,
    provider_array,
    provider_module,
    tensor,
)


def _matmul_cases(
    backend: str,
    rows: int,
    contraction: int,
    columns: int,
    dtype_name: str,
) -> list[Case]:
    """Build every rung of one matrix multiplication."""
    dtype = dtype_of(dtype_name)
    left_shape = (rows, contraction)
    right_shape = (contraction, columns)
    output_shape = (rows, columns)
    elements = rows * columns
    # Multiply-accumulates, which is what a GEMM's cost actually scales with.
    operations = rows * contraction * columns
    suffix = f"{dtype_name}/{rows}x{contraction}x{columns}"
    common: dict[str, Any] = {
        "family": "linalg/matmul",
        "dtype": dtype_name,
        "shape": output_shape,
        "elements": elements,
        "work_items": operations,
        "tags": {
            "ladder": f"matmul|{suffix}",
            "curve": f"matmul|{dtype_name}",
            "dtype_pair": "matmul",
            "shape_class": (
                "square" if rows == contraction == columns else "rectangular"
            ),
        },
    }
    cases: list[Case] = []
    left = tensor(left_shape, dtype_name=dtype_name, kind="ramp")
    right = tensor(right_shape, dtype_name=dtype_name, kind="constant", value=0.5)

    if backend in ACCELERATED:
        provider = provider_module(backend)
        kernels = kernel_module(backend)
        raw_left = provider_array(
            provider, left_shape, dtype_name=dtype_name, kind="ramp"
        )
        raw_right = provider_array(
            provider, right_shape, dtype_name=dtype_name,
            kind="constant", value=0.5,
        )
        cases.append(Case(
            name=f"provider.matmul/{suffix}",
            run=lambda: provider.matmul(raw_left, raw_right),
            layer="provider",
            validate=lambda: provider.matmul(raw_left, raw_right),
            description=f"raw {provider.__name__}.matmul",
            backends=ACCELERATED,
            **common,
        ))

        def run_kernel() -> Any:
            return kernels.matmul(
                left, right, dtype=dtype, output_shape=output_shape
            )

        def validate_kernel() -> None:
            if run_kernel() is None:
                raise Unsupported(
                    "the matmul kernel declines this dtype and defers to "
                    "the Python reference implementation"
                )

        cases.append(Case(
            name=f"kernel.matmul/{suffix}",
            run=run_kernel,
            layer="kernel",
            validate=validate_kernel,
            description=(
                "internal matmul kernel, including its finite-result check"
            ),
            backends=ACCELERATED,
            **common,
        ))
        cases.append(Case(
            name=f"dispatch.matmul/{suffix}",
            run=lambda: execute_matmul(
                left, right, dtype=dtype, output_shape=output_shape
            ),
            layer="dispatch",
            validate=lambda: execute_matmul(
                left, right, dtype=dtype, output_shape=output_shape
            ),
            description="execute_matmul: policy and kernel lookup",
            backends=ACCELERATED,
            **common,
        ))

    cases.append(Case(
        name=f"public.matmul/{suffix}",
        run=lambda: ts.matmul(left, right),
        layer="public",
        validate=lambda: ts.matmul(left, right),
        description="public matrix multiplication",
        **common,
    ))

    left_variable = ts.Variable(left, requires_grad=False)
    right_variable = ts.Variable(right, requires_grad=False)
    cases.append(Case(
        name=f"variable.matmul/{suffix}",
        run=lambda: ts.matmul(left_variable, right_variable),
        layer="variable",
        validate=lambda: ts.matmul(left_variable, right_variable),
        description="eager differentiable matrix multiplication",
        gc_enabled=True,
        **common,
    ))

    traced = ts.matmul(left_variable, right_variable)
    computation = Computation(traced)
    computation.forward()
    cases.append(Case(
        name=f"replay.matmul/{suffix}",
        run=computation.forward,
        layer="graph-replay",
        validate=computation.forward,
        description="forward replay of a compiled matrix multiplication",
        **common,
    ))

    if backend in ACCELERATED:
        gradient = tensor(
            output_shape, dtype_name=dtype_name, kind="constant", value=1.0
        )
        backward_common = dict(common)
        backward_common["tags"] = dict(common["tags"])
        backward_common["tags"]["ladder"] = f"matmul-vjp|{suffix}"
        backward_common["tags"]["phase"] = "backward"
        backward_common["tags"]["pair"] = f"matmul|{suffix}"

        def run_gradient() -> Any:
            return execute_matmul_gradient(gradient, left, right)

        def validate_gradient() -> None:
            if run_gradient() is None:
                raise Unsupported(
                    "the matmul VJP kernel declines this configuration"
                )

        cases.append(Case(
            name=f"vjp.matmul/{suffix}",
            run=run_gradient,
            layer="dispatch",
            validate=validate_gradient,
            description="both matmul VJPs through dispatch",
            backends=ACCELERATED,
            **backward_common,
        ))

    return cases


def _other_cases(backend: str, size: int) -> list[Case]:
    """Build cases for dot, outer, norm, and transpose."""
    common_tags = {"curve": "linalg-other"}
    cases: list[Case] = []
    vector = tensor((size,), dtype_name="float64", kind="ramp")
    other = tensor((size,), dtype_name="float64", kind="constant", value=0.5)

    definitions: dict[str, tuple[Any, Any, str, int]] = {
        "dot": (
            lambda: ts.dot(vector, other),
            lambda provider, raw_a, raw_b: provider.dot(raw_a, raw_b),
            "vector dot product",
            size,
        ),
        "norm": (
            lambda: ts.norm(vector),
            lambda provider, raw_a, raw_b: provider.linalg.norm(raw_a),
            "vector norm",
            size,
        ),
    }
    # An outer product is quadratic, so it gets its own smaller sizes.
    if size <= 2_048:
        definitions["outer"] = (
            lambda: ts.outer(vector, other),
            lambda provider, raw_a, raw_b: provider.outer(raw_a, raw_b),
            "outer product",
            size * size,
        )

    if backend in ACCELERATED:
        provider = provider_module(backend)
        raw_a = provider_array(
            provider, (size,), dtype_name="float64", kind="ramp"
        )
        raw_b = provider_array(
            provider, (size,), dtype_name="float64", kind="constant", value=0.5
        )
        for name, (_, provider_call, description, work) in definitions.items():
            cases.append(Case(
                name=f"provider.{name}/{size}",
                run=lambda call=provider_call: call(provider, raw_a, raw_b),
                layer="provider",
                validate=lambda call=provider_call: call(
                    provider, raw_a, raw_b
                ),
                description=f"raw provider {description}",
                family=f"linalg/{name}",
                dtype="float64",
                shape=(size,),
                elements=size,
                work_items=work,
                backends=ACCELERATED,
                tags={**common_tags, "ladder": f"{name}|float64|{size}"},
            ))

    for name, (public_call, _, description, work) in definitions.items():
        cases.append(Case(
            name=f"public.{name}/{size}",
            run=public_call,
            layer="public",
            validate=public_call,
            description=f"public {description}",
            family=f"linalg/{name}",
            dtype="float64",
            shape=(size,),
            elements=size,
            work_items=work,
            tags={**common_tags, "ladder": f"{name}|float64|{size}"},
        ))
    return cases


def groups() -> list[Group]:
    """Return square, rectangular, and vector linear-algebra groups."""
    result: list[Group] = []

    for dtype_name in FLOAT_DTYPES:
        for side in sorted({
            item for sides in MATRIX_SIDES.values() for item in sides
        }):
            def square(
                backend: str, side: int = side, dtype_name: str = dtype_name,
            ) -> Sequence[Case]:
                if side not in MATRIX_SIDES[backend]:
                    raise Unsupported(
                        f"a {side}x{side} product exceeds what the "
                        f"{backend} backend can carry in this run; its "
                        f"sides are {MATRIX_SIDES[backend]}"
                    )
                return _matmul_cases(backend, side, side, side, dtype_name)

            result.append(Group(
                name=f"linalg/matmul/square/{dtype_name}/{side}",
                factory=square,
                suite="linalg",
            ))

    for rows, contraction, columns in RECTANGULAR_SHAPES:
        def rectangular(
            backend: str,
            rows: int = rows,
            contraction: int = contraction,
            columns: int = columns,
        ) -> Sequence[Case]:
            if backend == "python":
                raise Unsupported(
                    "the rectangular shapes are sized for accelerated "
                    "backends; the Python backend contracts element by "
                    "element and would dominate the run"
                )
            return _matmul_cases(
                backend, rows, contraction, columns, "float64"
            )

        result.append(Group(
            name=(
                f"linalg/matmul/rectangular/"
                f"{rows}x{contraction}x{columns}"
            ),
            factory=rectangular,
            suite="linalg",
        ))

    for size in (8, 128, 2_048, 65_536, 1_000_000):
        def other(backend: str, size: int = size) -> Sequence[Case]:
            if backend == "python" and size > 10_000:
                raise Unsupported(
                    "exceeds the Python backend ceiling for vector products"
                )
            return _other_cases(backend, size)

        result.append(Group(
            name=f"linalg/vector/{size}",
            factory=other,
            suite="linalg",
        ))

    return result


__all__ = ["groups"]
