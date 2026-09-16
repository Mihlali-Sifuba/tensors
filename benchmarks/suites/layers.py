"""Layer ladders over elementwise work: provider to replayed graph.

The same arithmetic is measured at every depth it can be isolated at, on the
same values, so the difference between two rungs is the cost of the layer
between them and nothing else.

The rungs, innermost first:

``provider``
    ``numpy.add`` or ``cupy.add`` on native arrays.
``kernel``
    the internal guarded kernel, called directly with Tensors.
``dispatch``
    the dedicated arithmetic dispatch function, which adds the workload policy and kernel lookup.
``public``
    ``Tensor.__add__``, which adds dtype resolution, broadcasting
    validation, and result construction.
``variable``
    ``Variable.__add__``, which additionally records graph structure and
    compiles and runs a one-instruction program.
``graph-replay``
    ``Computation.forward()`` over the already-compiled program.
"""

from __future__ import annotations
from tensors.backend import dispatch as backend_dispatch
from collections.abc import Sequence
from typing import Any
import tensors as ts
from tensors.backend.dispatch import arithmetic
from tensors.backend.loading import load_backend
from tensors.graph import Computation
from benchmarks.case import Case, Group, Unsupported
from benchmarks.workloads import (
    ACCELERATED,
    FLOAT_DTYPES,
    INTEGER_DTYPES,
    SIZE_CEILING,
    close,
    dtype_of,
    first,
    is_integer,
    kernel_module,
    provider_array,
    provider_module,
    sizes_for,
    tensor,
)

_BINARY = {
    "add": ("add", lambda left, right: left + right),
    "subtract": ("subtract", lambda left, right: left - right),
    "multiply": ("multiply", lambda left, right: left * right),
    "divide": ("true_divide", lambda left, right: left / right),
}
_UNARY = ("exp", "log", "sqrt", "tanh", "sin", "abs", "sign")
_PUBLIC_UNARY = {
    "exp": lambda value: ts.exp(value),
    "log": lambda value: ts.log(value),
    "sqrt": lambda value: ts.sqrt(value),
    "tanh": lambda value: ts.tanh(value),
    "sin": lambda value: ts.sin(value),
    "abs": lambda value: ts.abs(value),
    "sign": lambda value: ts.sign(value),
    "relu": lambda value: ts.relu(value),
    "sigmoid": lambda value: ts.sigmoid(value),
    "softplus": lambda value: ts.softplus(value),
}
INTEGER_ELEMENTWISE_CEILING = 1_000_000
_COMPARISONS = {
    "equal": ("equal", ts.equal),
    "less": ("less", ts.less),
    "greater_equal": ("greater_equal", ts.greater_equal),
}


def _binary_ladder(
    backend: str, operation: str, dtype_name: str, size: int
) -> list[Case]:
    """Build every measurable rung of one binary operation."""
    provider_name, public_operation = _BINARY[operation]
    dtype = dtype_of(dtype_name)
    shape = (size,)
    ladder = f"{operation}|{dtype_name}|{size}"
    curve = f"binary-{operation}|{dtype_name}"
    common: dict[str, Any] = {
        "family": f"binary/{operation}",
        "dtype": dtype_name,
        "shape": shape,
        "elements": size,
        "work_items": size,
        "tags": {
            "ladder": ladder,
            "curve": curve,
            "dtype_pair": f"binary-{operation}",
            "op": operation,
        },
    }
    cases: list[Case] = []
    left = tensor(shape, dtype_name=dtype_name, kind="ramp")
    right = tensor(shape, dtype_name=dtype_name, kind="constant", value=2.0)
    expected = float(
        public_operation(
            first(left) if not is_integer(dtype_name) else int(first(left)),
            2.0 if not is_integer(dtype_name) else 2,
        )
    )
    if backend in ACCELERATED:
        provider = provider_module(backend)
        kernels = load_backend(backend)
        raw_left = provider_array(provider, shape, dtype_name=dtype_name, kind="ramp")
        raw_right = provider_array(
            provider, shape, dtype_name=dtype_name, kind="constant", value=2.0
        )
        provider_function = getattr(provider, provider_name)

        def run_provider() -> Any:
            return provider_function(raw_left, raw_right)

        def validate_provider() -> None:
            result = provider_function(raw_left, raw_right)
            assert result.size == size
            assert close(first(result), expected, tolerance=1e-05)

        cases.append(
            Case(
                name=f"provider.{operation}/{dtype_name}/{size}",
                run=run_provider,
                layer="provider",
                validate=validate_provider,
                description=f"raw {provider.__name__}.{provider_name}",
                backends=ACCELERATED,
                **common,
            )
        )

        def run_kernel() -> Any:
            return getattr(kernels, operation)(
                left, right, dtype=dtype, output_shape=shape
            )

        def validate_kernel() -> None:
            storage = run_kernel()
            if storage is None:
                raise Unsupported(
                    "the guarded array kernel declines this dtype and backend and defers to the Python reference implementation"
                )
            assert storage.size == size

        cases.append(
            Case(
                name=f"kernel.{operation}/{dtype_name}/{size}",
                run=run_kernel,
                layer="kernel",
                validate=validate_kernel,
                description="internal guarded array kernel over Tensors",
                backends=ACCELERATED,
                **common,
            )
        )

        def run_dispatch() -> Any:
            return getattr(arithmetic, f"execute_{operation}")(
                left, right, dtype=dtype, output_shape=shape
            )

        def validate_dispatch() -> None:
            run_dispatch()

        cases.append(
            Case(
                name=f"dispatch.{operation}/{dtype_name}/{size}",
                run=run_dispatch,
                layer="dispatch",
                validate=validate_dispatch,
                description="dedicated arithmetic dispatch: workload policy and provider lookup",
                backends=ACCELERATED,
                **common,
            )
        )

    def run_public() -> Any:
        return public_operation(left, right)

    def validate_public() -> None:
        result = public_operation(left, right)
        assert result.shape == shape
        assert close(first(result), expected, tolerance=1e-05)

    cases.append(
        Case(
            name=f"public.{operation}/{dtype_name}/{size}",
            run=run_public,
            layer="public",
            validate=validate_public,
            description="public Tensor operator",
            **common,
        )
    )
    if not is_integer(dtype_name):
        left_variable = ts.Variable(left, requires_grad=False)
        right_variable = ts.Variable(right, requires_grad=False)

        def run_variable() -> Any:
            return public_operation(left_variable, right_variable)

        def validate_variable() -> None:
            result = public_operation(left_variable, right_variable)
            assert result.shape == shape

        cases.append(
            Case(
                name=f"variable.{operation}/{dtype_name}/{size}",
                run=run_variable,
                layer="variable",
                validate=validate_variable,
                description="eager Variable operator: records structure, compiles a one-instruction program, and runs it",
                gc_enabled=True,
                **common,
            )
        )
        traced = public_operation(left_variable, right_variable)
        computation = Computation(traced)
        computation.forward()

        def run_replay() -> Any:
            return computation.forward()

        def validate_replay() -> None:
            result = computation.forward()
            assert result.shape == shape

        cases.append(
            Case(
                name=f"replay.{operation}/{dtype_name}/{size}",
                run=run_replay,
                layer="graph-replay",
                validate=validate_replay,
                description="forward replay of the compiled one-instruction program",
                **common,
            )
        )
    return cases


def _unary_ladder(
    backend: str, operation: str, dtype_name: str, size: int
) -> list[Case]:
    """Build provider, kernel, dispatch, and public rungs for a unary map."""
    dtype = dtype_of(dtype_name)
    shape = (size,)
    common: dict[str, Any] = {
        "family": f"unary/{operation}",
        "dtype": dtype_name,
        "shape": shape,
        "elements": size,
        "work_items": size,
        "tags": {
            "ladder": f"{operation}|{dtype_name}|{size}",
            "curve": f"unary-{operation}|{dtype_name}",
            "dtype_pair": f"unary-{operation}",
            "op": operation,
        },
    }
    cases: list[Case] = []
    value = tensor(shape, dtype_name=dtype_name, kind="ramp")
    if backend in ACCELERATED:
        provider = provider_module(backend)
        kernels = kernel_module(backend)
        raw = provider_array(provider, shape, dtype_name=dtype_name, kind="ramp")
        provider_function = getattr(provider, operation, None)
        if provider_function is not None:
            cases.append(
                Case(
                    name=f"provider.{operation}/{dtype_name}/{size}",
                    run=lambda: provider_function(raw),
                    layer="provider",
                    validate=lambda: provider_function(raw),
                    description=f"raw {provider.__name__}.{operation}",
                    backends=ACCELERATED,
                    **common,
                )
            )

        def run_kernel() -> Any:
            return getattr(kernels, operation)(value, dtype=dtype)

        def validate_kernel() -> None:
            if run_kernel() is None:
                raise Unsupported(
                    "the guarded array kernel declines this dtype and defers to the Python reference implementation"
                )

        cases.append(
            Case(
                name=f"kernel.{operation}/{dtype_name}/{size}",
                run=run_kernel,
                layer="kernel",
                validate=validate_kernel,
                description="internal guarded unary kernel",
                backends=ACCELERATED,
                **common,
            )
        )
        cases.append(
            Case(
                name=f"dispatch.{operation}/{dtype_name}/{size}",
                run=lambda: getattr(backend_dispatch, f"execute_{operation}")(
                    value, dtype=dtype
                ),
                layer="dispatch",
                validate=lambda: getattr(backend_dispatch, f"execute_{operation}")(
                    value, dtype=dtype
                ),
                description="execute_unary: policy and kernel lookup",
                backends=ACCELERATED,
                **common,
            )
        )
    public = _PUBLIC_UNARY[operation]
    cases.append(
        Case(
            name=f"public.{operation}/{dtype_name}/{size}",
            run=lambda: public(value),
            layer="public",
            validate=lambda: public(value),
            description="public unary function",
            **common,
        )
    )
    return cases


def _comparison_cases(
    backend: str, operation: str, dtype_name: str, size: int
) -> list[Case]:
    """Build provider, kernel, and public rungs for a comparison."""
    provider_name, public = _COMPARISONS[operation]
    shape = (size,)
    common: dict[str, Any] = {
        "family": f"comparison/{operation}",
        "dtype": dtype_name,
        "shape": shape,
        "elements": size,
        "work_items": size,
        "tags": {
            "ladder": f"cmp-{operation}|{dtype_name}|{size}",
            "curve": f"comparison-{operation}|{dtype_name}",
            "dtype_pair": f"comparison-{operation}",
            "op": operation,
        },
    }
    cases: list[Case] = []
    left = tensor(shape, dtype_name=dtype_name, kind="ramp")
    right = tensor(shape, dtype_name=dtype_name, kind="constant", value=1.5)
    if backend in ACCELERATED:
        provider = provider_module(backend)
        kernels = kernel_module(backend)
        raw_left = provider_array(provider, shape, dtype_name=dtype_name, kind="ramp")
        raw_right = provider_array(
            provider, shape, dtype_name=dtype_name, kind="constant", value=1.5
        )
        provider_function = getattr(provider, provider_name)
        cases.append(
            Case(
                name=f"provider.{operation}/{dtype_name}/{size}",
                run=lambda: provider_function(raw_left, raw_right),
                layer="provider",
                validate=lambda: provider_function(raw_left, raw_right),
                description=f"raw {provider.__name__}.{provider_name}",
                backends=ACCELERATED,
                **common,
            )
        )

        def run_kernel() -> Any:
            return getattr(kernels, operation)(left, right, output_shape=shape)

        def validate_kernel() -> None:
            if run_kernel() is None:
                raise Unsupported(
                    "the comparison kernel declines this dtype and defers to the Python reference implementation"
                )

        cases.append(
            Case(
                name=f"kernel.{operation}/{dtype_name}/{size}",
                run=run_kernel,
                layer="kernel",
                validate=validate_kernel,
                description="internal comparison kernel",
                backends=ACCELERATED,
                **common,
            )
        )
    cases.append(
        Case(
            name=f"public.{operation}/{dtype_name}/{size}",
            run=lambda: public(left, right),
            layer="public",
            validate=lambda: public(left, right),
            description="public comparison returning a boolean-valued Tensor",
            **common,
        )
    )
    return cases


def groups() -> list[Group]:
    """Return one group per (operation, dtype, size) ladder."""
    result: list[Group] = []
    for operation in _BINARY:
        for dtype_name in (*FLOAT_DTYPES, *INTEGER_DTYPES):
            if operation == "divide" and is_integer(dtype_name):
                continue
            for size in (1, 10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000):

                def factory(
                    backend: str,
                    operation: str = operation,
                    dtype_name: str = dtype_name,
                    size: int = size,
                ) -> Sequence[Case]:
                    if size > SIZE_CEILING[backend]:
                        raise Unsupported(
                            f"{size} elements exceeds the {backend} ceiling of {SIZE_CEILING[backend]}; the Python backend interprets element by element and the device has limited memory"
                        )
                    if is_integer(dtype_name) and size > INTEGER_ELEMENTWISE_CEILING:
                        raise Unsupported(
                            f"{size} integer elements exceeds the {INTEGER_ELEMENTWISE_CEILING} ceiling for integer elementwise work: no backend accelerates it (NumPy computes in object dtype, CUDA declines and uses the host reference implementation), so cost is per-element Python work in every case"
                        )
                    return _binary_ladder(backend, operation, dtype_name, size)

                result.append(
                    Group(
                        name=f"layers/binary/{operation}/{dtype_name}/{size}",
                        factory=factory,
                        suite="layers",
                    )
                )
    for operation in _UNARY:
        for dtype_name in FLOAT_DTYPES:
            for size in (1, 100, 10_000, 1_000_000, 10_000_000):

                def unary_factory(
                    backend: str,
                    operation: str = operation,
                    dtype_name: str = dtype_name,
                    size: int = size,
                ) -> Sequence[Case]:
                    if size > SIZE_CEILING[backend]:
                        raise Unsupported(
                            f"{size} elements exceeds the {backend} ceiling"
                        )
                    return _unary_ladder(backend, operation, dtype_name, size)

                result.append(
                    Group(
                        name=f"layers/unary/{operation}/{dtype_name}/{size}",
                        factory=unary_factory,
                        suite="layers",
                    )
                )
    for operation in ("relu", "sigmoid", "softplus"):
        for size in (1, 10_000, 1_000_000):

            def activation_factory(
                backend: str, operation: str = operation, size: int = size
            ) -> Sequence[Case]:
                if size > SIZE_CEILING[backend]:
                    raise Unsupported(f"{size} elements exceeds the {backend} ceiling")
                return _unary_ladder(backend, operation, "float64", size)

            result.append(
                Group(
                    name=f"layers/activation/{operation}/{size}",
                    factory=activation_factory,
                    suite="layers",
                )
            )
    for operation in _COMPARISONS:
        for dtype_name in ("float64", "float32", "int64"):
            for size in (1, 10_000, 1_000_000):

                def comparison_factory(
                    backend: str,
                    operation: str = operation,
                    dtype_name: str = dtype_name,
                    size: int = size,
                ) -> Sequence[Case]:
                    if size > SIZE_CEILING[backend]:
                        raise Unsupported(
                            f"{size} elements exceeds the {backend} ceiling"
                        )
                    return _comparison_cases(backend, operation, dtype_name, size)

                result.append(
                    Group(
                        name=f"layers/comparison/{operation}/{dtype_name}/{size}",
                        factory=comparison_factory,
                        suite="layers",
                    )
                )
    return result


__all__ = ["groups"]
