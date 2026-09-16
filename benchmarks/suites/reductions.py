"""Reductions at every isolatable depth, including their stability guards.

Reductions get the deepest treatment because they carry the most machinery
per element of real work: a scaled accumulation path, a separate safety
predicate over the same values, and a host-visible decision that consumes
both. Those pieces are measured on their own so the guard can be
distinguished from the reduction.

The data pattern matters as much as the size. The guard has a fast path for
finite same-sign values, so a same-sign buffer and an alternating-sign
buffer take different routes through the same kernel; both are measured and
tagged so the difference is attributable.
"""

from __future__ import annotations
from tensors.backend import dispatch as backend_dispatch
from collections.abc import Sequence
from typing import Any
import tensors as ts
import importlib

from ..case import Case, Group, Unsupported
from benchmarks.profiles import selected_sizes
from benchmarks.workloads import (
    ACCELERATED,
    FLOAT_DTYPES,
    REDUCTION_CEILING,
    dtype_of,
    kernel_module,
    provider_array,
    provider_module,
    tensor,
)

_PUBLIC = {
    "sum": ts.sum,
    "mean": ts.mean,
    "min": ts.min,
    "max": ts.max,
    "prod": ts.prod,
    "variance": ts.variance,
    "std": ts.std,
    "norm": ts.norm,
}
_PROVIDER_NAME = {
    "sum": "sum",
    "mean": "mean",
    "min": "min",
    "max": "max",
    "prod": "prod",
    "variance": "var",
    "std": "std",
    "norm": "linalg.norm",
}
_DIFFERENTIABLE = ("sum", "mean", "min", "max", "prod", "variance", "std")


def _provider_call(
    provider: Any, name: str, array: Any, axes: Any, keepdims: bool
) -> Any:
    """Invoke the provider primitive matching a reduction."""
    if name == "linalg.norm":
        return provider.linalg.norm(array)
    function = getattr(provider, name)
    if axes is None:
        return function(array)
    return function(array, axis=axes, keepdims=keepdims)


def _reduction_cases(
    backend: str,
    operation: str,
    dtype_name: str,
    shape: tuple[int, ...],
    axis: Any,
    keepdims: bool,
    data_kind: str,
) -> list[Case]:
    """Build every rung for one reduction configuration."""
    size = 1
    for dimension in shape:
        size *= dimension
    dtype = dtype_of(dtype_name)
    axis_label = (
        "all"
        if axis is None
        else (
            "-".join((str(item) for item in axis))
            if isinstance(axis, tuple)
            else str(axis)
        )
    )
    suffix = f"{dtype_name}/{'x'.join((str(item) for item in shape))}/axis-{axis_label}/keepdims-{int(keepdims)}/{data_kind}"
    ladder = f"{operation}|{suffix}"
    common: dict[str, Any] = {
        "family": f"reduction/{operation}",
        "dtype": dtype_name,
        "shape": shape,
        "elements": size,
        "work_items": size,
        "tags": {
            "ladder": ladder,
            "curve": f"reduction-{operation}|{dtype_name}|{axis_label}|{data_kind}",
            "dtype_pair": f"reduction-{operation}|{axis_label}|{data_kind}",
            "pair": f"reduction-{operation}|{suffix.replace(data_kind, '')}",
            "data": "same-sign" if data_kind == "ramp" else "mixed-sign",
            "op": operation,
            "axis": axis_label,
            "keepdims": str(keepdims),
        },
    }
    cases: list[Case] = []
    value = tensor(shape, dtype_name=dtype_name, kind=data_kind)
    if axis is None:
        axes = tuple(range(len(shape)))
    elif isinstance(axis, tuple):
        axes = axis
    else:
        axes = (axis,)
    if keepdims:
        output_shape = tuple(
            (1 if index in axes else item for index, item in enumerate(shape))
        )
    else:
        output_shape = tuple(
            (item for index, item in enumerate(shape) if index not in axes)
        )
        if axis is None:
            output_shape = (1,)
    if backend in ACCELERATED:
        provider = provider_module(backend)
        kernels = kernel_module(backend)
        raw = provider_array(provider, shape, dtype_name=dtype_name, kind=data_kind)
        provider_name = _PROVIDER_NAME[operation]

        def run_provider() -> Any:
            return _provider_call(
                provider, provider_name, raw, None if axis is None else axes, keepdims
            )

        cases.append(
            Case(
                name=f"provider.{operation}/{suffix}",
                run=run_provider,
                layer="provider",
                validate=run_provider,
                description=f"raw {provider.__name__}.{provider_name}",
                backends=ACCELERATED,
                **common,
            )
        )
        if operation in ("sum", "mean"):
            array_module = provider_module(backend)
            _view = importlib.import_module(
                f"tensors.backend.{backend}.conversion"
            )._view
            stability = importlib.import_module(
                f"tensors.backend.{backend}.kernels.reductions.stability"
            )
            _scaled_sum = stability._scaled_sum
            _summation_guard = stability._summation_guard
            working = _view(value).astype(array_module.float64, copy=False)
            guard_common = dict(common)
            guard_common["tags"] = dict(common["tags"])
            guard_common["tags"].pop("ladder", None)
            cases.append(
                Case(
                    name=f"guard.summation_guard/{suffix}",
                    run=lambda: _summation_guard(working, axes=axes, keepdims=True),
                    layer="kernel",
                    validate=lambda: _summation_guard(
                        working, axes=axes, keepdims=True
                    ),
                    description="the reduction stability predicate alone, without the host-visible bool() that consumes it",
                    backends=ACCELERATED,
                    **guard_common,
                )
            )
            cases.append(
                Case(
                    name=f"guard.summation_guard_bool/{suffix}",
                    run=lambda: bool(
                        _summation_guard(working, axes=axes, keepdims=True)
                    ),
                    layer="kernel",
                    validate=lambda: bool(
                        _summation_guard(working, axes=axes, keepdims=True)
                    ),
                    description="the stability predicate plus the bool() conversion, which is a host/device barrier on CUDA",
                    backends=ACCELERATED,
                    **guard_common,
                )
            )
            cases.append(
                Case(
                    name=f"guard.scaled_sum/{suffix}",
                    run=lambda: _scaled_sum(working, axes),
                    layer="kernel",
                    validate=lambda: _scaled_sum(working, axes),
                    description="the normalized accumulation path alone",
                    backends=ACCELERATED,
                    **guard_common,
                )
            )

        def run_kernel() -> Any:
            return getattr(kernels, "reduce_" + operation)(
                value, axes, keepdims=keepdims, dtype=dtype, output_shape=output_shape
            )

        def validate_kernel() -> None:
            if run_kernel() is None:
                raise Unsupported(
                    "the guarded reduction kernel declines this configuration and defers to the Python reference implementation"
                )

        cases.append(
            Case(
                name=f"kernel.{operation}/{suffix}",
                run=run_kernel,
                layer="kernel",
                validate=validate_kernel,
                description="internal guarded reduction kernel",
                backends=ACCELERATED,
                **common,
            )
        )
        cases.append(
            Case(
                name=f"dispatch.{operation}/{suffix}",
                run=lambda: getattr(backend_dispatch, f"execute_reduce_{operation}")(
                    value,
                    axes,
                    keepdims=keepdims,
                    dtype=dtype,
                    output_shape=output_shape,
                ),
                layer="dispatch",
                validate=lambda: getattr(
                    backend_dispatch, f"execute_reduce_{operation}"
                )(
                    value,
                    axes,
                    keepdims=keepdims,
                    dtype=dtype,
                    output_shape=output_shape,
                ),
                description="execute_reduction: policy and kernel lookup",
                backends=ACCELERATED,
                **common,
            )
        )
    public = _PUBLIC[operation]
    cases.append(
        Case(
            name=f"public.{operation}/{suffix}",
            run=lambda: public(value, axis=axis, keepdims=keepdims),
            layer="public",
            validate=lambda: public(value, axis=axis, keepdims=keepdims),
            description="public reduction function",
            **common,
        )
    )
    if backend in ACCELERATED and operation in _DIFFERENTIABLE:
        kernels = kernel_module(backend)
        gradient = tensor(output_shape, dtype_name=dtype_name, kind="constant")
        backward_common = dict(common)
        backward_common["tags"] = dict(common["tags"])
        backward_common["tags"]["ladder"] = f"{operation}-vjp|{suffix}"
        backward_common["tags"]["phase"] = "backward"
        backward_common["tags"]["pair"] = f"reduction-vjp|{ladder}"

        def run_gradient() -> Any:
            return getattr(backend_dispatch, f"execute_reduce_{operation}_gradient")(
                gradient, value, axes, keepdims=keepdims
            )

        def validate_gradient() -> None:
            if run_gradient() is None:
                raise Unsupported(
                    "the reduction VJP kernel declines this configuration and defers to the Python reference implementation"
                )

        cases.append(
            Case(
                name=f"vjp.{operation}/{suffix}",
                run=run_gradient,
                layer="dispatch",
                validate=validate_gradient,
                description="reduction VJP through dispatch",
                backends=ACCELERATED,
                **backward_common,
            )
        )
    return cases


def _normalization_cases(
    backend: str, operation: str, dtype_name: str, shape: tuple[int, ...]
) -> list[Case]:
    """Build rungs for softmax, log-softmax, and logsumexp."""
    size = shape[0] * shape[1]
    suffix = f"{dtype_name}/{shape[0]}x{shape[1]}"
    common: dict[str, Any] = {
        "family": f"normalization/{operation}",
        "dtype": dtype_name,
        "shape": shape,
        "elements": size,
        "work_items": size,
        "tags": {
            "ladder": f"{operation}|{suffix}",
            "curve": f"normalization-{operation}|{dtype_name}",
            "dtype_pair": f"normalization-{operation}",
            "op": operation,
        },
    }
    cases: list[Case] = []
    value = tensor(shape, dtype_name=dtype_name, kind="ramp")
    if backend in ACCELERATED and operation in ("softmax", "log_softmax"):
        kernels = kernel_module(backend)

        def run_kernel() -> Any:
            return getattr(kernels, operation)(value, -1, dtype=dtype_of(dtype_name))

        def validate_kernel() -> None:
            if run_kernel() is None:
                raise Unsupported(
                    "the normalization kernel declines this configuration"
                )

        cases.append(
            Case(
                name=f"kernel.{operation}/{suffix}",
                run=run_kernel,
                layer="kernel",
                validate=validate_kernel,
                description="internal normalization kernel",
                backends=ACCELERATED,
                **common,
            )
        )
    public = {
        "softmax": ts.softmax,
        "log_softmax": ts.log_softmax,
        "logsumexp": ts.logsumexp,
    }[operation]
    cases.append(
        Case(
            name=f"public.{operation}/{suffix}",
            run=lambda: public(value, axis=-1),
            layer="public",
            validate=lambda: public(value, axis=-1),
            description="public normalization function",
            **common,
        )
    )
    return cases


def _arg_extremum_cases(
    backend: str, operation: str, shape: tuple[int, ...]
) -> list[Case]:
    """Build rungs for argmin and argmax."""
    size = 1
    for dimension in shape:
        size *= dimension
    suffix = f"float64/{'x'.join((str(item) for item in shape))}"
    common: dict[str, Any] = {
        "family": f"reduction/{operation}",
        "dtype": "float64",
        "shape": shape,
        "elements": size,
        "work_items": size,
        "tags": {
            "ladder": f"{operation}|{suffix}",
            "curve": f"reduction-{operation}|float64",
            "op": operation,
        },
    }
    cases: list[Case] = []
    value = tensor(shape, dtype_name="float64", kind="ramp")
    if backend in ACCELERATED:
        provider = provider_module(backend)
        kernels = kernel_module(backend)
        raw = provider_array(provider, shape, dtype_name="float64", kind="ramp")
        provider_function = getattr(provider, operation)
        cases.append(
            Case(
                name=f"provider.{operation}/{suffix}",
                run=lambda: provider_function(raw),
                layer="provider",
                validate=lambda: provider_function(raw),
                description=f"raw {provider.__name__}.{operation}",
                backends=ACCELERATED,
                **common,
            )
        )

        def run_kernel() -> Any:
            return getattr(kernels, operation)(
                value, None, keepdims=False, output_shape=(1,)
            )

        def validate_kernel() -> None:
            if run_kernel() is None:
                raise Unsupported("the arg-extremum kernel declines this configuration")

        cases.append(
            Case(
                name=f"kernel.{operation}/{suffix}",
                run=run_kernel,
                layer="kernel",
                validate=validate_kernel,
                description="internal arg-extremum kernel",
                backends=ACCELERATED,
                **common,
            )
        )
    public = ts.argmax if operation == "argmax" else ts.argmin
    cases.append(
        Case(
            name=f"public.{operation}/{suffix}",
            run=lambda: public(value),
            layer="public",
            validate=lambda: public(value),
            description="public arg-extremum function",
            **common,
        )
    )
    return cases


def _register(
    result: list[Group],
    name: str,
    builder: Any,
    *,
    elements: int,
    ceiling: dict[str, int] | None = None,
) -> None:
    """Add one group, gating it on the per-backend element ceiling."""
    limits = ceiling or REDUCTION_CEILING

    def factory(backend: str) -> Sequence[Case]:
        if elements > limits[backend]:
            raise Unsupported(
                f"{elements} elements exceeds the {backend} reduction ceiling of {limits[backend]}; guarded reductions build several full-size temporaries"
            )
        return builder(backend)

    result.append(Group(name=name, factory=factory, suite="reductions"))


def groups() -> list[Group]:
    """Return reduction groups over operations, shapes, axes, and dtypes."""
    result: list[Group] = []
    for operation in _PUBLIC:
        for dtype_name in FLOAT_DTYPES:
            for size in selected_sizes((1, 100, 10_000, 1_000_000, 4_000_000)):
                for data_kind in ("ramp", "mixed"):
                    _register(
                        result,
                        f"reductions/{operation}/{dtype_name}/{size}/{data_kind}",
                        lambda backend, operation=operation, dtype_name=dtype_name, size=size, data_kind=data_kind: _reduction_cases(
                            backend,
                            operation,
                            dtype_name,
                            (size,),
                            None,
                            False,
                            data_kind,
                        ),
                        elements=size,
                    )
    for operation in ("sum", "mean", "max", "std"):
        for shape in ((512, 512), (64, 16_384), (16_384, 64)):
            for axis in (None, 0, 1):
                for keepdims in (False, True):
                    if axis is None and keepdims:
                        continue
                    _register(
                        result,
                        f"reductions/axis/{operation}/{shape[0]}x{shape[1]}/{axis}/{int(keepdims)}",
                        lambda backend, operation=operation, shape=shape, axis=axis, keepdims=keepdims: _reduction_cases(
                            backend, operation, "float64", shape, axis, keepdims, "ramp"
                        ),
                        elements=shape[0] * shape[1],
                    )
    for operation in ("sum", "mean"):
        for axis in ((0, 1), (1, 2), (0, 2)):
            _register(
                result,
                f"reductions/multiaxis/{operation}/{'-'.join((str(item) for item in axis))}",
                lambda backend, operation=operation, axis=axis: _reduction_cases(
                    backend, operation, "float64", (64, 64, 64), axis, False, "ramp"
                ),
                elements=64**3,
            )
    for operation in ("softmax", "log_softmax", "logsumexp"):
        for dtype_name in FLOAT_DTYPES:
            for shape in ((32, 32), (256, 1_024), (1_024, 1_024)):
                _register(
                    result,
                    f"reductions/{operation}/{dtype_name}/{shape[0]}x{shape[1]}",
                    lambda backend, operation=operation, dtype_name=dtype_name, shape=shape: _normalization_cases(
                        backend, operation, dtype_name, shape
                    ),
                    elements=shape[0] * shape[1],
                )
    for operation in ("argmax", "argmin"):
        for size in selected_sizes((100, 10_000, 1_000_000)):
            _register(
                result,
                f"reductions/{operation}/{size}",
                lambda backend, operation=operation, size=size: _arg_extremum_cases(
                    backend, operation, (size,)
                ),
                elements=size,
            )
    return result


__all__ = ["groups"]
