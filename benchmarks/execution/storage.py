"""Storage: construction, conversion, caching, and scalar extraction.

A Tensor holds one authoritative backend-native buffer and caches the other
representations it has been asked for. That makes three distinct costs worth
separating: producing the authoritative buffer, converting to another
representation the first time, and finding an already-converted one.

Scalar extraction is measured here rather than with the arithmetic because
it is a storage operation: reading one value requires the host
representation, which on CUDA means a device-to-host transfer and therefore
a synchronization.
"""

from __future__ import annotations
import array
from collections.abc import Sequence
from typing import Any
import tensors as ts
from tensors.backend.conversion import convert_storage
from benchmarks.case import Case, Group, Unsupported
from tensors.shape import Shape
from benchmarks.profiles import selected_sizes
from benchmarks.inputs import (
    ACCELERATED,
    FLOAT_DTYPES,
    _native_storage,
    dtype_of,
    provider_module,
    ramp_values,
    tensor,
)

_NATIVE_KIND = {"python": "python", "numpy": "numpy", "cuda": "cuda"}


def _construction_cases(backend: str, size: int, dtype_name: str) -> list[Case]:
    """Measure building a Tensor from each kind of source."""
    dtype = dtype_of(dtype_name)
    shape = (size,)
    values = ramp_values(size, dtype_name)
    typecode = dtype.typecode
    host_array = array.array(typecode, values)
    cases: list[Case] = []

    def common(name: str) -> dict[str, Any]:
        return {
            "family": f"storage/{name}",
            "dtype": dtype_name,
            "shape": shape,
            "elements": size,
            "work_items": size,
            "tags": {
                "ladder": f"storage-{name}|{dtype_name}|{size}",
                "curve": f"storage-{name}|{dtype_name}",
                "dtype_pair": f"storage-{name}",
            },
        }

    cases.append(
        Case(
            name=f"public.from_list/{dtype_name}/{size}",
            run=lambda: ts.Tensor(values, dtype=dtype, shape=shape),
            layer="storage",
            validate=lambda: ts.Tensor(values, dtype=dtype, shape=shape),
            description="construct a Tensor from a Python list",
            memory=True,
            **common("from_list"),
        )
    )
    cases.append(
        Case(
            name=f"public.from_array/{dtype_name}/{size}",
            run=lambda: ts.Tensor(host_array, dtype=dtype, shape=shape),
            layer="storage",
            validate=lambda: ts.Tensor(host_array, dtype=dtype, shape=shape),
            description="construct a Tensor from an array.array buffer",
            **common("from_array"),
        )
    )
    if backend in ACCELERATED:
        provider = provider_module(backend)
        native = getattr(provider, dtype_name)
        native_array = provider.asarray(values, dtype=native)

        def unsupported_provider() -> Any:
            return ts.Tensor(native_array, dtype=dtype, shape=shape)

        def validate_unsupported() -> None:
            try:
                unsupported_provider()
            except TypeError as error:
                raise Unsupported(
                    f"Tensor() does not accept a provider array: {error}. The supported route from provider values is to wrap them in the matching Storage class, measured as storage.from_provider_storage"
                ) from error

        cases.append(
            Case(
                name=f"public.from_provider_array/{dtype_name}/{size}",
                run=unsupported_provider,
                layer="storage",
                validate=validate_unsupported,
                description=f"construct a Tensor directly from a {provider.__name__} array",
                backends=ACCELERATED,
                **common("from_provider_array"),
            )
        )
        storage = _native_storage(backend, native_array, dtype)
        cases.append(
            Case(
                name=f"storage.from_provider_storage/{dtype_name}/{size}",
                run=lambda: ts.Tensor(storage, dtype=dtype, shape=shape),
                layer="storage",
                validate=lambda: ts.Tensor(storage, dtype=dtype, shape=shape),
                description="construct a Tensor from provider-backed Storage, which public construction copies for independent ownership",
                backends=ACCELERATED,
                memory=True,
                **common("from_provider_storage"),
            )
        )
        cases.append(
            Case(
                name=f"storage.from_owned_storage/{dtype_name}/{size}",
                run=lambda: ts.Tensor._from_owned_storage(
                    storage, dtype=dtype, shape=Shape.from_iterable(shape)
                ),
                layer="storage",
                validate=lambda: ts.Tensor._from_owned_storage(
                    storage, dtype=dtype, shape=Shape.from_iterable(shape)
                ),
                description="the internal path every accelerated operation ends with: adopt a provider result without copying it",
                backends=ACCELERATED,
                memory=True,
                **common("from_owned_storage"),
            )
        )
        if backend == "cuda":
            import numpy

            host_native = numpy.asarray(values, dtype=dtype_name)
            cases.append(
                Case(
                    name=f"storage.host_to_device/{dtype_name}/{size}",
                    run=lambda: _native_storage("cuda", host_native, dtype),
                    layer="storage",
                    validate=lambda: _native_storage("cuda", host_native, dtype),
                    description="move host NumPy values into device storage, which is the transfer a device Tensor needs",
                    backends=frozenset({"cuda"}),
                    memory=True,
                    **common("host_to_device"),
                )
            )
    source = tensor(shape, dtype_name=dtype_name, kind="ramp")
    cases.append(
        Case(
            name=f"public.from_tensor/{dtype_name}/{size}",
            run=lambda: ts.Tensor(source),
            layer="storage",
            validate=lambda: ts.Tensor(source),
            description="clone a Tensor through the public constructor",
            **common("from_tensor"),
        )
    )
    return cases


def _conversion_cases(backend: str, size: int, dtype_name: str) -> list[Case]:
    """Measure representation conversion, caching, and invalidation."""
    dtype = dtype_of(dtype_name)
    shape = (size,)
    native_kind = _NATIVE_KIND[backend]
    cases: list[Case] = []

    def common(name: str) -> dict[str, Any]:
        return {
            "family": f"storage/{name}",
            "dtype": dtype_name,
            "shape": shape,
            "elements": size,
            "work_items": size,
            "tags": {
                "curve": f"storage-{name}|{dtype_name}",
                "dtype_pair": f"storage-{name}",
            },
        }

    for target_kind in ("python", "numpy", "cuda"):
        if target_kind == "cuda" and "cuda" not in ts.available_backends():
            continue
        if target_kind == "numpy" and "numpy" not in ts.available_backends():
            continue
        holder: dict[str, ts.Tensor] = {}

        def reset(holder: dict[str, ts.Tensor] = holder) -> None:
            holder["value"] = tensor(shape, dtype_name=dtype_name, kind="ramp")

        def run_first(
            holder: dict[str, ts.Tensor] = holder, target_kind: str = target_kind
        ) -> Any:
            return holder["value"]._storage_for(target_kind)

        first_common = common(f"first_lookup_{target_kind}")
        cases.append(
            Case(
                name=f"storage.first_lookup/{target_kind}/{dtype_name}/{size}",
                run=run_first,
                layer="storage",
                validate=run_first,
                reset=reset,
                single_shot=True,
                description=f"first request for the {target_kind} representation of a {native_kind}-backed Tensor, which converts and caches it",
                memory=True,
                **first_common,
            )
        )
        warm = tensor(shape, dtype_name=dtype_name, kind="ramp")
        warm._storage_for(target_kind)
        cases.append(
            Case(
                name=f"storage.cached_lookup/{target_kind}/{dtype_name}/{size}",
                run=lambda warm=warm, target_kind=target_kind: warm._storage_for(
                    target_kind
                ),
                layer="storage",
                validate=lambda warm=warm, target_kind=target_kind: warm._storage_for(
                    target_kind
                ),
                description=f"repeated request for an already-cached {target_kind} representation",
                **common(f"cached_lookup_{target_kind}"),
            )
        )
    base = tensor(shape, dtype_name=dtype_name, kind="ramp")
    native_storage = base._storage_for(native_kind)
    for target_kind in ("python", "numpy", "cuda"):
        if target_kind == native_kind:
            continue
        if target_kind not in ts.available_backends():
            continue
        cases.append(
            Case(
                name=f"storage.convert/{native_kind}-to-{target_kind}/{dtype_name}/{size}",
                run=lambda target_kind=target_kind: convert_storage(
                    native_storage, target_kind
                ),
                layer="storage",
                validate=lambda target_kind=target_kind: convert_storage(
                    native_storage, target_kind
                ),
                description=f"convert {native_kind} storage to {target_kind} storage",
                memory=True,
                **common(f"convert_{native_kind}_to_{target_kind}"),
            )
        )
    cases.append(
        Case(
            name=f"storage.copy/{dtype_name}/{size}",
            run=native_storage.copy,
            layer="storage",
            validate=native_storage.copy,
            description="duplicate the authoritative buffer",
            memory=True,
            **common("copy"),
        )
    )
    cases.append(
        Case(
            name=f"storage.materialize_host/{dtype_name}/{size}",
            run=lambda: base.tolist(),
            layer="storage",
            validate=lambda: base.tolist(),
            description="materialize every logical value as a Python list, which requires the host representation",
            memory=True,
            **common("materialize_host"),
        )
    )
    other_dtype = "float32" if dtype_name == "float64" else "float64"
    cases.append(
        Case(
            name=f"storage.astype/{dtype_name}-to-{other_dtype}/{size}",
            run=lambda: base.astype(dtype_of(other_dtype)),
            layer="storage",
            validate=lambda: base.astype(dtype_of(other_dtype)),
            description="convert a Tensor to another dtype",
            **common("astype"),
        )
    )
    invalidation_holder: dict[str, ts.Tensor] = {}

    def reset_invalidation() -> None:
        value = tensor(shape, dtype_name=dtype_name, kind="ramp")
        for kind in ts.available_backends():
            value._storage_for(kind)
        invalidation_holder["value"] = value

    def run_invalidation() -> None:
        invalidation_holder["value"][0] = 9.0

    cases.append(
        Case(
            name=f"storage.invalidate/{dtype_name}/{size}",
            run=run_invalidation,
            layer="storage",
            validate=run_invalidation,
            reset=reset_invalidation,
            single_shot=True,
            description="an in-place write, which installs host storage as authoritative and discards the other cached representations",
            **common("invalidate"),
        )
    )
    scalar_source = tensor(shape, dtype_name=dtype_name, kind="ramp")
    for name, call, description in (
        (
            "item",
            lambda: tensor((1,), dtype_name=dtype_name, kind="constant").item(),
            "read the single value of a one-element Tensor",
        ),
        (
            "index",
            lambda: scalar_source[0],
            "read one element out of a larger Tensor by index",
        ),
        (
            "float",
            lambda: float(scalar_source[0]),
            "read one element and convert it to a Python float",
        ),
    ):
        cases.append(
            Case(
                name=f"storage.scalar_{name}/{dtype_name}/{size}",
                run=call,
                layer="storage",
                validate=call,
                description=description,
                work_items=None,
                family=f"storage/scalar_{name}",
                dtype=dtype_name,
                shape=shape,
                elements=size,
                tags={
                    "curve": f"storage-scalar_{name}|{dtype_name}",
                    "dtype_pair": f"storage-scalar_{name}",
                },
            )
        )
    other = tensor(shape, dtype_name=dtype_name, kind="ramp")
    cases.append(
        Case(
            name=f"storage.equality/{dtype_name}/{size}",
            run=lambda: base == other,
            layer="storage",
            validate=lambda: base == other,
            description="Tensor equality, which compares host lists and therefore materializes both operands",
            **common("equality"),
        )
    )
    return cases


def groups() -> list[Group]:
    """Return storage construction and conversion groups."""
    result: list[Group] = []
    for dtype_name in FLOAT_DTYPES:
        for size in selected_sizes((1, 100, 10_000, 1_000_000)):

            def construction(
                backend: str, size: int = size, dtype_name: str = dtype_name
            ) -> Sequence[Case]:
                if backend == "python" and size > 100_000:
                    raise Unsupported("exceeds the Python backend construction ceiling")
                return _construction_cases(backend, size, dtype_name)

            result.append(
                Group(
                    name=f"storage/construction/{dtype_name}/{size}",
                    factory=construction,
                    suite="storage",
                )
            )

            def conversion(
                backend: str, size: int = size, dtype_name: str = dtype_name
            ) -> Sequence[Case]:
                if size > 100_000 and backend == "python":
                    raise Unsupported("exceeds the Python backend conversion ceiling")
                return _conversion_cases(backend, size, dtype_name)

            result.append(
                Group(
                    name=f"storage/conversion/{dtype_name}/{size}",
                    factory=conversion,
                    suite="storage",
                )
            )
    return result


__all__ = ["groups"]
