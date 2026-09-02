"""Storage conversion, caching, mutation, and materialization benchmarks."""

from __future__ import annotations

from array import array

import tensors as ts
from tensors.storage import PythonStorage, convert_storage

from .runner import BenchmarkBackend, BenchmarkCase


_CUDA = frozenset[BenchmarkBackend]({"cuda"})
_ACCELERATED = frozenset[BenchmarkBackend]({"numpy", "cuda"})


def _host_storage(size: int) -> PythonStorage:
    return PythonStorage(array("d", [1.25]) * size, ts.float64)


def cases() -> list[BenchmarkCase]:
    """Measure cold conversion separately from cached and public operations."""
    backend = ts.get_backend()
    if backend == "python":
        return []

    selected = frozenset[BenchmarkBackend]({backend})
    benchmarks: list[BenchmarkCase] = []
    for size in (100_000, 1_000_000):
        host = _host_storage(size)
        native_kind = "cuda" if backend == "cuda" else "numpy"

        def host_to_native(source=host, kind=native_kind):
            return convert_storage(source, kind)

        def validate_host_to_native(run=host_to_native, expected_size=size) -> None:
            storage = run()
            assert storage.size == expected_size
            assert float(storage.buffer[-1]) == 1.25

        benchmarks.append(BenchmarkCase(
            name=f"storage.python_to_{native_kind}/{size}",
            run=host_to_native,
            validate=validate_host_to_native,
            work_items=size,
            description="uncached conversion from Python storage to native storage",
            layer="storage",
            backends=selected,
        ))

        cached_tensor = ts.Tensor(host, shape=(size,))
        cached_tensor._storage_for(native_kind)

        def cached_lookup(value=cached_tensor, kind=native_kind):
            return value._storage_for(kind)

        benchmarks.append(BenchmarkCase(
            name=f"storage.cached_lookup/{size}",
            run=cached_lookup,
            validate=lambda run=cached_lookup: run(),
            work_items=1,
            description="lookup of an already cached native representation",
            layer="storage",
            backends=_ACCELERATED,
        ))

        def first_public_operation(source=host, expected_size=size):
            value = ts.Tensor(source, shape=(expected_size,))
            return value + 1.0

        def validate_first_operation(
            run=first_public_operation,
            expected_size=size,
        ) -> None:
            result = run()
            assert result.shape == (expected_size,)
            assert result[-1] == 2.25

        benchmarks.append(BenchmarkCase(
            name=f"storage.first_public_operation/{size}",
            run=first_public_operation,
            validate=validate_first_operation,
            work_items=size,
            description="first native operation from uncached Python storage",
            layer="storage",
            backends=_ACCELERATED,
        ))

        native_tensor = ts.full((size,), 1.25)

        def native_to_python(value=native_tensor):
            return convert_storage(value._storage, "python")

        def validate_native_to_python(
            run=native_to_python,
            expected_size=size,
        ) -> None:
            storage = run()
            assert storage.size == expected_size
            assert storage.buffer[-1] == 1.25

        benchmarks.append(BenchmarkCase(
            name=f"storage.{native_kind}_to_python/{size}",
            run=native_to_python,
            validate=validate_native_to_python,
            work_items=size,
            description="uncached native-to-host storage conversion",
            layer="storage",
            backends=selected,
        ))

        if backend == "cuda":
            def cuda_to_numpy(value=native_tensor):
                return convert_storage(value._storage, "numpy")

            def validate_cuda_to_numpy(
                run=cuda_to_numpy,
                expected_size=size,
            ) -> None:
                storage = run()
                assert storage.size == expected_size
                assert float(storage.buffer[-1]) == 1.25

            benchmarks.append(BenchmarkCase(
                name=f"storage.cuda_to_numpy/{size}",
                run=cuda_to_numpy,
                validate=validate_cuda_to_numpy,
                work_items=size,
                description="uncached device-to-NumPy conversion",
                layer="storage",
                backends=_CUDA,
            ))

        def public_materialization(value=native_tensor, expected_size=size):
            fresh = ts.Tensor(
                value._storage.copy(),
                dtype=value.dtype,
                shape=value.shape,
            )
            result = fresh.tolist()
            assert len(result) == expected_size
            return result

        benchmarks.append(BenchmarkCase(
            name=f"storage.public_tolist/{size}",
            run=public_materialization,
            validate=lambda run=public_materialization: run(),
            work_items=size,
            description="public host materialization from fresh native storage",
            layer="storage",
            backends=_ACCELERATED,
        ))

        def mutation_roundtrip(value=native_tensor):
            fresh = ts.Tensor(
                value._storage.copy(),
                dtype=value.dtype,
                shape=value.shape,
            )
            fresh[0] = 2.0
            return fresh + 1.0

        def validate_mutation_roundtrip(run=mutation_roundtrip) -> None:
            result = run()
            assert result[0] == 3.0
            assert result[-1] == 2.25

        benchmarks.append(BenchmarkCase(
            name=f"storage.mutation_roundtrip/{size}",
            run=mutation_roundtrip,
            validate=validate_mutation_roundtrip,
            work_items=size,
            description="native copy, host mutation, cache invalidation, and reuse",
            layer="storage",
            backends=_ACCELERATED,
        ))

        def conversion_roundtrip(
            source=host,
            target_kind=native_kind,
        ):
            native = convert_storage(source, target_kind)
            return convert_storage(native, "python")

        def validate_conversion_roundtrip(
            run=conversion_roundtrip,
            expected_size=size,
        ) -> None:
            storage = run()
            assert storage.size == expected_size
            assert storage.buffer[-1] == 1.25

        benchmarks.append(BenchmarkCase(
            name=f"storage.conversion_roundtrip/{size}",
            run=conversion_roundtrip,
            validate=validate_conversion_roundtrip,
            work_items=2 * size,
            description="uncached host-to-native-to-host storage conversion cycle",
            layer="storage",
            backends=selected,
        ))

        def persistent_state(
            expected_size=size,
            kind=native_kind,
        ):
            value = ts.full((expected_size,), 1.25)
            value._storage_for(kind)
            return value

        persistent_holder = [persistent_state()]

        def reset_persistent(
            holder=persistent_holder,
            create=persistent_state,
        ) -> None:
            holder[0] = create()

        def mutation_invalidation(
            holder=persistent_holder,
            kind=native_kind,
        ):
            value = holder[0]
            value[0] = value[0] + 1.0
            return value._storage_for(kind)

        def validate_mutation_invalidation(
            run=mutation_invalidation,
            expected_size=size,
        ) -> None:
            storage = run()
            assert storage.size == expected_size
            assert float(storage.buffer[-1]) == 1.25

        benchmarks.append(BenchmarkCase(
            name=f"storage.mutation_invalidation/{size}",
            run=mutation_invalidation,
            validate=validate_mutation_invalidation,
            work_items=size,
            description="in-place mutation followed by a fresh native conversion",
            layer="storage",
            backends=selected,
            reset=reset_persistent,
        ))

        if backend == "cuda":
            def host_device_roundtrip(value=native_tensor):
                host_view = convert_storage(value._storage, "python")
                return convert_storage(host_view, "cuda")

            def validate_host_device_roundtrip(
                run=host_device_roundtrip,
                expected_size=size,
            ) -> None:
                storage = run()
                assert storage.size == expected_size
                assert float(storage.buffer[-1]) == 1.25

            benchmarks.append(BenchmarkCase(
                name=f"storage.host_device_roundtrip/{size}",
                run=host_device_roundtrip,
                validate=validate_host_device_roundtrip,
                work_items=2 * size,
                description="uncached device-to-host-to-device transfer cycle",
                layer="storage",
                backends=_CUDA,
            ))

    return benchmarks
