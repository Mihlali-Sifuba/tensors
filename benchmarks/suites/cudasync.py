"""CUDA-specific costs: launch floor, barriers, transfers, and allocation.

These are the primitives every other CUDA measurement is built out of. A
public operation's host latency can only be interpreted against them: a
call that costs three launch floors plus one barrier is explained, while the
same number without them is just a number.

Every case here is CUDA-only by construction, and says so rather than being
silently absent on the other backends.
"""

from __future__ import annotations
from collections.abc import Sequence
from typing import Any
import tensors as ts
from benchmarks.case import Case, Group, Unsupported
from benchmarks.workloads import CUDA_ONLY, tensor


def _primitive_cases(backend: str) -> list[Case]:
    """Measure the launch floor, barriers, and scalar reads."""
    if backend != "cuda":
        raise Unsupported(
            "these cases measure CUDA launch, synchronization, and transfer primitives, which have no counterpart on the Python or NumPy backends"
        )
    import cupy

    stream = cupy.cuda.get_current_stream()
    device = cupy.cuda.Device()
    tiny = cupy.full(1, 1.5, dtype=cupy.float64)
    small = cupy.full(1_024, 1.5, dtype=cupy.float64)
    common: dict[str, Any] = {
        "family": "cuda/primitive",
        "backends": CUDA_ONLY,
        "elements": 1,
        "tags": {"curve": "cuda-primitive"},
    }
    cases: list[Case] = [
        Case(
            name="cuda.launch_floor",
            run=lambda: cupy.add(tiny, tiny),
            layer="sync",
            validate=lambda: cupy.add(tiny, tiny),
            description="one elementwise kernel on a single element: the launch cost with the arithmetic removed",
            **common,
        ),
        Case(
            name="cuda.stream_synchronize_idle",
            run=stream.synchronize,
            layer="sync",
            validate=stream.synchronize,
            description="synchronize a stream that has nothing outstanding",
            **common,
        ),
        Case(
            name="cuda.device_synchronize_idle",
            run=device.synchronize,
            layer="sync",
            validate=device.synchronize,
            description="synchronize an idle device",
            **common,
        ),
        Case(
            name="cuda.launch_then_synchronize",
            run=lambda: (cupy.add(tiny, tiny), stream.synchronize()),
            layer="sync",
            validate=lambda: (cupy.add(tiny, tiny), stream.synchronize()),
            description="one launch followed by a barrier: the round trip a host-visible result costs",
            **common,
        ),
        Case(
            name="cuda.scalar_read_provider",
            run=lambda: float(small[0]),
            layer="sync",
            validate=lambda: float(small[0]),
            description="read one device value to the host through the provider",
            **common,
        ),
        Case(
            name="cuda.bool_of_reduction",
            run=lambda: bool(cupy.all(small > 0.0)),
            layer="sync",
            validate=lambda: bool(cupy.all(small > 0.0)),
            description="the pattern every numerical guard uses: reduce on the device, then convert the result to a Python bool",
            **common,
        ),
        Case(
            name="cuda.isfinite_all_bool",
            run=lambda: bool(cupy.all(cupy.isfinite(small))),
            layer="sync",
            validate=lambda: bool(cupy.all(cupy.isfinite(small))),
            description="the exact finite-result check the kernels perform, including its host barrier",
            **common,
        ),
    ]
    pool = cupy.get_default_memory_pool()
    for size, label in ((1_024, "4KB"), (1_000_000, "8MB")):
        allocate_common = dict(common)
        allocate_common["elements"] = size
        allocate_common["tags"] = {"curve": "cuda-allocation"}
        cases.append(
            Case(
                name=f"cuda.allocate_pooled/{label}",
                run=lambda size=size: cupy.empty(size, dtype=cupy.float64),
                layer="sync",
                validate=lambda size=size: cupy.empty(size, dtype=cupy.float64),
                description="allocate device memory with a warm pool, which reuses a cached block",
                **allocate_common,
            )
        )

        def allocate_cold(size: int = size) -> Any:
            pool.free_all_blocks()
            return cupy.empty(size, dtype=cupy.float64)

        cases.append(
            Case(
                name=f"cuda.allocate_unpooled/{label}",
                run=allocate_cold,
                layer="sync",
                validate=allocate_cold,
                description="allocate device memory after releasing the pool, so the driver has to serve the request",
                **allocate_common,
            )
        )
    return cases


def _transfer_cases(backend: str, size: int) -> list[Case]:
    """Measure host-to-device and device-to-host transfers."""
    if backend != "cuda":
        raise Unsupported(
            "host/device transfer has no counterpart on the Python or NumPy backends"
        )
    import cupy
    import numpy
    from tensors.backend.cuda.storage import CudaStorage

    host = numpy.full(size, 1.5, dtype=numpy.float64)
    device_array = cupy.full(size, 1.5, dtype=cupy.float64)
    pinned = cupy.cuda.alloc_pinned_memory(host.nbytes)
    pinned_host = numpy.frombuffer(pinned, dtype=numpy.float64, count=size)
    pinned_host[:] = host
    common: dict[str, Any] = {
        "family": "cuda/transfer",
        "backends": CUDA_ONLY,
        "dtype": "float64",
        "shape": (size,),
        "elements": size,
        "work_items": size * 8,
        "tags": {"curve": "cuda-transfer"},
    }
    cases = [
        Case(
            name=f"cuda.host_to_device/{size}",
            run=lambda: cupy.asarray(host),
            layer="sync",
            validate=lambda: cupy.asarray(host),
            description="copy host values to the device",
            **common,
        ),
        Case(
            name=f"cuda.host_to_device_pinned/{size}",
            run=lambda: cupy.asarray(pinned_host),
            layer="sync",
            validate=lambda: cupy.asarray(pinned_host),
            description="copy from pinned host memory to the device",
            **common,
        ),
        Case(
            name=f"cuda.device_to_host/{size}",
            run=lambda: cupy.asnumpy(device_array),
            layer="sync",
            validate=lambda: cupy.asnumpy(device_array),
            description="copy device values back to the host, which necessarily waits for the device",
            **common,
        ),
        Case(
            name=f"cuda.device_to_device/{size}",
            run=device_array.copy,
            layer="sync",
            validate=device_array.copy,
            description="copy within device memory",
            **common,
        ),
    ]
    device_tensor = tensor((size,), dtype_name="float64", kind="constant", value=1.5)
    package_common = dict(common)
    package_common["tags"] = {"curve": "cuda-transfer-package"}
    cases.append(
        Case(
            name=f"cuda.tensor_from_host/{size}",
            run=lambda: ts.Tensor(
                CudaStorage(host, ts.float64), dtype=ts.float64, shape=(size,)
            ),
            layer="storage",
            validate=lambda: ts.Tensor(
                CudaStorage(host, ts.float64), dtype=ts.float64, shape=(size,)
            ),
            description="construct a device Tensor from host NumPy values through device storage, which public construction then copies",
            **package_common,
        )
    )
    cases.append(
        Case(
            name=f"cuda.tensor_to_host/{size}",
            run=device_tensor.tolist,
            layer="storage",
            validate=device_tensor.tolist,
            description="materialize a device Tensor as a Python list, which transfers to the host and then converts element by element",
            **package_common,
        )
    )
    return cases


def _guard_cases(backend: str, size: int) -> list[Case]:
    """Count the barriers a public operation performs, by comparison.

    Each case is a public operation whose kernel is known to include a
    host-visible check. The synchronization probe recorded for every CUDA
    case decides whether it blocks; these exist so that the blocking ones
    can be compared against the primitives above at the same size.
    """
    if backend != "cuda":
        raise Unsupported(
            "these cases exist to attribute CUDA host barriers and have no counterpart on the other backends"
        )
    common: dict[str, Any] = {
        "family": "cuda/guard",
        "backends": CUDA_ONLY,
        "shape": (size,),
        "elements": size,
        "work_items": size,
    }
    cases: list[Case] = []
    for dtype_name in ("float64", "float32"):
        value = tensor((size,), dtype_name=dtype_name, kind="ramp")
        for name, call, description in (
            (
                "add",
                lambda value=value: value + value,
                "elementwise addition, whose float32 path range-checks the narrowed result",
            ),
            ("sum", lambda value=value: ts.sum(value), "a guarded reduction"),
            ("mean", lambda value=value: ts.mean(value), "a guarded mean"),
            ("exp", lambda value=value: ts.exp(value), "a unary transform"),
        ):
            cases.append(
                Case(
                    name=f"cuda.guard_{name}/{dtype_name}/{size}",
                    run=call,
                    layer="public",
                    validate=call,
                    description=description,
                    dtype=dtype_name,
                    family=f"cuda/guard/{name}",
                    tags={
                        "curve": f"cuda-guard-{name}|{dtype_name}",
                        "dtype_pair": f"cuda-guard-{name}",
                    },
                    **{
                        key: item
                        for key, item in common.items()
                        if key not in ("family",)
                    },
                )
            )
    return cases


def groups() -> list[Group]:
    """Return the CUDA primitive, transfer, and guard groups."""
    result: list[Group] = [
        Group(name="cuda/primitives", factory=_primitive_cases, suite="cuda")
    ]
    for size in (1, 1_024, 1_000_000, 10_000_000):

        def transfer(backend: str, size: int = size) -> Sequence[Case]:
            return _transfer_cases(backend, size)

        result.append(
            Group(name=f"cuda/transfer/{size}", factory=transfer, suite="cuda")
        )
    for size in (1_024, 100_000, 1_000_000):

        def guard(backend: str, size: int = size) -> Sequence[Case]:
            return _guard_cases(backend, size)

        result.append(Group(name=f"cuda/guard/{size}", factory=guard, suite="cuda"))
    return result


__all__ = ["groups"]
