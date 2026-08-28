"""Cross-entropy preprocessing, kernel, forward, and backward benchmarks."""

from __future__ import annotations

import importlib
import math

import tensors as ts
from tensors.math.cross_entropy import _one_hot_targets

from .runner import BenchmarkBackend, BenchmarkCase


_ACCELERATED = frozenset[BenchmarkBackend]({"numpy", "cuda"})


def _shape_cases(
    samples: int,
    classes: int,
    backends: frozenset[BenchmarkBackend] | None,
) -> list[BenchmarkCase]:
    shape = (samples, classes)
    logits = ts.reshape(
        ts.linspace(-2.0, 2.0, samples * classes),
        shape,
    )
    indices = ts.Tensor(
        [sample % classes for sample in range(samples)],
        dtype=ts.int64,
    )
    dense = ts.full(shape, 1.0 / classes)
    prefix = f"{samples}x{classes}"
    benchmarks: list[BenchmarkCase] = []

    def prepare_targets() -> ts.Tensor:
        return _one_hot_targets(logits, indices, 1)

    def validate_preparation() -> None:
        result = prepare_targets()
        assert result.shape == shape
        assert result[0, 0] == 1.0

    benchmarks.append(BenchmarkCase(
        name=f"loss.cross_entropy_target_preparation/{prefix}",
        run=prepare_targets,
        validate=validate_preparation,
        work_items=samples * classes,
        description="class-index to one-hot target preparation",
        backends=backends,
    ))

    def class_index_forward() -> ts.Tensor:
        return ts.cross_entropy(logits, indices)

    def dense_forward() -> ts.Tensor:
        return ts.cross_entropy(logits, dense)

    def validate_loss(run) -> None:
        result = run()
        assert result.shape == (1,)
        assert math.isfinite(float(result.item()))

    benchmarks.extend((
        BenchmarkCase(
            name=f"loss.cross_entropy_indices/{prefix}",
            run=class_index_forward,
            validate=lambda: validate_loss(class_index_forward),
            work_items=samples * classes,
            description="public cross-entropy including class-index preparation",
            backends=backends,
        ),
        BenchmarkCase(
            name=f"loss.cross_entropy_dense/{prefix}",
            run=dense_forward,
            validate=lambda: validate_loss(dense_forward),
            work_items=samples * classes,
            description="public cross-entropy with preconstructed dense targets",
            backends=backends,
        ),
    ))

    backend = ts.get_backend()
    if backend in _ACCELERATED:
        kernels = importlib.import_module(f"tensors.backend.{backend}")

        def dense_kernel():
            return kernels.cross_entropy(
                logits,
                dense,
                1,
                reduction="mean",
                dtype=ts.float64,
                output_shape=(1,),
            )

        def validate_dense_kernel() -> None:
            storage = dense_kernel()
            assert storage is not None
            assert storage.size == 1
            value = storage.buffer[0]
            if hasattr(value, "item"):
                value = value.item()
            assert math.isfinite(float(value))

        benchmarks.append(BenchmarkCase(
            name=f"kernel.cross_entropy_dense/{prefix}",
            run=dense_kernel,
            validate=validate_dense_kernel,
            work_items=samples * classes,
            description="guarded dense cross-entropy backend kernel",
            layer="kernel",
            backends=_ACCELERATED,
        ))

    def forward_backward():
        variable = ts.Variable(logits, name="loss_logits")
        loss = ts.cross_entropy(variable, dense)
        ts.backward(loss)
        return variable.grad

    def validate_forward_backward() -> None:
        result = forward_backward()
        assert isinstance(result, ts.Tensor)
        assert result.shape == shape

    benchmarks.append(BenchmarkCase(
        name=f"loss.cross_entropy_forward_backward/{prefix}",
        run=forward_backward,
        validate=validate_forward_backward,
        work_items=samples * classes,
        gc_enabled=True,
        description="fresh public cross-entropy forward and reverse pass",
        layer="autograd",
        backends=backends,
    ))
    return benchmarks


def cases() -> list[BenchmarkCase]:
    """Separate Python target work from dense kernels and differentiation."""
    benchmarks = _shape_cases(128, 64, None)
    if ts.get_backend() != "python":
        benchmarks.extend(_shape_cases(1024, 256, _ACCELERATED))
    return benchmarks
