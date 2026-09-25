"""Selected-backend execution for binary and multiclass cross-entropy."""

from __future__ import annotations

import math
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy

import tensors as ts
import tensors.backend.numpy.kernels as numpy_backend
import tensors.backend.python.kernels as python_backend
from tensors.backend import dispatch as backend_dispatch
from tensors.backend.config import (
    BackendMismatchError,
    BackendOperationUnsupportedError,
)
from tensors.graph.state import reset_graph_state


class LossBackendExecutionTests(unittest.TestCase):
    BACKENDS = ("python", "numpy", "cuda")

    def setUp(self):
        self.previous_backend = ts.get_backend()
        reset_graph_state()

    def tearDown(self):
        ts.set_backend(self.previous_backend)
        reset_graph_state()

    def for_each_backend(self, body):
        for backend in self.BACKENDS:
            with self.subTest(backend=backend):
                if backend not in ts.available_backends():
                    continue
                reset_graph_state()
                with ts.use_backend(backend):
                    body(backend)

    def assert_values_close(self, actual, expected, places=12):
        self.assertEqual(len(actual), len(expected))
        for produced, reference in zip(actual, expected):
            if math.isnan(reference):
                self.assertTrue(math.isnan(produced))
            elif math.isinf(reference):
                self.assertEqual(produced, reference)
            else:
                self.assertAlmostEqual(produced, reference, places=places)

    def test_small_and_large_forwards_stay_on_selected_backend(self):
        def body(backend):
            for dtype, places in ((ts.float32, 6), (ts.float64, 12)):
                probabilities = ts.Tensor([0.25, 0.75], dtype=dtype)
                targets = ts.Tensor([0.0, 1.0], dtype=dtype)
                logits = ts.Tensor([[2.0, 0.0], [0.0, 2.0]], dtype=dtype)
                classes = ts.Tensor([0, 1], dtype=ts.int64)
                for reduction in ("none", "mean", "sum"):
                    binary = ts.binary_cross_entropy(
                        probabilities, targets, reduction=reduction
                    )
                    binary_logits = ts.binary_cross_entropy(
                        ts.Tensor([2.0, -2.0], dtype=dtype),
                        targets,
                        from_logits=True,
                        reduction=reduction,
                    )
                    multiclass = ts.cross_entropy(logits, classes, reduction=reduction)
                    for result in (binary, binary_logits, multiclass):
                        self.assertEqual(result.backend_storage.kind, backend)
                        self.assertIs(result.dtype, dtype)
                self.assertAlmostEqual(
                    ts.binary_cross_entropy(probabilities, targets).item(),
                    (-math.log(0.75) - math.log(0.75)) / 2.0,
                    places=places,
                )

                large_probabilities = ts.Tensor(
                    [0.25 if index % 2 == 0 else 0.75 for index in range(4096)],
                    dtype=dtype,
                )
                large_targets = ts.Tensor(
                    [0.0 if index % 2 == 0 else 1.0 for index in range(4096)],
                    dtype=dtype,
                )
                large = ts.binary_cross_entropy(
                    large_probabilities, large_targets, reduction="none"
                )
                self.assertEqual(large.backend_storage.kind, backend)
                self.assertEqual(large.shape, (4096,))
                large_logits = ts.Tensor(
                    [float(index % 13) - 6.0 for index in range(4096)],
                    dtype=dtype,
                    shape=(64, 64),
                )
                large_classes = ts.Tensor(
                    [index % 64 for index in range(64)], dtype=ts.int64
                )
                large_multiclass = ts.cross_entropy(
                    large_logits, large_classes, reduction="none"
                )
                self.assertEqual(large_multiclass.backend_storage.kind, backend)
                self.assertEqual(large_multiclass.shape, (64,))

        self.for_each_backend(body)

    def test_broadcasting_dense_targets_axes_and_reductions(self):
        def body(backend):
            probabilities = ts.Variable(ts.Tensor([[0.8], [0.25]], dtype=ts.float64))
            targets = ts.Variable(ts.Tensor([[1.0, 0.0]], dtype=ts.float64))
            loss = ts.binary_cross_entropy(probabilities, targets, reduction="sum")
            prediction_gradient, target_gradient = ts.grad(
                loss, (probabilities, targets)
            )
            self.assertEqual(prediction_gradient.shape, (2, 1))
            self.assertEqual(target_gradient.shape, (1, 2))
            self.assertEqual(prediction_gradient.backend_storage.kind, backend)
            self.assertEqual(target_gradient.backend_storage.kind, backend)

            logits = ts.Variable(
                ts.Tensor(
                    [
                        [[2.0, 0.0, 1.0], [0.0, 2.0, 1.0]],
                        [[1.0, 3.0, 0.0], [2.0, 1.0, 4.0]],
                    ],
                    dtype=ts.float64,
                )
            )
            dense = ts.Tensor([[0.25], [0.75]], dtype=ts.float64)
            losses = ts.cross_entropy(logits, dense, axis=1, reduction="none")
            total = ts.cross_entropy(logits, dense, axis=1, reduction="sum")
            gradient = ts.grad(total, logits)
            self.assertEqual(losses.shape, (2, 3))
            self.assertEqual(losses.data.backend_storage.kind, backend)
            self.assertEqual(gradient.backend_storage.kind, backend)

            dense_variable = ts.Variable(ts.Tensor([[0.25, 0.75]], dtype=ts.float64))
            dense_loss = ts.cross_entropy(
                ts.Tensor([[2.0, -1.0]], dtype=ts.float64),
                dense_variable,
                reduction="sum",
            )
            dense_gradient = ts.grad(dense_loss, dense_variable)
            self.assertEqual(dense_gradient.shape, dense_variable.shape)
            self.assertEqual(dense_gradient.backend_storage.kind, backend)
            with self.assertRaisesRegex(TypeError, "Class-index targets"):
                ts.cross_entropy(
                    ts.Tensor([[1.0, -1.0]]),
                    ts.Variable(ts.Tensor([0], dtype=ts.int64), requires_grad=False),
                )

            direct = backend_dispatch.execute_cross_entropy_gradient(
                ts.Tensor([1.0], dtype=ts.float64),
                ts.Tensor([[0.0, -20.0]], dtype=ts.float32),
                ts.Tensor([[1.0, 0.0]], dtype=ts.float32),
                1,
                reduction="sum",
                needs_input_grad=(True, False),
            )[0]
            self.assertIsNotNone(direct)
            self.assertIs(direct.dtype, ts.float64)
            self.assertLess(direct.buffer[0], 0.0)
            self.assertGreater(direct.buffer[1], 0.0)

        self.for_each_backend(body)

    def test_bce_boundaries_extremes_and_validation(self):
        def body(backend):
            probabilities = ts.Variable(ts.Tensor([0.0, 1.0], dtype=ts.float64))
            boundary = ts.binary_cross_entropy(
                probabilities, [0.0, 1.0], reduction="sum"
            )
            gradient = ts.grad(boundary, probabilities)
            self.assertEqual(boundary.data.item(), 0.0)
            self.assert_values_close(gradient.tolist(), [1.0, -1.0])

            logits = ts.Variable(
                ts.Tensor(
                    [1000.0, -1000.0, math.inf, -math.inf],
                    dtype=ts.float64,
                )
            )
            extreme = ts.binary_cross_entropy(
                logits,
                [1.0, 0.0, 1.0, 0.0],
                from_logits=True,
                reduction="none",
            )
            extreme_gradient = ts.grad(ts.sum(extreme), logits)
            self.assert_values_close(extreme.data.tolist(), [0.0, 0.0, 0.0, 0.0])
            self.assert_values_close(extreme_gradient.tolist(), [0.0, 0.0, 0.0, 0.0])
            nan_loss = ts.binary_cross_entropy([math.nan], [0.5], from_logits=True)
            self.assertTrue(math.isnan(nan_loss.item()))
            self.assertTrue(math.isnan(ts.binary_cross_entropy([], []).item()))
            empty_logits = ts.Tensor([], shape=(0, 2))
            empty_classes = ts.Tensor([], dtype=ts.int64)
            self.assertTrue(
                math.isnan(ts.cross_entropy(empty_logits, empty_classes).item())
            )
            for from_logits in (False, True):
                with self.assertRaisesRegex(ValueError, "between 0 and 1"):
                    ts.binary_cross_entropy([0.5], [1.5], from_logits=from_logits)

        self.for_each_backend(body)

    def test_cross_entropy_extremes_nan_and_dominant_vjp(self):
        def body(backend):
            logits = ts.Variable(ts.Tensor([[0.0, -40.0]], dtype=ts.float64))
            loss = ts.cross_entropy(
                logits,
                ts.Tensor([[1.0, 0.0]], dtype=ts.float64),
                reduction="sum",
            )
            gradient = ts.grad(loss, logits)
            tail = math.exp(-40.0) / (1.0 + math.exp(-40.0))
            self.assertLess(gradient.tolist()[0], 0.0)
            self.assertGreater(gradient.tolist()[1], 0.0)
            self.assertAlmostEqual(-gradient.tolist()[0] / tail, 1.0)
            self.assertAlmostEqual(gradient.tolist()[1] / tail, 1.0)
            self.assertEqual(gradient.backend_storage.kind, backend)

            infinity = ts.cross_entropy(
                ts.Tensor([[math.inf, 1.0, math.inf]], dtype=ts.float64),
                ts.Tensor([[0.5, 0.0, 0.5]], dtype=ts.float64),
                reduction="sum",
            )
            self.assertAlmostEqual(infinity.item(), math.log(2.0))
            nan = ts.cross_entropy(
                ts.Tensor([[math.nan, 0.0]], dtype=ts.float64),
                ts.Tensor([[1.0, 0.0]], dtype=ts.float64),
                reduction="sum",
            )
            self.assertTrue(math.isnan(nan.item()))

        self.for_each_backend(body)

    def test_graph_replay_and_supported_higher_order_paths(self):
        def body(backend):
            @ts.Graph
            def model(logits, targets):
                return ts.binary_cross_entropy(
                    logits, targets, from_logits=True, reduction="sum"
                )

            first_run = model(ts.Tensor([0.0, 1.0]), ts.Tensor([1.0, 0.0]))
            second_run = model(ts.Tensor([-1.0, 2.0]), ts.Tensor([0.0, 1.0]))
            self.assertEqual(first_run.data.backend_storage.kind, backend)
            self.assertEqual(second_run.data.backend_storage.kind, backend)

            @ts.Graph
            def multiclass_model(logits, targets):
                return ts.cross_entropy(logits, targets, reduction="sum")

            multiclass_first = multiclass_model(
                ts.Tensor([[1.0, -1.0]]), ts.Tensor([[1.0, 0.0]])
            )
            multiclass_second = multiclass_model(
                ts.Tensor([[-1.0, 1.0]]), ts.Tensor([[0.0, 1.0]])
            )
            self.assertEqual(multiclass_first.data.backend_storage.kind, backend)
            self.assertEqual(multiclass_second.data.backend_storage.kind, backend)

            value = ts.Variable(ts.Tensor([0.0], dtype=ts.float64))
            binary = ts.binary_cross_entropy(value, ts.Tensor([1.0]), from_logits=True)
            first = ts.grad(binary, value, create_graph=True)
            second = ts.grad(first, value)
            self.assertAlmostEqual(second.item(), 0.25)
            self.assertEqual(first.data.backend_storage.kind, backend)
            self.assertEqual(second.backend_storage.kind, backend)

            reset_graph_state()
            logits = ts.Variable(ts.Tensor([[0.0, 0.0]], dtype=ts.float64))
            multiclass = ts.cross_entropy(logits, [0])
            first = ts.grad(multiclass, logits, create_graph=True)
            second = ts.grad(first[0, 0], logits)
            self.assert_values_close(second.tolist(), [0.25, -0.25])
            self.assertEqual(first.data.backend_storage.kind, backend)
            self.assertEqual(second.backend_storage.kind, backend)

        self.for_each_backend(body)

    @unittest.skipUnless("numpy" in ts.available_backends(), "NumPy unavailable")
    def test_numpy_kernels_receive_native_arrays_without_python_fallback(self):
        names = (
            "binary_cross_entropy",
            "binary_cross_entropy_gradient",
            "cross_entropy",
            "cross_entropy_gradient",
        )
        originals = {name: getattr(numpy_backend, name) for name in names}

        def native_spy(name):
            def call(*arguments, **keywords):
                for argument in arguments[:3]:
                    if not isinstance(argument, tuple):
                        self.assertIsInstance(argument, numpy.ndarray)
                return originals[name](*arguments, **keywords)

            return call

        with ts.use_backend("numpy"):
            with (
                patch.object(
                    python_backend,
                    "binary_cross_entropy",
                    side_effect=AssertionError,
                ),
                patch.object(
                    python_backend,
                    "binary_cross_entropy_gradient",
                    side_effect=AssertionError,
                ),
                patch.object(
                    python_backend,
                    "cross_entropy",
                    side_effect=AssertionError,
                ),
                patch.object(
                    python_backend,
                    "cross_entropy_gradient",
                    side_effect=AssertionError,
                ),
            ):
                with (
                    patch.object(
                        numpy_backend,
                        "binary_cross_entropy",
                        side_effect=native_spy("binary_cross_entropy"),
                    ),
                    patch.object(
                        numpy_backend,
                        "binary_cross_entropy_gradient",
                        side_effect=native_spy("binary_cross_entropy_gradient"),
                    ),
                    patch.object(
                        numpy_backend,
                        "cross_entropy",
                        side_effect=native_spy("cross_entropy"),
                    ),
                    patch.object(
                        numpy_backend,
                        "cross_entropy_gradient",
                        side_effect=native_spy("cross_entropy_gradient"),
                    ),
                ):
                    prediction = ts.Variable(ts.Tensor([0.25, 0.75]))
                    target = ts.Tensor([0.0, 1.0])
                    binary = ts.binary_cross_entropy(
                        prediction, target, reduction="sum"
                    )
                    ts.grad(binary, prediction)
                    logits = ts.Variable(ts.Tensor([[1.0, -1.0]]))
                    dense = ts.Tensor([[1.0, 0.0]])
                    multiclass = ts.cross_entropy(logits, dense, reduction="sum")
                    ts.grad(multiclass, logits)

    @unittest.skipUnless("cuda" in ts.available_backends(), "CUDA unavailable")
    def test_cuda_kernels_receive_device_arrays_without_python_fallback(self):
        import cupy
        import tensors.backend.cuda.kernels as cuda_backend

        names = (
            "binary_cross_entropy",
            "binary_cross_entropy_gradient",
            "cross_entropy",
            "cross_entropy_gradient",
        )
        originals = {name: getattr(cuda_backend, name) for name in names}

        def native_spy(name):
            def call(*arguments, **keywords):
                for argument in arguments[:3]:
                    if not isinstance(argument, tuple):
                        self.assertIsInstance(argument, cupy.ndarray)
                return originals[name](*arguments, **keywords)

            return call

        with ts.use_backend("cuda"):
            with (
                patch.object(
                    python_backend,
                    "binary_cross_entropy",
                    side_effect=AssertionError,
                ),
                patch.object(
                    python_backend,
                    "binary_cross_entropy_gradient",
                    side_effect=AssertionError,
                ),
                patch.object(
                    python_backend,
                    "cross_entropy",
                    side_effect=AssertionError,
                ),
                patch.object(
                    python_backend,
                    "cross_entropy_gradient",
                    side_effect=AssertionError,
                ),
            ):
                with (
                    patch.object(
                        cuda_backend,
                        "binary_cross_entropy",
                        side_effect=native_spy("binary_cross_entropy"),
                    ),
                    patch.object(
                        cuda_backend,
                        "binary_cross_entropy_gradient",
                        side_effect=native_spy("binary_cross_entropy_gradient"),
                    ),
                    patch.object(
                        cuda_backend,
                        "cross_entropy",
                        side_effect=native_spy("cross_entropy"),
                    ),
                    patch.object(
                        cuda_backend,
                        "cross_entropy_gradient",
                        side_effect=native_spy("cross_entropy_gradient"),
                    ),
                ):
                    prediction = ts.Variable(ts.Tensor([0.25, 0.75]))
                    target = ts.Tensor([0.0, 1.0])
                    binary = ts.binary_cross_entropy(
                        prediction, target, reduction="sum"
                    )
                    ts.grad(binary, prediction)
                    logits = ts.Variable(ts.Tensor([[1.0, -1.0]]))
                    dense = ts.Tensor([[1.0, 0.0]])
                    multiclass = ts.cross_entropy(logits, dense, reduction="sum")
                    ts.grad(multiclass, logits)

    @unittest.skipUnless("numpy" in ts.available_backends(), "NumPy unavailable")
    def test_decline_and_foreign_residency_raise_without_fallback(self):
        with ts.use_backend("numpy"):
            prediction = ts.Tensor([0.25, 0.75])
            target = ts.Tensor([0.0, 1.0])
            logits = ts.Tensor([[1.0, -1.0]])
            dense = ts.Tensor([[1.0, 0.0]])
            grad = ts.Tensor([1.0])
            calls = (
                (
                    "binary_cross_entropy",
                    lambda: ts.binary_cross_entropy(prediction, target),
                ),
                (
                    "binary_cross_entropy_gradient",
                    lambda: backend_dispatch.execute_binary_cross_entropy_gradient(
                        grad,
                        prediction,
                        target,
                        from_logits=False,
                        reduction="mean",
                        needs_input_grad=(True, False),
                    ),
                ),
                (
                    "cross_entropy",
                    lambda: ts.cross_entropy(logits, dense),
                ),
                (
                    "cross_entropy_gradient",
                    lambda: backend_dispatch.execute_cross_entropy_gradient(
                        grad,
                        logits,
                        dense,
                        1,
                        reduction="mean",
                        needs_input_grad=(True, False),
                    ),
                ),
            )
            for name, call in calls:
                with (
                    self.subTest(name=name),
                    patch.object(numpy_backend, name, return_value=None),
                ):
                    with self.assertRaises(BackendOperationUnsupportedError):
                        call()

        with ts.use_backend("python"):
            foreign_prediction = ts.Tensor([0.25, 0.75])
            foreign_target = ts.Tensor([0.0, 1.0])
        with ts.use_backend("numpy"):
            with self.assertRaises(BackendMismatchError):
                backend_dispatch.execute_binary_cross_entropy(
                    foreign_prediction,
                    foreign_target,
                    from_logits=False,
                    reduction="mean",
                    dtype=ts.float64,
                    output_shape=(1,),
                )

    def test_dispatchers_and_kernels_have_strict_native_boundaries(self):
        repository = Path(__file__).resolve().parents[3]
        names = (
            "binary_cross_entropy.py",
            "binary_cross_entropy_gradient.py",
            "cross_entropy.py",
            "cross_entropy_gradient.py",
        )
        for filename in names:
            source = (
                repository / "tensors" / "backend" / "dispatch" / "nn" / filename
            ).read_text(encoding="utf-8")
            for forbidden in (
                "_array_work_is_large_enough",
                "_NUMPY_ELEMENTWISE_MIN_SIZE",
                "_backend_kernel",
                "python.kernels",
                "as reference",
            ):
                self.assertNotIn(forbidden, source)
            self.assertIn("validate_backend_residency", source)
            self.assertIn("BackendOperationUnsupportedError", source)

        for backend in ("numpy", "cuda"):
            for filename in names:
                source = (
                    repository
                    / "tensors"
                    / "backend"
                    / backend
                    / "kernels"
                    / "nn"
                    / filename
                ).read_text(encoding="utf-8")
                for forbidden in (
                    "Tensor",
                    "tensor_to_logical_array",
                    "_working_values",
                    "._data",
                    "asnumpy",
                    ".get(",
                ):
                    self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
