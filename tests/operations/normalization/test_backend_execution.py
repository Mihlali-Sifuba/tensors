"""Selected-backend execution for softmax and log-softmax."""

from __future__ import annotations

from pathlib import Path
import math
import unittest
from unittest.mock import patch

import numpy

import tensors as ts
import tensors.backend.numpy.kernels as numpy_backend
import tensors.backend.python.kernels as python_backend
import tensors.tensor as tensor_module
from tensors.backend import dispatch as backend_dispatch
from tensors.backend.config import (
    BackendMismatchError,
    BackendOperationUnsupportedError,
)
from tensors.graph.state import reset_graph_state


class HostReadCounter:
    """Count accesses to the Tensor host-value compatibility property."""

    def __init__(self):
        self.reads = 0
        self._original = tensor_module.Tensor._data.fget

    def __enter__(self):
        counter = self

        def counting(instance):
            counter.reads += 1
            return counter._original(instance)

        self._patch = patch.object(tensor_module.Tensor, "_data", property(counting))
        self._patch.__enter__()
        return self

    def __exit__(self, *arguments):
        return self._patch.__exit__(*arguments)


class NormalizationBackendExecutionTests(unittest.TestCase):
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
            else:
                self.assertAlmostEqual(produced, reference, places=places)

    def test_small_and_large_forwards_stay_on_the_selected_backend(self):
        def body(backend):
            for dtype, places in ((ts.float32, 6), (ts.float64, 12)):
                small = ts.Tensor([[1000.0, 1001.0], [-3.0, 2.0]], dtype=dtype)
                probabilities = ts.softmax(small, axis=-1)
                log_probabilities = ts.log_softmax(small, axis=1)
                large = ts.softmax(
                    ts.Tensor(
                        [float(index % 9) - 4.0 for index in range(4096)],
                        dtype=dtype,
                        shape=(64, 64),
                    ),
                    axis=1,
                )
                self.assertEqual(probabilities.backend_storage.kind, backend)
                self.assertEqual(log_probabilities.backend_storage.kind, backend)
                self.assertEqual(large.backend_storage.kind, backend)
                self.assertIs(probabilities.dtype, dtype)
                self.assertIs(log_probabilities.dtype, dtype)
                self.assertEqual(probabilities.shape, (2, 2))
                self.assertEqual(log_probabilities.shape, (2, 2))
                for start in range(0, large.size, 64):
                    self.assertAlmostEqual(
                        sum(large.tolist()[start : start + 64]), 1.0, places=places
                    )
                for probability, log_probability in zip(
                    probabilities.tolist(), log_probabilities.tolist()
                ):
                    self.assertAlmostEqual(
                        math.exp(log_probability), probability, places=places
                    )

        self.for_each_backend(body)

    def test_axes_multidimensional_values_and_integer_promotion(self):
        def body(backend):
            values = ts.Tensor(
                [1.0, -2.0, 3.0, 4.0, 0.5, -1.5, 2.5, -3.5, 5.0, 0.0, -4.0, 1.5],
                shape=(2, 2, 3),
            )
            along_zero = ts.softmax(values, axis=0)
            along_middle = ts.log_softmax(values, axis=-2)
            integer = ts.softmax(ts.Tensor([[1, 2, 3]], dtype=ts.int32), axis=1)
            self.assertEqual(along_zero.shape, values.shape)
            self.assertEqual(along_middle.shape, values.shape)
            self.assertIs(integer.dtype, ts.float64)
            self.assertEqual(integer.backend_storage.kind, backend)
            flat = along_zero.tolist()
            for offset in range(6):
                self.assertAlmostEqual(flat[offset] + flat[6 + offset], 1.0)

        self.for_each_backend(body)

    def test_exceptional_forward_values_keep_the_reference_semantics(self):
        def body(backend):
            values = ts.Tensor([[math.inf, 1.0, math.inf], [math.nan, math.inf, 0.0]])
            probabilities = ts.softmax(values, axis=1).tolist()
            log_probabilities = ts.log_softmax(values, axis=1).tolist()
            self.assert_values_close(probabilities[:3], [0.5, 0.0, 0.5])
            self.assert_values_close(
                log_probabilities[:3], [-math.log(2.0), -math.inf, -math.log(2.0)]
            )
            self.assertTrue(all(math.isnan(item) for item in probabilities[3:]))
            self.assertTrue(all(math.isnan(item) for item in log_probabilities[3:]))
            with self.assertRaisesRegex(ValueError, "every value"):
                ts.softmax(ts.Tensor([[-math.inf, -math.inf]]), axis=1)
            with self.assertRaisesRegex(ValueError, "every value"):
                ts.log_softmax(ts.Tensor([[-math.inf, -math.inf]]), axis=1)

        self.for_each_backend(body)

    def test_first_order_vjps_stay_resident_and_match_their_rules(self):
        def body(backend):
            logits = ts.Variable(ts.Tensor([[0.2, -0.4, 0.7]], dtype=ts.float64))
            upstream = ts.Tensor([[1.5, -2.0, 0.25]], dtype=ts.float64)
            probabilities = ts.softmax(logits, axis=-1)
            softmax_gradient = ts.grad(probabilities, logits, grad_outputs=upstream)
            expected_probability = probabilities.data.tolist()
            expectation = sum(
                weight * probability
                for weight, probability in zip(upstream.tolist(), expected_probability)
            )
            expected_softmax = [
                probability * (weight - expectation)
                for weight, probability in zip(upstream.tolist(), expected_probability)
            ]
            reset_graph_state()
            logits = ts.Variable(ts.Tensor([[0.2, -0.4, 0.7]], dtype=ts.float64))
            log_probabilities = ts.log_softmax(logits, axis=1)
            log_gradient = ts.grad(log_probabilities, logits, grad_outputs=upstream)
            total = sum(upstream.tolist())
            expected_log = [
                weight - probability * total
                for weight, probability in zip(upstream.tolist(), expected_probability)
            ]
            self.assert_values_close(softmax_gradient.tolist(), expected_softmax)
            self.assert_values_close(log_gradient.tolist(), expected_log)
            self.assertEqual(softmax_gradient.backend_storage.kind, backend)
            self.assertEqual(log_gradient.backend_storage.kind, backend)

            large_value = ts.Tensor(
                [float(index % 11) - 5.0 for index in range(4096)],
                shape=(64, 64),
            )
            large_upstream = ts.Tensor(
                [float(index % 7) - 3.0 for index in range(4096)],
                shape=(64, 64),
            )
            large_softmax_gradient = backend_dispatch.execute_softmax_gradient(
                large_upstream, large_value, 1
            )
            large_log_gradient = backend_dispatch.execute_log_softmax_gradient(
                large_upstream, large_value, 1
            )
            self.assertEqual(large_softmax_gradient.kind, backend)
            self.assertEqual(large_log_gradient.kind, backend)
            self.assertEqual(large_softmax_gradient.size, 4096)
            self.assertEqual(large_log_gradient.size, 4096)

        self.for_each_backend(body)

    def test_dominant_probability_vjps_preserve_the_small_tail(self):
        def body(backend):
            value = ts.Tensor([[0.0, -40.0]], dtype=ts.float64)
            upstream = ts.Tensor([[1.0, 0.0]], dtype=ts.float64)
            softmax_gradient = backend_dispatch.execute_softmax_gradient(
                upstream, value, 1
            )
            log_gradient = backend_dispatch.execute_log_softmax_gradient(
                upstream, value, 1
            )
            tail = math.exp(-40.0) / (1.0 + math.exp(-40.0))
            self.assertGreater(softmax_gradient.buffer[0], 0.0)
            self.assertLess(softmax_gradient.buffer[1], 0.0)
            self.assertGreater(log_gradient.buffer[0], 0.0)
            self.assertLess(log_gradient.buffer[1], 0.0)
            self.assertAlmostEqual(softmax_gradient.buffer[0] / tail, 1.0)
            self.assertAlmostEqual(-softmax_gradient.buffer[1] / tail, 1.0)
            self.assertAlmostEqual(log_gradient.buffer[0] / tail, 1.0)
            self.assertAlmostEqual(-log_gradient.buffer[1] / tail, 1.0)

        self.for_each_backend(body)

    def test_graph_replay_and_supported_higher_order_paths_remain_resident(self):
        def body(backend):
            @ts.Graph
            def model(value):
                return ts.log_softmax(value, axis=-1)

            first_run = model(ts.Tensor([[1.0, 2.0]], dtype=ts.float64))
            second_run = model(ts.Tensor([[3.0, 1.0]], dtype=ts.float64))
            self.assertEqual(first_run.data.backend_storage.kind, backend)
            self.assertEqual(second_run.data.backend_storage.kind, backend)

            for function in (ts.softmax, ts.log_softmax):
                reset_graph_state()
                value = ts.Variable(ts.Tensor([[0.2, -0.4]], dtype=ts.float64))
                output = function(value, axis=1)
                first = ts.grad(
                    output,
                    value,
                    grad_outputs=ts.Tensor([[1.0, 1.0]]),
                    create_graph=True,
                )
                second = ts.grad(
                    first,
                    value,
                    grad_outputs=ts.Tensor([[1.0, 1.0]]),
                )
                self.assert_values_close(second.tolist(), [0.0, 0.0])
                self.assertEqual(first.data.backend_storage.kind, backend)
                self.assertEqual(second.backend_storage.kind, backend)

        self.for_each_backend(body)

    def test_noncompact_inputs_are_lowered_without_tensor_host_reads(self):
        def body(backend):
            base = ts.Tensor([0.2, -0.4, 1.5, 0.7, -1.0, 2.0])
            view = ts.Tensor._from_metadata(
                base.backend_storage,
                shape=(2, 3),
                strides=(1, 2),
            )
            with HostReadCounter() as counter:
                probabilities = ts.softmax(view, axis=1)
                log_probabilities = ts.log_softmax(view, axis=-1)
                softmax_gradient = backend_dispatch.execute_softmax_gradient(
                    view, view, 1
                )
                log_gradient = backend_dispatch.execute_log_softmax_gradient(
                    view, view, 1
                )
            self.assertEqual(counter.reads, 0)
            self.assertEqual(probabilities.backend_storage.kind, backend)
            self.assertEqual(log_probabilities.backend_storage.kind, backend)
            self.assertEqual(softmax_gradient.kind, backend)
            self.assertEqual(log_gradient.kind, backend)
            for start in (0, 3):
                self.assertAlmostEqual(
                    sum(probabilities.tolist()[start : start + 3]), 1.0
                )

        self.for_each_backend(body)

    @unittest.skipUnless("numpy" in ts.available_backends(), "NumPy unavailable")
    def test_numpy_kernels_receive_native_arrays_without_python_fallback(self):
        originals = {
            name: getattr(numpy_backend, name)
            for name in (
                "softmax",
                "log_softmax",
                "softmax_gradient",
                "log_softmax_gradient",
            )
        }

        def native_spy(name):
            def call(*arguments, **keywords):
                self.assertIsInstance(arguments[0], numpy.ndarray)
                if "gradient" in name:
                    self.assertIsInstance(arguments[1], numpy.ndarray)
                return originals[name](*arguments, **keywords)

            return call

        with ts.use_backend("numpy"):
            with (
                patch.object(python_backend, "softmax", side_effect=AssertionError),
                patch.object(python_backend, "log_softmax", side_effect=AssertionError),
                patch.object(
                    python_backend, "softmax_gradient", side_effect=AssertionError
                ),
                patch.object(
                    python_backend, "log_softmax_gradient", side_effect=AssertionError
                ),
            ):
                with (
                    patch.object(
                        numpy_backend, "softmax", side_effect=native_spy("softmax")
                    ),
                    patch.object(
                        numpy_backend,
                        "log_softmax",
                        side_effect=native_spy("log_softmax"),
                    ),
                    patch.object(
                        numpy_backend,
                        "softmax_gradient",
                        side_effect=native_spy("softmax_gradient"),
                    ),
                    patch.object(
                        numpy_backend,
                        "log_softmax_gradient",
                        side_effect=native_spy("log_softmax_gradient"),
                    ),
                ):
                    value = ts.Tensor([[1.0, 2.0]])
                    grad = ts.Tensor([[0.5, -0.25]])
                    ts.softmax(value, axis=1)
                    ts.log_softmax(value, axis=1)
                    backend_dispatch.execute_softmax_gradient(grad, value, 1)
                    backend_dispatch.execute_log_softmax_gradient(grad, value, 1)

    @unittest.skipUnless("numpy" in ts.available_backends(), "NumPy unavailable")
    def test_decline_and_foreign_residency_raise_instead_of_falling_back(self):
        with ts.use_backend("numpy"):
            value = ts.Tensor([[1.0, 2.0]])
            grad = ts.Tensor([[0.5, -0.25]])
            calls = (
                ("softmax", lambda: ts.softmax(value, axis=1)),
                ("log_softmax", lambda: ts.log_softmax(value, axis=1)),
                (
                    "softmax_gradient",
                    lambda: backend_dispatch.execute_softmax_gradient(grad, value, 1),
                ),
                (
                    "log_softmax_gradient",
                    lambda: backend_dispatch.execute_log_softmax_gradient(
                        grad, value, 1
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
            foreign = ts.Tensor([[1.0, 2.0]])
        with ts.use_backend("numpy"):
            with self.assertRaises(BackendMismatchError):
                backend_dispatch.execute_softmax(foreign, 1, dtype=ts.float64)
            with self.assertRaises(BackendMismatchError):
                backend_dispatch.execute_log_softmax(foreign, 1, dtype=ts.float64)

    def test_dispatchers_and_kernels_have_strict_native_boundaries(self):
        repository = Path(__file__).resolve().parents[3]
        for filename in (
            "softmax.py",
            "softmax_gradient.py",
            "log_softmax.py",
            "log_softmax_gradient.py",
        ):
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
            for filename in (
                "softmax.py",
                "softmax_gradient.py",
                "log_softmax.py",
                "log_softmax_gradient.py",
            ):
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
