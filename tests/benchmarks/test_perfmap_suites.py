"""Regression coverage for the perfmap benchmark suites.

A benchmark suite only reports an error when it is actually run, and a full
run is expensive, so a call that no longer matches its kernel's signature can
sit unnoticed. These tests build the cases and run each one's validation once
at a small size, which is cheap enough to keep in the ordinary suite.
"""

import unittest

import tensors as ts

from benchmarks.perfmap import registry
from benchmarks.perfmap.harness import Unsupported


def _available_backends():
    return ts.available_backends()


class SuiteCaseExecutionTests(unittest.TestCase):
    """Every case a suite builds runs without raising."""

    #: Suites whose cases are cheap enough to run in full here. The heavier
    #: suites are covered by the benchmark run itself.
    SUITES = ("creation", "initializers")

    def setUp(self):
        self.previous_backend = ts.get_backend()

    def tearDown(self):
        ts.set_backend(self.previous_backend)

    def _run_suite(self, suite):
        """Return how many cases ran, raising whatever a broken case raises."""
        executed = 0
        for group in registry.suite_groups(suite):
            for backend in _available_backends():
                try:
                    with ts.use_backend(backend):
                        cases = list(group.factory(backend))
                except Unsupported:
                    continue
                for case in cases:
                    if case.backends is not None and backend not in case.backends:
                        continue
                    if case.elements is not None and case.elements > 10_000:
                        continue
                    with ts.use_backend(backend):
                        if case.setup is not None:
                            case.setup()
                        if case.reset is not None:
                            case.reset()
                        try:
                            if case.validate is not None:
                                case.validate()
                            else:
                                case.run()
                        except Unsupported:
                            continue
                        finally:
                            if case.teardown is not None:
                                case.teardown()
                    executed += 1
        return executed

    def test_suites_build_and_run_every_case(self):
        for suite in self.SUITES:
            with self.subTest(suite=suite):
                executed = self._run_suite(suite)
                self.assertGreater(executed, 0, f"{suite} ran no cases")


class CreationSuiteArgumentTests(unittest.TestCase):
    """The creation suite calls the kernels with the arguments they take.

    ``full`` takes ``(shape, fill_value)``, ``arange`` takes
    ``(start, step, count)``, and neither takes an ``output_shape``. Passing
    the wrong ones raised a TypeError for every case in the suite.
    """

    def setUp(self):
        self.previous_backend = ts.get_backend()

    def tearDown(self):
        ts.set_backend(self.previous_backend)

    def test_creation_cases_call_their_kernels_successfully(self):
        """Each case runs; a kernel returning ``None`` is a valid decline."""
        from benchmarks.perfmap.suites import creation

        produced = 0
        for backend in _available_backends():
            for group in creation.groups():
                try:
                    with ts.use_backend(backend):
                        cases = list(group.factory(backend))
                except Unsupported:
                    continue
                for case in cases:
                    if case.backends is not None and backend not in case.backends:
                        continue
                    if case.elements is not None and case.elements > 10_000:
                        continue
                    with self.subTest(backend=backend, case=case.name):
                        with ts.use_backend(backend):
                            try:
                                result = case.run()
                            except Unsupported:
                                continue
                        if result is not None:
                            produced += 1
        self.assertGreater(produced, 0, "no creation case produced a result")

    def test_creation_kernels_reject_an_output_shape_argument(self):
        """Pin the signatures the suite is written against."""
        from tensors.backend import execute_arange, execute_full, execute_linspace

        with self.assertRaises(TypeError):
            execute_full((4,), 3, dtype=ts.float64, output_shape=(4,))
        with self.assertRaises(TypeError):
            execute_arange(0, 1, 4, dtype=ts.float64, output_shape=(4,))
        with self.assertRaises(TypeError):
            execute_linspace(0.0, 1.0, 4, dtype=ts.float64, output_shape=(4,))

    def test_creation_dispatch_matches_the_public_constructors(self):
        from tensors.backend import execute_arange, execute_eye, execute_full

        self.assertEqual(
            list(execute_full((2, 3), 2.5, dtype=ts.float64).buffer),
            ts.full((2, 3), 2.5).tolist(),
        )
        self.assertEqual(
            list(execute_arange(0, 1, 8, dtype=ts.float64).buffer),
            ts.arange(0, 8, 1, dtype=ts.float64).tolist(),
        )
        self.assertEqual(
            list(execute_eye(3, 3, 0, dtype=ts.float64).buffer),
            ts.eye(3).tolist(),
        )


class InitializerSuiteShapeTests(unittest.TestCase):
    """Fan-scaled initializers are only built for shapes that have a fan.

    Every sampling initializer scales its variance by the shape's fan, which a
    one-dimensional parameter does not have. The suite has to decline such a
    shape rather than build cases that raise.
    """

    def setUp(self):
        self.previous_backend = ts.get_backend()

    def tearDown(self):
        ts.set_backend(self.previous_backend)

    def test_one_dimensional_shapes_are_declined(self):
        from benchmarks.perfmap.suites.initializers import _initializer_cases

        with self.assertRaises(Unsupported):
            _initializer_cases("python", (64,), "float64")

    def test_two_dimensional_shapes_build_runnable_cases(self):
        from benchmarks.perfmap.suites.initializers import _initializer_cases

        cases = _initializer_cases("python", (8, 4), "float64")

        self.assertGreater(len(cases), 0)
        for case in cases:
            with self.subTest(case=case.name):
                result = case.run()
                self.assertEqual(result.shape, (8, 4))

    def test_fan_initializers_reject_a_one_dimensional_shape(self):
        """The suite's guard matches what the initializers actually do."""
        for name in ("he_normal", "xavier_uniform", "lecun_normal"):
            with self.subTest(initializer=name):
                with self.assertRaisesRegex(ValueError, "two dimensions"):
                    getattr(ts.init, name)((64,), dtype=ts.float64)


if __name__ == "__main__":
    unittest.main()
