import unittest

import tensors as ts


class TensorPowerTests(unittest.TestCase):
    def test_scalar_exponent_preserves_integer_dtype_for_non_negative_powers(self):
        base = ts.Tensor([2, 3], dtype=ts.int32)

        result = base**3

        self.assertIs(result.dtype, ts.int32)
        self.assertEqual(result.tolist(), [8, 27])

    def test_tensor_exponent_uses_elementwise_broadcasting(self):
        base = ts.Tensor([[1.0], [2.0]])
        exponent = ts.Tensor([2.0, 3.0])

        result = base**exponent

        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result.tolist(), [1.0, 1.0, 4.0, 8.0])

    def test_reverse_power_uses_tensor_as_the_exponent(self):
        exponent = ts.Tensor([1.0, 2.0, 3.0])

        result = 2.0**exponent

        self.assertEqual(result.tolist(), [2.0, 4.0, 8.0])

    def test_reverse_power_preserves_float32_exponent_dtype(self):
        exponent = ts.Tensor([1.0, 2.0, 3.0], dtype=ts.float32)

        result = 2.0**exponent

        self.assertIs(result.dtype, ts.float32)
        self.assertEqual(result.tolist(), [2.0, 4.0, 8.0])

    def test_package_power_function_delegates_to_operator(self):
        result = ts.pow(ts.Tensor([2.0, 3.0]), 2.0)

        self.assertEqual(result.tolist(), [4.0, 9.0])

    def test_fractional_power_of_negative_values_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "real-valued"):
            _ = ts.Tensor([-1.0]) ** 0.5


class ScalarBasePowerTests(unittest.TestCase):
    """A scalar base raised to a tensor exponent, under rule S-p.

    The result dtype comes from the declared dtypes alone
    (`docs/arithmetic-semantics.md` §12.5.3). An integer scalar base converts
    into the exponent tensor's integer dtype, so the result is that integer
    dtype whatever the exponent's values happen to be.

    These tests previously asserted that a negative exponent anywhere promoted
    the whole result to `float64`. That rule inspected element values, which
    §6.4 forbids and D6 removed: the same declared dtypes now always give the
    same result dtype. What a negative integer exponent should *do* is settled
    by D4 (§12.4.2) and is covered, deferred, in
    `tests/operations/arithmetic/test_power_dtype.py`.
    """

    def setUp(self):
        self.previous_backend = ts.get_backend()

    def tearDown(self):
        ts.set_backend(self.previous_backend)

    #: Large enough to clear the array-backend size policy, so the accelerated
    #: kernel runs rather than the Python reference.
    ACCELERATED_SIZE = 64

    def _backends(self):
        return ts.available_backends()

    def test_integer_exponent_dtype_does_not_depend_on_its_values(self):
        """Replaces a test asserting value-dependent promotion to float64."""
        for backend in self._backends():
            for dtype in (ts.int32, ts.int64):
                for values in ([1, 2, 3, 0], [1, -2, 3, 0], [-5, -5, -5, -5]):
                    with self.subTest(
                        backend=backend, dtype=dtype.name, exponent=values
                    ):
                        with ts.use_backend(backend):
                            exponent = ts.Tensor(values, dtype=dtype)
                            self.assertIs((2**exponent).dtype, dtype)

    def test_non_negative_exponent_values_are_unchanged(self):
        for backend in self._backends():
            for dtype in (ts.int32, ts.int64):
                with self.subTest(backend=backend, dtype=dtype.name):
                    with ts.use_backend(backend):
                        exponent = ts.Tensor([1, 2, 3, 0], dtype=dtype)
                        result = 2**exponent

                    self.assertIs(result.dtype, dtype)
                    self.assertEqual(result.tolist(), [2, 4, 8, 1])

    def test_non_negative_exponent_values_are_unchanged_when_accelerated(self):
        pattern = [1, 2, 3, 0]
        repeats = self.ACCELERATED_SIZE // len(pattern)
        for backend in self._backends():
            for dtype in (ts.int32, ts.int64):
                with self.subTest(backend=backend, dtype=dtype.name):
                    with ts.use_backend(backend):
                        exponent = ts.Tensor(
                            pattern * repeats,
                            shape=(self.ACCELERATED_SIZE,),
                            dtype=dtype,
                        )
                        result = 2**exponent

                    self.assertIs(result.dtype, dtype)
                    self.assertEqual(result.tolist(), [2, 4, 8, 1] * repeats)

    def test_non_negative_integer_exponent_keeps_the_integer_dtype(self):
        for backend in self._backends():
            with self.subTest(backend=backend):
                with ts.use_backend(backend):
                    exponent = ts.Tensor([0, 1, 2, 3] * 16, dtype=ts.int32)
                    result = 2**exponent

                self.assertIs(result.dtype, ts.int32)
                self.assertEqual(result.tolist(), [1, 2, 4, 8] * 16)

    def test_float_scalar_base_promotes_an_integer_exponent(self):
        for backend in self._backends():
            with self.subTest(backend=backend):
                with ts.use_backend(backend):
                    exponent = ts.Tensor([1, -2, 3] * 16, dtype=ts.int32)
                    result = 2.0**exponent

                self.assertIs(result.dtype, ts.float64)
                self.assertEqual(result.tolist(), [2.0, 0.25, 8.0] * 16)

    def test_every_backend_agrees_on_a_negative_integer_exponent(self):
        """The promoted dtype is exact; the values agree within tolerance.

        A device ``pow`` can land one unit in the last place away from the
        host result, so only the dtype is compared exactly.
        """
        exponent_values = [3, -1, 0, -4, 2, -3] * 16

        def evaluate(backend):
            with ts.use_backend(backend):
                exponent = ts.Tensor(exponent_values, dtype=ts.int64)
                result = 2**exponent
                return result.dtype, result.tolist()

        expected_dtype, expected = evaluate("python")
        for backend in self._backends():
            with self.subTest(backend=backend):
                dtype, values = evaluate(backend)
                self.assertIs(dtype, expected_dtype)
                for actual, expected_value in zip(values, expected):
                    self.assertAlmostEqual(actual, expected_value)

    def test_zero_base_rejects_a_negative_integer_exponent(self):
        for backend in self._backends():
            with self.subTest(backend=backend):
                with (
                    ts.use_backend(backend),
                    self.assertRaises((ValueError, ZeroDivisionError)),
                ):
                    _ = 0 ** ts.Tensor([-1] * 64, dtype=ts.int32)


if __name__ == "__main__":
    unittest.main()
