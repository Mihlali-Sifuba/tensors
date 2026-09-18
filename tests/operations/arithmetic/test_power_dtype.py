"""Result dtypes and scalar conversion for power (D6, specification §12.5).

Expectations come from :mod:`_spec`, which derives the §6.2 promotion table
from its stated principles, and from the scalar rules written out in §6.5 and
§12.5.3. Nothing here is taken from NumPy or from any backend's behaviour.

This milestone covers dtype determination only. Where a case cannot be
completed without a later milestone — integer wraparound (D3), the
negative-exponent domain rule (D4), or the floating exceptional values (D2) —
the test asserts the dtype decision and the value assertion is deferred, with
the dependency named.
"""

import unittest

import tensors as ts

from . import _spec
from ._support import BACKENDS, DTYPE, ArithmeticTestCase, tensor

CAST = _spec.CAST


class TensorTensorResultDtype(ArithmeticTestCase):
    """§12.5.1 — the §6.2 table, applied to base and exponent."""

    def test_all_49_combinations(self):
        for base_name in _spec.DTYPES:
            for exponent_name in _spec.DTYPES:
                expected = _spec.promote(base_name, exponent_name)
                with self.subTest(base=base_name, exponent=exponent_name):
                    base = tensor(base_name, [2])
                    exponent = tensor(exponent_name, [2])
                    if expected == CAST:
                        with self.assertRaises(TypeError):
                            _ = base**exponent
                        continue
                    self.assertIs((base**exponent).dtype, DTYPE[expected])

    def test_every_int64_mixed_floating_combination_requires_a_cast(self):
        """The four `cast` cells §12.5.1 names explicitly."""
        for other in ("float32", "float64"):
            for base_name, exponent_name in (("int64", other), (other, "int64")):
                with self.subTest(base=base_name, exponent=exponent_name):
                    self.assertEqual(_spec.promote(base_name, exponent_name), CAST)
                    with self.assertRaises(ts.dtype.DtypePromotionError):
                        _ = tensor(base_name, [2]) ** tensor(exponent_name, [2])

    def test_an_explicit_cast_resolves_the_refusal(self):
        base = tensor("float32", [2.0])
        exponent = tensor("int64", [3])
        with self.assertRaises(TypeError):
            _ = base**exponent
        self.assertIs((base ** exponent.astype(ts.float32)).dtype, ts.float32)


class ResultDtypeIgnoresValues(ArithmeticTestCase):
    """§6.4 — promotion never inspects an element."""

    def test_exponent_values_do_not_change_the_result_dtype(self):
        base = tensor("int32", [2, 2, 2, 2])
        for values in ([1, 2, 3, 4], [1, -2, 3, 0], [-9, -9, -9, -9], [0, 0, 0, 0]):
            with self.subTest(exponent=values):
                exponent = tensor("int32", values)
                self.assertIs((base**exponent).dtype, ts.int32)

    def test_base_values_do_not_change_the_result_dtype(self):
        exponent = tensor("int32", [2, 2])
        for values in ([2, 3], [-2, -3], [0, 0]):
            with self.subTest(base=values):
                self.assertIs((tensor("int32", values) ** exponent).dtype, ts.int32)

    def test_reflected_exponent_values_do_not_change_the_result_dtype(self):
        for values in ([1, 2], [1, -2], [0, 0]):
            with self.subTest(exponent=values):
                self.assertIs((2 ** tensor("int32", values)).dtype, ts.int32)

    @unittest.skipUnless("cuda" in BACKENDS, "CUDA backend is not installed")
    def test_resolving_a_dtype_reads_no_device_memory(self):
        """A dtype decision must not pull a device tensor back to the host.

        Scoped to resolution. Evaluating an *integer* power on CUDA still
        transfers, because the CUDA kernel declines integer operands and the
        Python reference then reads them; that is the capability gap left to
        the kernel milestone, not a dtype decision.
        """
        from tensors.dtype import resolve_power, resolve_power_scalar_base

        with ts.use_backend("cuda"):
            base = ts.full((4096,), 2, dtype=ts.int32)
            exponent = ts.full((4096,), 2, dtype=ts.int32)
            with self._counting_device_reads() as reads:
                resolve_power(base.dtype, exponent)
                resolve_power(base.dtype, 2)
                resolve_power(base.dtype, 2.0)
                resolve_power_scalar_base(2, exponent.dtype)
                resolve_power_scalar_base(2.5, exponent.dtype)
        self.assertEqual(reads.count, 0)

    @unittest.skipUnless("cuda" in BACKENDS, "CUDA backend is not installed")
    def test_a_floating_power_transfers_nothing_end_to_end(self):
        """Where no kernel declines, the whole operation stays on the device."""
        with ts.use_backend("cuda"):
            base = ts.full((4096,), 2.0, dtype=ts.float64)
            with self._counting_device_reads() as reads:
                result = base**2.0
            self.assertIs(result.dtype, ts.float64)
        self.assertEqual(reads.count, 0)

    @staticmethod
    def _counting_device_reads():
        """Count materialisations of a tensor's host-side element list."""
        import contextlib

        import tensors.tensor as tensor_module

        class Counter:
            count = 0

        @contextlib.contextmanager
        def counting():
            counter = Counter()
            original = tensor_module.Tensor._data.fget

            def counted(self):
                counter.count += 1
                return original(self)

            tensor_module.Tensor._data = property(counted)
            try:
                yield counter
            finally:
                tensor_module.Tensor._data = property(original)

        return counting()


class TensorScalarExponent(ArithmeticTestCase):
    """§12.5.2 — S1 to S4 with the tensor's dtype as the target."""

    def test_s1_integer_scalar_into_an_integer_tensor(self):
        for name in _spec.INTEGER_DTYPES:
            with self.subTest(dtype=name):
                self.assertIs((tensor(name, [2]) ** 2).dtype, DTYPE[name])

    def test_s1_rejects_a_scalar_outside_the_tensor_range(self):
        for name, value in (("uint8", 300), ("uint8", -1), ("int8", 128)):
            with self.subTest(dtype=name, scalar=value):
                self.assertRaisesConversion(lambda: tensor(name, [2]) ** value)

    def test_s2_integral_float_converts_to_the_integer_dtype(self):
        for name in _spec.INTEGER_DTYPES:
            with self.subTest(dtype=name):
                result = tensor(name, [2]) ** 2.0
                self.assertIs(result.dtype, DTYPE[name])

    def test_s2_rejects_a_non_integral_float_for_an_integer_tensor(self):
        for name in _spec.INTEGER_DTYPES:
            with self.subTest(dtype=name):
                self.assertRaisesConversion(lambda: tensor(name, [4]) ** 0.5)

    def test_s3_float_scalar_with_a_floating_tensor(self):
        for name in _spec.FLOAT_DTYPES:
            with self.subTest(dtype=name):
                self.assertIs((tensor(name, [2.0]) ** 0.5).dtype, DTYPE[name])

    def test_s4_integer_scalar_with_a_floating_tensor(self):
        for name in _spec.FLOAT_DTYPES:
            with self.subTest(dtype=name):
                self.assertIs((tensor(name, [2.0]) ** 2).dtype, DTYPE[name])

    def test_s4_rejects_an_integer_a_floating_dtype_cannot_hold_exactly(self):
        self.assertRaisesConversion(lambda: tensor("float32", [2.0]) ** (2**24 + 1))

    def test_bool_is_not_a_numeric_scalar(self):
        self.assertRaisesConversion(lambda: tensor("float64", [2.0]) ** True)

    def test_a_scalar_never_widens_the_result(self):
        """§6.5 — no scalar promotes the result to a wider dtype."""
        for name in tuple(_spec.INTEGER_DTYPES) + tuple(_spec.FLOAT_DTYPES):
            for scalar in (0, 1, 2):
                with self.subTest(dtype=name, scalar=scalar):
                    self.assertIs((tensor(name, [2]) ** scalar).dtype, DTYPE[name])


class ScalarBaseExponentTensor(ArithmeticTestCase):
    """§12.5.3 — rule S-p, where the scalar is the base."""

    def test_integer_base_takes_the_integer_exponent_dtype(self):
        for name in _spec.INTEGER_DTYPES:
            with self.subTest(dtype=name):
                self.assertIs((2 ** tensor(name, [3])).dtype, DTYPE[name])

    def test_integer_base_outside_the_exponent_range_is_rejected(self):
        self.assertRaisesConversion(lambda: (-2) ** tensor("uint8", [3]))
        self.assertRaisesConversion(lambda: 300 ** tensor("uint8", [2]))

    def test_float_base_over_an_integer_exponent_defaults_to_float64(self):
        for name in ("uint8", "int8", "int16", "int32"):
            with self.subTest(dtype=name):
                self.assertIs((2.5 ** tensor(name, [2])).dtype, ts.float64)

    def test_float_base_over_an_int64_exponent_requires_a_cast(self):
        """S-p does not bypass the §6.2 restriction: float64 with int64 casts."""
        with self.assertRaises(ts.dtype.DtypePromotionError):
            _ = 2.5 ** tensor("int64", [2])

    def test_any_scalar_base_takes_a_floating_exponent_dtype(self):
        for name in _spec.FLOAT_DTYPES:
            for base in (2, 2.5):
                with self.subTest(dtype=name, base=base):
                    self.assertIs((base ** tensor(name, [3.0])).dtype, DTYPE[name])

    def test_s3_rejects_a_base_that_would_round_to_infinity(self):
        self.assertRaisesConversion(lambda: 1e300 ** tensor("float32", [1.0]))

    def test_s4_rejects_a_base_the_floating_exponent_dtype_cannot_hold(self):
        self.assertRaisesConversion(lambda: (2**24 + 1) ** tensor("float32", [1.0]))

    def test_bool_base_is_rejected(self):
        self.assertRaisesConversion(lambda: True ** tensor("float64", [1.0]))


class VariableOperands(ArithmeticTestCase):
    """Variables carry a declared dtype and promote like any typed operand."""

    def test_integer_variable_raised_to_an_integer_variable(self):
        """Regression: this raised TypeError comparing a Variable with an int.

        The old resolution needed element values, so it compared the operand
        against zero and crashed on anything that was not a Tensor or a number.
        """
        base = ts.Variable(tensor("int32", [2]), requires_grad=False)
        exponent = ts.Variable(tensor("int32", [3]), requires_grad=False)
        result = base**exponent
        self.assertIs(result.dtype, ts.int32)

    def test_every_integer_variable_pair_resolves(self):
        for name in _spec.INTEGER_DTYPES:
            with self.subTest(dtype=name):
                base = ts.Variable(tensor(name, [2]), requires_grad=False)
                exponent = ts.Variable(tensor(name, [2]), requires_grad=False)
                self.assertIs((base**exponent).dtype, DTYPE[name])

    def test_floating_variable_pair_promotes_by_the_table(self):
        for base_name in _spec.FLOAT_DTYPES:
            for exponent_name in _spec.FLOAT_DTYPES:
                with self.subTest(base=base_name, exponent=exponent_name):
                    base = ts.Variable(tensor(base_name, [2.0]))
                    exponent = ts.Variable(tensor(exponent_name, [2.0]))
                    expected = _spec.promote(base_name, exponent_name)
                    self.assertIs((base**exponent).dtype, DTYPE[expected])

    def test_variable_scalar_exponent_follows_the_scalar_rules(self):
        variable = ts.Variable(tensor("int32", [2]), requires_grad=False)
        self.assertIs((variable**2).dtype, ts.int32)
        self.assertIs((variable**2.0).dtype, ts.int32)
        self.assertRaisesConversion(lambda: variable**0.5)

    def test_reflected_variable_power_follows_s_p(self):
        variable = ts.Variable(tensor("int32", [3]), requires_grad=False)
        self.assertIs((2**variable).dtype, ts.int32)
        self.assertIs((2.5**variable).dtype, ts.float64)
        with self.assertRaises(ts.dtype.DtypePromotionError):
            _ = 2.5 ** ts.Variable(tensor("int64", [3]), requires_grad=False)


class DeferredToLaterMilestones(ArithmeticTestCase):
    """Cases whose dtype is settled by D6 but whose value needs a later rule."""

    @unittest.expectedFailure
    def test_negative_integer_exponent_raises(self):
        """Deferred to D4 (§12.4.2).

        D6 fixes the result dtype at the integer dtype, so the reciprocal has
        nowhere to go. D4 makes this raise ``ValueError``; until then the
        evaluation truncates toward zero, which this asserts against so the
        milestone is visible.
        """
        with self.assertRaises(ValueError):
            _ = tensor("int32", [2]) ** -1

    @unittest.expectedFailure
    def test_reflected_negative_integer_exponent_raises(self):
        """Deferred to D4 (§12.4.2), reflected form."""
        with self.assertRaises(ValueError):
            _ = 2 ** tensor("int32", [-2])

    def test_the_dtype_half_is_already_correct(self):
        """What D6 does settle for those cases: the dtype, from declarations."""
        self.assertIs((tensor("int32", [2]) ** -1).dtype, ts.int32)
        self.assertIs((2 ** tensor("int32", [-2])).dtype, ts.int32)


if __name__ == "__main__":
    unittest.main()
